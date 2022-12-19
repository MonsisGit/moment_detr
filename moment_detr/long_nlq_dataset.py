import logging
import os
import numpy as np
import h5py
import random
import traceback

from moment_detr.span_utils import temporal_intersection_over_pred
from utils.basic_utils import load_jsonl

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class LongNlqDataset(Dataset):
    def __init__(self, opt, stride=0.5,
                 window_length=150,
                 video_fatures_path='CLIP_L14_frames_features_5fps.h5'):
        self.opt = opt
        self.data_ratio = opt.data_ratio_long_nlq
        self.lang_path = os.path.join(opt.t_feat_dir, opt.lang_feat_path)
        self.video_path = os.path.join(opt.v_feat_dirs[0], video_fatures_path)
        self.anno_path = os.path.join(opt.eval_path_long_nlq)
        data = self.get_data()
        self.lang_feats = data[0]
        self.video_feats = data[1]
        self.annos = data[2]
        self.keys = list(data[2].keys())
        self.stride = stride
        self.window_length = window_length
        self.cached_movie = {'movie': None,
                             'video_feats': None}

    def get_data(self):
        logger.info(f'LOADING: {self.lang_path}')
        lang_feats = h5py.File(self.lang_path, 'r')
        logger.info(f'LOADING: {self.video_path}')
        video_feats = h5py.File(self.video_path, 'r')
        logger.info(f'LOADING: {self.anno_path}')

        annos = load_jsonl(self.anno_path)[0]
        if self.data_ratio != 1:
            n_examples = int(len(annos) * self.data_ratio)
            if n_examples == 0:
                n_examples = 1
            annos = {key: annos[key] for key in list(annos.keys())[:n_examples]}

            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return [lang_feats, video_feats, annos]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        try:
            qid = self.keys[index]
            anno = self.annos[qid]

            model_input = self.prepare_inputs(qid=qid,
                                              anno=anno)

            windows, model_input = self.get_windows(model_input=model_input,
                                                    qid=qid)

            model_input = self.unfold_expand(qid=qid,
                                             model_input=model_input,
                                             windows=windows)

            target = self.get_foreground(windows=windows,
                                         qid=qid,
                                         model_input=model_input,
                                         anno=anno)

            return model_input, target
        except:
            traceback.format_stack()
            return None

    def get_windows(self, model_input, qid):
        vid_shape = model_input[qid]['src_vid'].shape
        idx_start = torch.arange(0, vid_shape[0] - self.window_length // 2, self.window_length // 2).clamp(
            max=vid_shape[0] - 1).view(-1, 1)
        idx_end = (idx_start + self.window_length)
        window = torch.cat([idx_start, idx_end], dim=1)

        pad_dim = idx_end.max() - vid_shape[0]
        pad_zeros = torch.zeros(pad_dim, self.opt.v_feat_dim - 2)
        model_input[qid]['src_vid'] = torch.cat([model_input[qid]['src_vid'], pad_zeros], dim=0)
        return window, model_input

    def get_foreground(self, windows, qid, model_input, anno):
        vid_shape = model_input[qid]['src_vid'].shape
        target = {}
        windows_s = torch.div(windows, 5)
        moment = torch.tensor(anno['ext_timestamps']).reshape(-1, 2)
        intersection = temporal_intersection_over_pred(moment, windows_s)
        intersection_idx = torch.where(intersection == intersection.max())[-1]
        foreground = torch.zeros(vid_shape[0])
        foreground[intersection_idx] = 1
        target[qid] = dict(is_foreground=foreground,
                           windows=windows_s,
                           anno=anno)
        return target

    def unfold_expand(self, qid, model_input, windows):
        # model_input[qid]['src_vid'] = model_input[qid]['src_vid'].unfold(0, self.window_length,
        #                                                                 self.window_length // 2)

        model_input[qid]['src_vid'] = torch.stack([model_input[qid]['src_vid'][w[0]:w[1]] for w in windows])

        model_input = self.cat_tef(qid=qid,
                                   model_input=model_input)
        model_input[qid]['src_txt'] = model_input[qid]['src_txt'].expand(model_input[qid]['src_vid'].shape[0],
                                                                         *model_input[qid]['src_txt'].shape[1:]).clone()
        model_input[qid]['src_vid_mask'] = torch.ones(model_input[qid]['src_vid'].shape[0:2])
        model_input[qid]['src_txt_mask'] = torch.ones(model_input[qid]['src_txt'].shape[0:2])
        return model_input

    def prepare_inputs(self, qid, anno):
        model_inputs = dict()

        if self.cached_movie['movie'] != anno['movie']:
            v_feats = np.array(self.video_feats[anno['movie']])
            self.cached_movie['movie'] = anno['movie']
            self.cached_movie['video_feats'] = v_feats
        else:
            v_feats = np.array(self.cached_movie['video_feats'])
        model_inputs[qid] = {'src_vid': torch.tensor(v_feats),
                             'src_txt': torch.tensor(np.array(self.lang_feats[qid])).view(1, -1,
                                                                                          self.opt.t_feat_dim).type(
                                 torch.float32)
                             }
        return model_inputs

    def cat_tef(self, qid, model_input):
        tef_st = torch.arange(0, self.window_length, 1.0) / self.window_length
        tef_ed = tef_st + 1.0 / self.window_length
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        tef = tef.expand(*model_input[qid]['src_vid'].shape[0:2], 2)
        model_input[qid]['src_vid'] = torch.cat([model_input[qid]['src_vid'], tef], dim=2)

        return model_input


def start_end_collate(batch):
    batch_keys = [list(b[1].keys())[0] for b in batch]
    batched_meta = [b[1][batch_keys[idx]] for idx, b in enumerate(batch)]
    batched_data = [b[0][batch_keys[idx]] for idx, b in enumerate(batch)]

    return batched_meta, batched_data, batch_keys


def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with another examples sampled randomly.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset which the DataLoader is loading.
            Specify it with functools.partial and pass the resulting partial function that only
            requires 'batch' argument to DataLoader's 'collate_fn' option.

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783

    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return start_end_collate(batch)
