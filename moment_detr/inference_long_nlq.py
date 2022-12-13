import logging
import os
import pprint
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.basic_utils import AverageMeter
import h5py

from moment_detr.span_utils import temporal_intersection_over_pred
from moment_detr.config import TestOptions
from moment_detr.inference import setup_model
from moment_detr.span_utils import span_cxw_to_xx
from standalone_eval.eval import compute_cls_acc
from utils.basic_utils import load_jsonl

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class LongNlqDataset(Dataset):
    def __init__(self, opt, stride=0.5,
                 window_length=150, test_file='MAD_test.json',
                 video_fatures_path='CLIP_L14_frames_features_5fps.h5'):
        self.opt = opt
        self.lang_path = os.path.join(opt.t_feat_dir, opt.lang_feat_path)
        self.video_path = os.path.join(opt.v_feat_dirs[0], video_fatures_path)
        self.anno_path = os.path.join(opt.t_feat_dir, 'annotations', test_file)
        data = self.get_data(opt)
        self.lang_feats = data[0]
        self.video_feats = data[1]
        self.annos = data[2]
        self.keys = list(data[2].keys())
        self.stride = stride
        self.window_length = window_length

    def get_data(self, opt):
        logger.info(f'LOADING: {self.lang_path}')
        lang_feats = h5py.File(self.lang_path, 'r')
        logger.info(f'LOADING: {self.video_path}')
        video_feats = h5py.File(self.video_path, 'r')
        logger.info(f'LOADING: {self.anno_path}')
        annos = load_jsonl(self.anno_path)[0]
        return [lang_feats, video_feats, annos]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        qid = self.keys[index]
        anno = self.annos[qid]
        len_movie = self.video_feats[self.annos[qid]['movie']].shape[0]
        foregrounds, model_inputs = [], []
        # for start_window in range(0, len_movie, self.window_length // 2):
        # window = [start_window, min(start_window + self.window_length, len_movie)]

        model_input = self.prepare_inputs(qid=qid,
                                          anno=anno)
        model_input = self.unfold_expand(qid=qid,
                                         model_input=model_input)
        target = self.get_foreground(qid=qid,
                                         model_input=model_input,
                                         anno=anno)

        # is_foreground = self.moment_inside_window(anno, window)
        # foregrounds.append({'is_foreground': is_foreground})

        return model_inputs, foregrounds

    def get_foreground(self, qid, model_input, anno):
        target = {}
        vid_shape = model_input[qid]['src_vid'].shape
        idx_start = torch.arange(0, int(vid_shape[0] / 2 * vid_shape[1]), self.window_length // 2).view(-1, 1)
        idx_end = (torch.arange(0, int(vid_shape[0] / 2 * vid_shape[1]),
                                self.window_length // 2) + self.window_length).view(-1, 1)
        window_idx = torch.cat([idx_start, idx_end], dim=1) / 5
        moment = torch.tensor(anno['ext_timestamps']).reshape(-1, 2)
        intersection = temporal_intersection_over_pred(moment, window_idx)
        intersection_idx = torch.where(intersection > 0)[-1]
        foreground = torch.zeros(vid_shape[0])
        foreground[intersection_idx] = 1
        target[qid] = dict(is_foreground=foreground)
        return target

    def unfold_expand(self, qid, model_input):
        model_input[qid]['src_vid'] = model_input[qid]['src_vid'].unfold(0, self.window_length,
                                                                         self.window_length // 2).reshape(-1,
                                                                                                          self.window_length,
                                                                                                          self.opt.v_feat_dim - 2)
        model_input = self.cat_tef(qid=qid,
                                   model_input=model_input)
        model_input[qid]['src_txt'] = model_input[qid]['src_txt'].expand(model_input[qid]['src_vid'].shape[0],
                                                                         *model_input[qid]['src_txt'].shape[1:])
        model_input[qid]['src_vid_mask'] = torch.ones(model_input[qid]['src_vid'].shape[0:2])
        model_input[qid]['src_txt_mask'] = torch.ones(model_input[qid]['src_txt'].shape[0:2])
        return model_input

    def prepare_inputs(self, qid, anno):
        model_inputs = dict()

        model_inputs[qid] = {'src_vid': torch.tensor(np.array(self.video_feats[anno['movie']])),
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

    def moment_inside_window(self, anno, window):
        window = np.divide(window, 5)
        moment = anno['ext_timestamps']
        if window[0] < moment[0] < window[1] or window[0] < moment[1] < window[1]:
            return True
        else:
            return False


def start_inference_long_nlq():
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    model, criterion, _, _ = setup_model(opt)
    save_submission_filename = "inference_long_nlq_preds.jsonl"

    with torch.no_grad():
        logger.info("Generate submissions")
        model.eval()
        criterion.eval()
        window_length = 150

        metrics = {}
        long_nlq_dataset = LongNlqDataset(opt)
        long_nlq_loader = DataLoader(
            long_nlq_dataset,
            batch_size=opt.eval_bsz,
            num_workers=opt.num_workers,
            shuffle=False,
            pin_memory=opt.pin_memory
        )
        for batch in tqdm(long_nlq_loader):
            len_movie = video_feats[annos[qid]['movie']].shape[0]
            pred_spans, submission, ground_truth = [], [], []

            len_movie_batched = int(len_movie / opt.eval_bsz)
            for idx, window in enumerate(range(0, len_movie_batched, window_length // 2)):
                sample_window = [window, min(len_movie, window + window_length)]

                ground_truth.append({'is_foreground': is_foreground})

                outputs = model(**model_inputs)

                pred_spans.append((span_cxw_to_xx(outputs["pred_spans"]) * annos[qid]['movie_duration'])[0, ...])
                submission.append({'pred_cls': float(outputs['pred_cls'][0][0][0])})

            acc, recall, precision = compute_cls_acc(submission, ground_truth)
            metrics[qid] = {'accuracy': acc,
                            'recall': recall,
                            'precision': precision
                            }
            if opt.debug:
                break

        avg_metrics = {'accuracy': np.mean([metrics[key]['accuracy'] for key in metrics.keys()]),
                       'recall': np.mean([metrics[key]['recall'] for key in metrics.keys()]),
                       'precision': np.mean([metrics[key]['precision'] for key in metrics.keys()])
                       }
        logger.info("\naverage metrics\n{}".format(pprint.pformat(avg_metrics, indent=4)))


if __name__ == '__main__':
    start_inference_long_nlq()
