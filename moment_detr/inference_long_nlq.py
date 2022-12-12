import logging
import os
import pprint
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.basic_utils import AverageMeter
import h5py

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
                 window_length=150):
        self.opt = opt
        data = self.get_data(opt)
        self.lang_feats = data[0]
        self.video_feats = data[1]
        self.annos = data[2]
        self.keys = list(data[2].keys())
        self.stride = stride
        self.window_length = window_length

    def get_data(self, opt):
        lang_path = os.path.join(opt.t_feat_dir, opt.lang_feat_path)
        logger.info(f'LOADING: {lang_path}')
        lang_feats = h5py.File(lang_path, 'r')
        video_path = os.path.join(opt.v_feat_dirs[0], 'CLIP_L14_frames_features_5fps.h5')
        logger.info(f'LOADING: {video_path}')
        video_feats = h5py.File(video_path, 'r')
        anno_path = os.path.join(opt.t_feat_dir, 'annotations/MAD_test.json')
        logger.info(f'LOADING: {anno_path}')
        annos = load_jsonl(anno_path)[0]
        return [lang_feats, video_feats, annos]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        qid = self.keys[index]
        anno = self.annos[qid]
        len_movie = self.video_feats[self.annos[qid]['movie']].shape[0]
        foregrounds, model_inputs = [], []
        for start_window in range(0, len_movie, self.window_length // 2):
            window = [start_window, min(start_window + self.window_length, len_movie)]
            model_input = self.prepare_inputs(qid=qid,
                                              anno=anno,
                                              window=window)
            model_inputs.append(model_input)
            is_foreground = self.moment_inside_window(anno, window)
            foregrounds.append({'is_foreground': is_foreground})
        return model_inputs, foregrounds

    def prepare_inputs(self, qid, anno, window):
        model_inputs = dict()
        model_inputs['src_txt'] = torch.tensor(np.array(self.lang_feats[qid])).view(1, -1, self.opt.t_feat_dim).to(
            self.opt.device).type(
            torch.float32)

        vid = torch.tensor(self.video_feats[anno['movie']][window[0]:window[1]])
        ctx_l = len(vid)
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        model_inputs['src_vid'] = torch.cat([vid, tef], dim=1).view(1, -1, self.opt.v_feat_dim).to(self.opt.device)

        model_inputs['src_txt_mask'] = torch.ones_like(model_inputs['src_txt'][..., 0]).to(self.opt.device)
        model_inputs['src_vid_mask'] = torch.ones_like(model_inputs['src_vid'][..., 0]).to(self.opt.device)

        return model_inputs

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
