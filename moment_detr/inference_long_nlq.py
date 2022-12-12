import logging
import os

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.basic_utils import AverageMeter
import h5py

from moment_detr.config import TestOptions
from moment_detr.inference import setup_model
from moment_detr.start_end_dataset import collate_fn_replace_corrupted, prepare_batch_inputs
from moment_detr.start_end_dataset_long_nlq import StartEndDatasetLong
from moment_detr.span_utils import span_cxw_to_xx
from standalone_eval.utils import compute_temporal_iou_batch_paired
from utils.basic_utils import load_jsonl

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def get_data(opt):
    lang_path = os.path.join(opt.t_feat_dir, opt.lang_feat_path)
    logger.info(f'LOADING: {lang_path}')
    lang_feats = h5py.File(lang_path, 'r')
    video_path = os.path.join(opt.v_feat_dirs[0], 'CLIP_L14_frames_features_5fps.h5')
    logger.info(f'LOADING: {video_path}')
    video_feats = h5py.File(video_path, 'r')
    anno_path = os.path.join(opt.t_feat_dir, 'annotations/MAD_test.json')
    logger.info(f'LOADING: {anno_path}')
    annos = load_jsonl(anno_path)[0]
    return lang_feats, video_feats, annos


def prepare_inputs(qid, anno, window, lang_feats, video_feats, device, opt):
    model_inputs = dict()
    model_inputs['src_txt'] = torch.tensor(np.array(lang_feats[qid])).view(1, -1, opt.t_feat_dim).to(device).type(
        torch.float32)

    vid = torch.tensor(video_feats[anno['movie']][window[0]:window[1]])
    ctx_l = len(vid)
    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
    model_inputs['src_vid'] = torch.cat([vid, tef], dim=1).view(1, -1, opt.v_feat_dim).to(device)

    model_inputs['src_txt_mask'] = torch.ones_like(model_inputs['src_txt'][..., 0]).to(device)
    model_inputs['src_vid_mask'] = torch.ones_like(model_inputs['src_vid'][..., 0]).to(device)

    return model_inputs


def moment_inside_window(anno, window):
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
    lang_feats, video_feats, annos = get_data(opt)

    with torch.no_grad():
        logger.info("Generate submissions")
        model.eval()
        criterion.eval()
        loss_meters = defaultdict(AverageMeter)
        window_length = 150

        for qid in tqdm(annos.keys()):
            len_movie = video_feats[annos[qid]['movie']].shape[0]
            pred_spans, pred_cls, foreground = [], [], []

            for idx, window in enumerate(range(0, len_movie, window_length // 2)):
                sample_window = [window, min(len_movie, window + window_length)]
                model_inputs = prepare_inputs(qid=qid,
                                              anno=annos[qid],
                                              window=sample_window,
                                              lang_feats=lang_feats,
                                              video_feats=video_feats,
                                              device=opt.device,
                                              opt=opt)

                if moment_inside_window(anno=annos[qid], window=sample_window):
                    foreground.append(idx)

                outputs = model(**model_inputs)

                pred_spans.append((span_cxw_to_xx(outputs["pred_spans"]) * annos[qid]['movie_duration'])[0, ...])
                pred_cls.append(outputs['pred_cls'][0][0][0])


if __name__ == '__main__':
    start_inference_long_nlq()
