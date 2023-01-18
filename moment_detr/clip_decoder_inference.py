import logging
import pprint
import numpy as np
from tqdm import tqdm
import functools
import os
import traceback
import random
from collections import Counter

from moment_detr.config import TestOptions
from moment_detr.model import build_model
from moment_detr.span_utils import span_cxw_to_xx
from moment_detr.long_nlq_dataset import LongNlqDataset, collate_fn_replace_corrupted
from standalone_eval.eval import eval_submission, sort_pos_predicted, remove_zero_predictions
from utils.basic_utils import save_json
from moment_detr.clip_similarity import clip_filter_proposals
from moment_detr.clip_training import setup_training, data_to_device

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def sort_spans(i, outputs, windows, prob):
    scaled_spans = torch.zeros_like(outputs["pred_spans"])
    for idx, output in enumerate(outputs["pred_spans"]):
        start_window = windows[i][idx, 0]
        window_length = windows[i][idx, 1] - windows[i][idx, 0]
        for idxx, out in enumerate(output):
            if not torch.isclose(input=out.cpu(), other=torch.tensor([0.0, 0.0]).float(), atol=1e-4).all():
                scaled_spans[idx, idxx] = start_window + span_cxw_to_xx(out) * window_length

    pred_spans = torch.cat([scaled_spans, prob], dim=2).tolist()
    ranked_spans = [sorted(_s, key=lambda x: x[2], reverse=True) for _s in pred_spans]
    return ranked_spans


def clip_decoder_inference(opt=None,
                           model=None,
                           criterion=None,
                           dataloader=None,
                           tb_writer=None):
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    pooling_topk = 7
    topk = 10

    if model is None:
        model, criterion = build_model(opt)
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    set_seed(opt.seed)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)
    save_submission_filename = "inference_long_nlq_preds.jsonl"

    with torch.no_grad():
        logger.info("Generate submissions")
        model.eval()
        criterion.eval()

        preds, metrics, clip_metrics = {}, {}, {}
        for batch in tqdm(dataloader):
            target, data, qid, windows, clip_metrics = batch
            data = data_to_device(data, opt)

            for i in range(len(data)):
                _target = target[i]
                _data = data[i]

                outputs = model(**_data)

                prob = F.softmax(outputs["pred_logits"], -1)[..., 0, None]
                sorted_spans = sort_spans(i, outputs, windows, prob)

                _pred = {'pred_spans': sorted_spans}

                # R@K should be calculated for windows, which are predicted foreground
                # Retrieval metrics are calculated on foreground windows only
                _ground_truth = [{'qid': qid[i],
                                  'relevant_windows': [_target['anno']['ext_timestamps']],
                                  'is_foreground': bool(_is_foreground)} for _is_foreground in
                                 _target['is_foreground']]

                _submission = [{'qid': qid[i],
                                'pred_relevant_windows': _span} for idx, _span in
                               enumerate(_pred['pred_spans'])]

                metrics[qid[i]] = eval_submission(_submission, _ground_truth,
                                                  verbose=False,
                                                  is_long_nlq=True,
                                                  length_ranges=[[0, 200]],
                                                  range_names=['full'],
                                                  is_nms=True,
                                                  iou_thds=[0.1, 0.3, 0.5],
                                                  top_ks=[1, 2, 5, 10, 50, 100],
                                                  match_number=False)

                # _submission, _ground_truth = remove_zero_predictions(_submission, _ground_truth)
                _submission, _ground_truth = sort_pos_predicted(_submission, _ground_truth)

                preds[qid[i]] = {'_ground_truth': _ground_truth,
                                 '_submission': _submission}

            if opt.debug:
                break

        eval_postprocessing(metrics, preds,
                            opt=opt,
                            save_submission_filename=save_submission_filename,
                            tb_writer=tb_writer)
        eval_postprocessing_clip(clip_metrics,
                                 opt,
                                 save_submission_filename='clip_inference.jsonl')


def eval_postprocessing_clip(metrics, opt, save_submission_filename):
    for k_key in metrics.keys():
        for m_key in metrics[k_key].keys():
            metrics[k_key][m_key] = np.mean(metrics[k_key][m_key]).round(4)

    logger.info("\naverage metrics\n{}".format(pprint.pformat(metrics, indent=4)))
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)


def eval_postprocessing(metrics, preds, opt, save_submission_filename,tb_writer=None, epoch_i=None):
    mr_metrics = [metrics[key]['brief'] for key in metrics.keys()]

    # mr metrics return -1, if there are no foreground predictions or no data is in the selected window range (length_ranges)
    avg_mr_metrics = {k: np.array([m[k] if m[k] != -1 else 0 for m in mr_metrics]).mean() for k in mr_metrics[0].keys()}
    percentage_no_intersection = {k: np.array([1 for m in mr_metrics if m[k] == -1]).sum() / len(mr_metrics) for k in
                                  mr_metrics[0].keys()}

    avg_metrics = {'avg_mr_metrics': avg_mr_metrics,
                   'percentage_no_intersection': percentage_no_intersection}

    logger.info("\naverage metrics\n{}".format(pprint.pformat(avg_metrics, indent=4)))
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json(avg_metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    save_json(preds, submission_path)

    if tb_writer is not None:
        for k, v in avg_mr_metrics.items():
            tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i + 1)

    return avg_metrics


if __name__ == '__main__':
    model, criterion, optimizer, lr_scheduler, opt, data_loader, long_nlq_dataset_val\
        = setup_training(mode='val')

    clip_decoder_inference(model=model,
                           criterion=criterion,
                           opt=opt,
                           dataloader=data_loader)
