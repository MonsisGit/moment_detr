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


def start_inference_long_nlq():
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    model, criterion = build_model(opt)
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

        long_nlq_dataset = LongNlqDataset(opt)
        collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset)
        long_nlq_loader = DataLoader(
            long_nlq_dataset,
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=opt.num_workers,
            shuffle=False,
            pin_memory=opt.pin_memory
        )
        preds, metrics = {}, {}
        for batch in tqdm(long_nlq_loader):
            target, data, qid = batch

            for i in range(len(data)):
                for key in data[i].keys():
                    data[i][key] = data[i][key].to(opt.device, non_blocking=True)

            for i in range(len(data)):
                _target = target[i]
                _data = data[i]
                outputs = model(**_data)

                prob = F.softmax(outputs["pred_logits"], -1)[..., 0, None]
                pred_spans = span_cxw_to_xx(outputs["pred_spans"]) * _target['anno']['movie_duration']
                pred_spans = torch.cat([pred_spans, prob], dim=2).tolist()
                ranked_spans = [sorted(_s, key=lambda x: x[2], reverse=True) for _s in pred_spans]

                try:
                    _pred = {'pred_spans': ranked_spans,
                             'pred_cls': outputs['pred_cls'][:, 0]}

                    # R@K should be calculated for windows, which are predicted foreground
                    # Retrieval metrics are calculated on foreground windows only
                    _ground_truth = [{'qid': qid[i],
                                      'relevant_windows': [_target['anno']['ext_timestamps']],
                                      'is_foreground': bool(_is_foreground)} for _is_foreground in
                                     _target['is_foreground']]

                    _submission = [{'qid': qid[i],
                                    'pred_relevant_windows': _span,
                                    'pred_cls': [float(_pred['pred_cls'][idx, 0])]} for idx, _span in
                                   enumerate(_pred['pred_spans'])]

                    metrics[qid[i]] = eval_submission(_submission, _ground_truth,
                                                      verbose=False,
                                                      is_long_nlq=True,
                                                      length_ranges=[[0, 200]],
                                                      range_names=['full'],
                                                      is_nms=True,
                                                      iou_thds=[0.1, 0.3, 0.5],
                                                      top_ks=[1, 5, 10, 50, 100])

                    #_submission, _ground_truth = remove_zero_predictions(_submission, _ground_truth)
                    _submission, _ground_truth = sort_pos_predicted(_submission, _ground_truth, n=100)

                    preds[qid[i]] = {'_ground_truth': _ground_truth,
                                     '_submission': _submission}
                except:
                    logger.info("Failed to calculate metrics for qid: {}".format(qid[i]))
                    logger.info(traceback.format_exc())

            if opt.debug:
                break

        eval_postprocessing(metrics, preds,
                            opt=opt,
                            save_submission_filename=save_submission_filename)


def eval_postprocessing(metrics, preds, opt, save_submission_filename):
    ret_metrics = [metrics[key]['full']['CLS'] for key in metrics.keys()]
    mr_metrics = [metrics[key]['brief'] for key in metrics.keys()]

    # retrieval metrics are calculated only on foreground windows
    avg_ret_metrics = {'accuracy': np.array([metric['accuracy'] for metric in ret_metrics]).mean().round(5),
                       'recall': np.array([metric['recall'] for metric in ret_metrics]).mean().round(5),
                       'precision': np.array([metric['precision'] for metric in ret_metrics]).mean().round(5)}

    # mr metrics return -1, if there are no foreground predictions or no data is in the selected window range (length_ranges)
    avg_mr_metrics = {k: np.array([m[k] if m[k] != -1 else 0 for m in mr_metrics]).mean() for k in mr_metrics[0].keys()}
    percentage_no_intersection = {k: np.array([1 for m in mr_metrics if m[k] == -1]).sum() / len(mr_metrics) for k in
                                  mr_metrics[0].keys()}

    avg_metrics = {'avg_mr_metrics': avg_mr_metrics,
                   'avg_ret_metrics': avg_ret_metrics,
                   'percentage_no_intersection': percentage_no_intersection}

    logger.info("\naverage metrics\n{}".format(pprint.pformat(avg_metrics, indent=4)))
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json(avg_metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    save_json(preds, submission_path)


if __name__ == '__main__':
    start_inference_long_nlq()
