import logging
import pprint
import numpy as np
from tqdm import tqdm
import functools

from moment_detr.config import TestOptions
from moment_detr.inference import setup_model
from moment_detr.span_utils import span_cxw_to_xx
from moment_detr.long_nlq_dataset import LongNlqDataset, collate_fn_replace_corrupted
from standalone_eval.eval import long_nlq_metrics

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


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
        collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset)
        long_nlq_loader = DataLoader(
            long_nlq_dataset,
            collate_fn=collate_fn,
            batch_size=8,
            num_workers=8,
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

                preds[qid[i]] = {
                    'pred_spans': span_cxw_to_xx(outputs["pred_spans"]) * _target['anno']['movie_duration'],
                    'pred_cls': outputs['pred_cls'][:, 0]}

            _pred = list( map(preds.get, qid) )
            metrics = long_nlq_metrics(_pred, target, qid=qid, metrics=metrics)

            if opt.debug:
                break

        avg_metrics = {'accuracy': np.mean([metrics[key]['accuracy'] for key in metrics.keys()]),
                       'recall': np.mean([metrics[key]['recall'] for key in metrics.keys()]),
                       'precision': np.mean([metrics[key]['precision'] for key in metrics.keys()])
                       }
        logger.info("\naverage metrics\n{}".format(pprint.pformat(avg_metrics, indent=4)))


if __name__ == '__main__':
    start_inference_long_nlq()
