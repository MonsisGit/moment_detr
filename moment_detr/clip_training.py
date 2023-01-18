import logging
import pprint
import numpy as np
from tqdm import tqdm
import functools
import os
import time
import json
from collections import defaultdict
import random

from moment_detr.config import BaseOptions
from moment_detr.inference import setup_model, eval_epoch
from utils.model_utils import count_parameters
from moment_detr.long_nlq_dataset import LongNlqDataset, collate_fn_replace_corrupted
from moment_detr.clip_similarity import clip_filter_proposals
from moment_detr.clip_decoder_inference import set_seed, clip_decoder_inference
from utils.basic_utils import dict_to_markdown, AverageMeter
from moment_detr.span_utils import span_xx_to_cxw

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def setup_training(mode='train'):
    opt = BaseOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    model, criterion, optimizer, lr_scheduler = setup_model(opt, losses=['spans', 'labels', 'saliency'])
    logger.info(f"Model {model} with #Params: {count_parameters(model)}")

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    set_seed(opt.seed)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    long_nlq_dataset = LongNlqDataset(opt,
                                      mode=mode,
                                      use_clip_prefiltering=True,
                                      topk_frames_for_pooling=7,
                                      topk_proposals=50
                                      )
    collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset)
    train_loader = DataLoader(
        long_nlq_dataset,
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=8,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    long_nlq_dataset_val = LongNlqDataset(opt,
                                          mode='val',
                                          use_clip_prefiltering=True,
                                          topk_frames_for_pooling=7,
                                          topk_proposals=10
                                          )
    collate_fn_val = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset_val)
    val_loader = DataLoader(
        long_nlq_dataset,
        collate_fn=collate_fn_val,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    return model, criterion, optimizer, lr_scheduler, opt, train_loader, val_loader


def data_to_device(data, opt):
    for i in range(len(data)):
        for key in data[i].keys():
            data[i][key] = data[i][key].to(opt.device, non_blocking=True)
    return data


def schedule(opt, epoch_i, lr_scheduler):
    if opt.use_warmup and epoch_i < 3:
        lr_scheduler.optimizer.param_groups[0]['lr'] = \
            (0.01 + (epoch_i / 3)) * lr_scheduler.optimizer.defaults['lr']
    if opt.use_warmup and epoch_i == 3:
        lr_scheduler.optimizer.param_groups[0]['lr'] = lr_scheduler.optimizer.defaults['lr']

    return lr_scheduler


def get_saliency_labels(gt_window, foreground, ctx_l, max_n=2):
    if not foreground:
        return [-1, -1], [-1, -1]

    gt_st = int(gt_window[0] / 0.2)
    gt_ed = max(0, min(int(gt_window[1] / 0.2), ctx_l) - 1)
    if gt_st > gt_ed:
        gt_st = gt_ed

    if gt_st != gt_ed:
        pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
    else:
        pos_clip_indices = [gt_st, gt_st]

    neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))

    # this sets the saliency score to zero, since no negative window exists
    if len(neg_pool) <= 0:
        return [-1, -1], [-1, -1]
    else:
        neg_clip_indices = random.sample(neg_pool, k=max_n)
    return pos_clip_indices, neg_clip_indices


def get_windows(_target, _data, opt):
    window = _target['anno']['ext_timestamps']
    window = torch.Tensor(window) / (_data['src_vid'].shape[1] * 0.2)  # normalized windows in xx
    window = span_xx_to_cxw(window)
    windows = [{'spans': (window.unsqueeze(0) * fg).to(opt.device, non_blocking=True)} for fg in
               _target['is_foreground']]
    return windows


def prepare_targets(_target, _data, opt):
    windows = get_windows(_target, _data, opt)
    pos_inds, neg_inds = [], []
    for foreground in _target['is_foreground']:
        pos_clip_indices, neg_clip_indices = get_saliency_labels(_target['anno']['ext_timestamps'], foreground,
                                                                 _data['src_vid'].shape[1])
        pos_inds.append(pos_clip_indices)
        neg_inds.append(neg_clip_indices)

    targets = {'cls_label': _target['is_foreground'].bool().to(opt.device, non_blocking=True),
               'span_labels': windows,
               'saliency_pos_labels': torch.tensor(pos_inds).to(opt.device, non_blocking=True),
               'saliency_neg_labels': torch.tensor(neg_inds).to(opt.device, non_blocking=True)}
    return targets


def print_logs(loss_meters, clip_metrics_meter, epoch_i, tb_writer, opt, optimizer):
    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i + 1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i + 1)

    temp = {}
    for clip_metric in clip_metrics_meter:
        for key in clip_metric.keys():
            if key not in temp.keys():
                temp[key] = []
            temp[key].append(clip_metric[key])

    for k_key in temp.keys():
        temp[k_key] = np.mean(temp[k_key]).round(4)

    logger.info("\naverage metrics\n{}".format(pprint.pformat(temp, indent=4)))


def train(model, criterion, long_nlq_loader, optimizer, opt, epoch_i, tb_writer, clip_metrics):
    logger.info(f"[Epoch {epoch_i + 1}]")

    model.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)
    clip_metrics_meter = []
    for batch in tqdm(long_nlq_loader):

        target, data, qid, windows, clip_metrics = batch
        data = data_to_device(data, opt)

        for i in range(len(data)):
            _target = target[i]
            _data = data[i]

            processed_targets = prepare_targets(_target, _data, opt)

            outputs = model(**_data)
            loss_dict = criterion(outputs, processed_targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            clip_metrics_meter.append(clip_metrics[i])
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    print_logs(loss_meters, clip_metrics_meter, epoch_i, tb_writer, opt, optimizer)


def evaluate(model, criterion, opt, data_loader, epoch_i, tb_writer, prev_best_metric, optimizer):
    with torch.no_grad():
        avg_metrics = clip_decoder_inference(model=model,
                                             criterion=criterion,
                                             opt=opt,
                                             dataloader=data_loader,
                                             tb_writer=tb_writer)

        # if opt.scheduler == 'reduce_plateau':
        #    lr_scheduler.step(eval_loss_meters['loss_overall'].val)

    if avg_metrics['avg_mr_metrics']["brief"]["MR-R5@0.5"] > prev_best_metric:
        file_name = opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")
        save_model(model, optimizer, opt, epoch_i, file_name)
        prev_best_metric = avg_metrics['avg_mr_metrics']["brief"]["MR-R5@0.5"]

    return prev_best_metric


def save_model(model, optimizer, opt, epoch_i, file_name):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch_i,
        "opt": opt
    }
    torch.save(checkpoint, file_name)


def clip_decoder_training():
    model, criterion, optimizer, lr_scheduler, opt, train_loader, val_loader = setup_training()

    preds, metrics, clip_metrics = {}, {}, {}
    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    prev_best_metric = 0

    for epoch_i in range(opt.n_epoch):
        lr_scheduler = schedule(opt, epoch_i, lr_scheduler)

        train(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, clip_metrics)

        lr_scheduler.step()

        if opt.eval_path is not None and (epoch_i + 1) % 1 == 0:
            prev_best_metric = evaluate(model=model,
                                        criterion=criterion,
                                        opt=opt,
                                        data_loader=val_loader,
                                        epoch_i=epoch_i,
                                        tb_writer=tb_writer,
                                        prev_best_metric=prev_best_metric,
                                        optimizer=optimizer)
        if opt.debug:
            break

        save_interval = 2
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            file_name = opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt")
            save_model(model, optimizer, opt, epoch_i, file_name)

    tb_writer.close()


if __name__ == '__main__':
    clip_decoder_training()
