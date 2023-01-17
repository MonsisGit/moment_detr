import logging
import pprint
import numpy as np
from tqdm import tqdm
import functools
import os
import time
import json
from collections import defaultdict

from moment_detr.config import BaseOptions
from moment_detr.inference import setup_model, eval_epoch
from utils.model_utils import count_parameters
from moment_detr.long_nlq_dataset import LongNlqDataset, collate_fn_replace_corrupted
from moment_detr.clip_similarity import clip_filter_proposals
from moment_detr.clip_decoder_inference import set_seed
from utils.basic_utils import dict_to_markdown,AverageMeter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def setup_training():
    opt = BaseOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
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

    long_nlq_dataset = LongNlqDataset(opt, mode='train')
    collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset)
    train_loader = DataLoader(
        long_nlq_dataset,
        collate_fn=collate_fn,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    long_nlq_dataset_val = LongNlqDataset(opt, mode='val')
    return model, criterion, optimizer, lr_scheduler, opt, train_loader, long_nlq_dataset_val


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

def prepare_targets(_target):
    targets = {}
    #TODO

def train(model, criterion, long_nlq_loader, optimizer, opt, epoch_i, tb_writer, clip_metrics):
    logger.info(f"[Epoch {epoch_i + 1}]")

    model.train()
    criterion.train()

    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    for batch in tqdm(long_nlq_loader):

        target, data, qid, windows = batch
        data = data_to_device(data, opt)

        for i in range(len(data)):
            _target = target[i]
            _data = data[i]

            _data, _target, clip_metrics, windows = clip_filter_proposals(_data,
                                                                          _target,
                                                                          opt.topk_pooling_frames,
                                                                          opt.clip_topk,
                                                                          clip_metrics,
                                                                          windows,
                                                                          i)

            targets = prepare_targets(_target)

            outputs = model(**_data)
            loss_dict = criterion(outputs, targets)
            # TODO doesnt return CLS loss yet
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            timer_start = time.time()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))


        # print/add logs
        tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i + 1)
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i + 1)

        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i + 1,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)



def eval(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer, lr_scheduler, optimizer):
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

        if opt.scheduler == 'reduce_plateau':
            lr_scheduler.step(eval_loss_meters['loss_overall'].val)
    # log
    to_write = opt.eval_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
        eval_metrics_str=json.dumps(metrics_no_nms))

    with open(opt.eval_log_filepath, "a") as f:
        f.write(to_write)
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

    metrics = metrics_no_nms
    for k, v in metrics["brief"].items():
        tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i + 1)

    # writing no_nms results to tensorboard
    if metrics_nms is not None:
        for k, v in metrics_nms["brief"].items():
            tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i + 1)

    # save ckpt
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch_i,
        "opt": opt
    }
    torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))


def clip_decoder_training():
    model, criterion, optimizer, lr_scheduler, opt, train_loader, long_nlq_dataset_val = setup_training()

    preds, metrics, clip_metrics = {}, {}, {}
    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)

    for epoch_i in range(opt.n_epoch):
        lr_scheduler = schedule(opt, epoch_i, lr_scheduler)

        train(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, clip_metrics)

        lr_scheduler.step()

        if opt.eval_path is not None and (epoch_i + 1) % 1 == 0:
            eval(model, long_nlq_dataset_val, opt, save_submission_filename, epoch_i, criterion, tb_writer, lr_scheduler, optimizer)

        if opt.debug:
            break

        save_interval = 10
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

    tb_writer.close()


if __name__ == '__main__':
    clip_decoder_training()
