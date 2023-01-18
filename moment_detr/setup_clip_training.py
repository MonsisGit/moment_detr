import functools
import logging
import random
import numpy as np
import os

from moment_detr.config import BaseOptions
from moment_detr.inference import setup_model
from utils.model_utils import count_parameters
from moment_detr.long_nlq_dataset import LongNlqDataset, collate_fn_replace_corrupted
#from moment_detr.long_nlq_dataset import LongNlqSampler

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def data_to_device(data, opt):
    for i in range(len(data)):
        for key in data[i].keys():
            data[i][key] = data[i][key].to(opt.device, non_blocking=True)
    return data


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def postprocess_clip_metrics(clip_metrics_meter):
    temp = {}
    meter_key = list(clip_metrics_meter[0].keys())[0]
    for clip_metric in clip_metrics_meter:
        for key in clip_metric[meter_key].keys():
            if key not in temp.keys():
                temp[key] = []
            temp[key].append(clip_metric[meter_key][key])

    for k_key in temp.keys():
        temp[k_key] = np.mean(temp[k_key]).round(4)
    return temp


def setup_training(mode='train'):
    opt = BaseOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False
    if opt.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, opt.cuda_visible_devices))

    model, criterion, optimizer, lr_scheduler = setup_model(opt, losses=['spans', 'labels', 'saliency'])
    logger.info(f"Model {model} with #Params: {count_parameters(model)}")
    logger.info(f'Current GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})')

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
                                      topk_frames_for_pooling=opt.topk_pooling_frames,
                                      topk_proposals=opt.clip_topk
                                      )
    collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset)
    #sampler = LongNlqSampler(batch_size=10000,
    #                         shuffle=True,
    #                         len_dataset=len(long_nlq_dataset))

    train_loader = DataLoader(
        long_nlq_dataset,
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=8,
        shuffle=True,
        pin_memory=opt.pin_memory,
    )

    long_nlq_dataset_val = LongNlqDataset(opt,
                                          mode='val',
                                          use_clip_prefiltering=True,
                                          topk_frames_for_pooling=opt.topk_pooling_frames,
                                          topk_proposals=opt.clip_topk
                                          )
    collate_fn_val = functools.partial(collate_fn_replace_corrupted, dataset=long_nlq_dataset_val)
    val_loader = DataLoader(
        long_nlq_dataset_val,
        collate_fn=collate_fn_val,
        batch_size=2,
        num_workers=4,
        shuffle=True,
        pin_memory=opt.pin_memory)

    return model, criterion, optimizer, lr_scheduler, opt, train_loader, val_loader
