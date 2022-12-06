import json
import random
import numpy as np
from tqdm import tqdm
import functools
import argparse
from utils.basic_utils import dict_to_markdown
from moment_detr.start_end_dataset import StartEndDataset,  collate_fn_replace_corrupted
from moment_detr.config import BaseOptions

import torch
from torch.utils.data import DataLoader

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

opt = BaseOptions().parse()
set_seed(opt.seed)
root="/nfs/data3/goldhofer/mad_dataset/"

dataset_config = dict(
    dset_name='train',
    data_path=f'{root}annotations/MAD_train_SMNone_FPS5_CL30_L2False_extsTrue.json',
    v_feat_dirs=root,
    q_feat_dir=opt.t_feat_dir,
    q_feat_type="last_hidden_state",
    max_q_l=100,
    max_v_l=150,
    ctx_mode=opt.ctx_mode,
    data_ratio=opt.data_ratio,
    normalize_v=not opt.no_norm_vfeat,
    normalize_t=not opt.no_norm_tfeat,
    clip_len=0.2,
    max_windows=opt.max_windows,
    span_loss_type=opt.span_loss_type,
    txt_drop_ratio=opt.txt_drop_ratio,
    sampling_fps=5,
    sampling_mode=opt.sampling_mode,
    lang_feat_path=opt.lang_feat_path,
    v_feat_dim=768,
    dataset_fps=5,
    use_exact_ts=True,

)


def _save_annos(self, anno_path):
    anno_save_path = f'{self.root}{anno_path}'
    with open(anno_save_path, "w") as f:
        f.write("\n".join([json.dumps(e) for e in self.annos]))
    print(f'Saved annotations to: {anno_save_path}')


if __name__ == "__main__":
    dataset_config["data_path"] = opt.train_path
    dataset = StartEndDataset(**dataset_config)
    collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)
    train_loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=not opt.no_shuffle,
        pin_memory=opt.pin_memory
    )

    for batch_idx, batch in tqdm(enumerate(train_loader)):
        print(batch)
