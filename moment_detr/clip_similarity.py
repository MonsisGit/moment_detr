import numpy as np
import os

import torch


def clip_similarity(src_txt, src_txt_mask, src_vid, src_vid_mask, k=5):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:<4096>'
    pooling_type = 'topk'

    src_txt = src_txt[:, -1, :]
    src_vid = src_vid[..., :-2]
    if pooling_type == 'topk':
        vid_embeds_pooled = topk_pooling(src_txt, src_vid, k=k)
    else:
        vid_embeds_pooled = avg_pooling(src_txt, src_vid)
    vid_embeds_pooled = normalize(vid_embeds_pooled)
    sims = sim_matrix(src_txt, vid_embeds_pooled)

    return sims


def normalize(embeds):
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds


def avg_pooling(text_embeds, video_embeds):
    pooled_embeds = video_embeds.sum(dim=1)
    return pooled_embeds


def topk_pooling(text_embeds, video_embeds, k):
    embed_dim = video_embeds.shape[-1]
    sims = torch.bmm(text_embeds.unsqueeze(1), video_embeds.permute(0, 2, 1))
    topk_inds = torch.topk(sims, k, dim=2)[1]
    video_embeds_topk = torch.gather(video_embeds, dim=1, index=topk_inds.permute(0, 2, 1).expand(-1, -1, embed_dim))
    pooled_embeds = video_embeds_topk.sum(dim=1)
    return pooled_embeds


def sim_matrix(text_embeds, video_embeds):
    sims = text_embeds[0, :] @ video_embeds.T
    return sims


def compute_metrics(sims, ground_truth, k, top_ks, metrics):
    sims = sims.cpu()
    _gt = ground_truth['is_foreground']
    if len(metrics) == 0:
        for keys_k in top_ks:
            metrics[str(keys_k)] = {m: [] for m in ['R@1', 'R@5', 'R@10', 'R@50', 'R@100']}

    for R, metric in zip([1, 5, 10, 50, 100], ['R@1', 'R@5', 'R@10', 'R@50', 'R@100']):
        top_R_inds = torch.topk(sims, R, dim=0)[1]
        temp = []
        for gt in torch.where(_gt != 0)[0]:
            temp.append(torch.any(top_R_inds == gt, dim=0))

        metrics[str(k)][metric].append(float(torch.stack(temp).any(dim=0)) * 100)

    return metrics


def topk_from_data(_data, topk_inds):
    for key in _data.keys():
        if not 'mask' in key:
            topk_inds_gather = topk_inds.unsqueeze(-1).unsqueeze(-1).expand(topk_inds.shape[0],
                                                                            _data[key].shape[1],
                                                                            _data[key].shape[2])
            _data[key] = torch.gather(_data[key], dim=0, index=topk_inds_gather)
        else:
            topk_inds_gather = topk_inds.unsqueeze(-1).expand(topk_inds.shape[0],
                                                              _data[key].shape[1])
            _data[key] = torch.gather(_data[key], dim=0, index=topk_inds_gather)
    return _data


def topk_from_target(_target, topk_inds):
    _target['is_foreground'] = _target['is_foreground'][topk_inds.cpu()]
    _target['windows'] = torch.gather(_target['windows'], dim=0,
                                      index=topk_inds.cpu().unsqueeze(-1).expand(topk_inds.shape[0], 2))
    return _target


def clip_filter_proposals(_data, _target, pooling_topk, topk, metrics, windows, i):
    sims = clip_similarity(**_data, k=pooling_topk)
    metrics = compute_metrics(sims, _target, k=pooling_topk, top_ks=[pooling_topk], metrics=metrics)
    topk_inds = torch.topk(sims, topk, dim=0)[1]
    _data = topk_from_data(_data, topk_inds)
    _target = topk_from_target(_target, topk_inds)
    sims = normalize(sims[topk_inds])

    if i == -1:
        windows = windows[topk_inds.cpu(), :]
    else:
        windows[i] = windows[i][topk_inds.cpu(), :]

    return _data, _target, windows, metrics, sims
