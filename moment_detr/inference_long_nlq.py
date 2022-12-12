import logging
import os
from tqdm import tqdm
from collections import defaultdict
from utils.basic_utils import AverageMeter
import h5py

from moment_detr.config import TestOptions
from moment_detr.inference import setup_model
from moment_detr.start_end_dataset import collate_fn_replace_corrupted, prepare_batch_inputs
from moment_detr.start_end_dataset_long_nlq import StartEndDatasetLong
from moment_detr.span_utils import span_cxw_to_xx
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


def prepare_inputs(qid, anno, window, lang_feats, video_feats, device,opt):

    model_inputs = dict()
    model_inputs['src_txt'] = torch.tensor(lang_feats[qid]).view(1, -1, opt.q_feat_dim).to(device)

    vid = torch.tensor(video_feats[anno['movie']][window[0]:window[1]])
    ctx_l = len(vid)
    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
    model_inputs['src_vid'] = torch.cat([vid, tef], dim=1).view(1, -1, opt.v_feat_dim).to(device)

    model_inputs['src_txt_mask'] = torch.ones_like(model_inputs['src_txt'][...,0]).to(device)
    model_inputs['src_vid_mask'] = torch.ones_like(model_inputs['src_vid'][...,0]).to(device)

    return model_inputs


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
            for window in range(0, len_movie, window_length):

                model_inputs = prepare_inputs(qid=qid,
                                              anno=annos[qid],
                                              window=[window, window + window_length],
                                              lang_feats=lang_feats,
                                              video_feats=video_feats,
                                              device=opt.device,
                                              opt=opt)
                outputs = model(**model_inputs)

                prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
                if opt.span_loss_type == "l1":
                    scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
                    pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
                    _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
                    saliency_scores = []
                    valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
                    for j in range(len(valid_vid_lengths)):
                        saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
                else:
                    bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
                    pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
                    # TODO use more advanced decoding method with st_ed product
                    pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(
                        -1)  # 2 * (bsz, #queries, 2)
                    scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
                    pred_spans[:, 1] += 1
                    pred_spans *= opt.clip_length

                # compose predictions
                for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
                    if opt.span_loss_type == "l1":
                        spans = span_cxw_to_xx(spans) * meta["duration"]
                    # # (#queries, 3), [st(float), ed(float), score(float)]
                    cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
                    if not opt.no_sort_results:
                        cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
                    cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

                    cur_query_pred = dict(
                        qid=meta['qid'],
                        query=meta["query"],
                        vid=meta['vid'],
                        pred_relevant_windows=cur_ranked_preds,
                        pred_saliency_scores=saliency_scores[idx],
                        pred_cls=outputs['pred_cls'].tolist()[idx][0]
                    )
                    mr_res.append(cur_query_pred)

                if criterion:
                    loss_dict = criterion(outputs, targets)
                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    loss_dict["loss_overall"] = float(losses)  # for logging only
                    for k, v in loss_dict.items():
                        loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

                if opt.debug:
                    break


if __name__ == '__main__':
    start_inference_long_nlq()
