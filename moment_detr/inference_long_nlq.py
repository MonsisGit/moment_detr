import logging
import functools
from tqdm import tqdm
from collections import defaultdict
from utils.basic_utils import AverageMeter

from moment_detr.config import TestOptions
from moment_detr.inference import setup_model
from moment_detr.start_end_dataset import collate_fn_replace_corrupted, prepare_batch_inputs
from moment_detr.start_end_dataset_long_nlq import StartEndDatasetLong
from moment_detr.span_utils import span_cxw_to_xx

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def start_inference_long_nlq():
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    opt.eval_path = opt.eval_path.replace('val', 'test')
    opt.dset_name = 'test'

    eval_dataset = StartEndDatasetLong(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        sampling_fps=opt.sampling_fps,
        sampling_mode=opt.sampling_mode,
        lang_feat_path=opt.lang_feat_path,
        v_feat_dim=opt.v_feat_dim,
        dataset_fps=opt.dataset_fps,
        use_exact_ts=opt.use_exact_ts,
    )

    model, criterion, _, _ = setup_model(opt)

    save_submission_filename = "inference_long_nlq_preds.jsonl"
    logger.info(f"Starting inference on {opt.eval_path.split('/')[-1]}")
    with torch.no_grad():
        logger.info("Generate submissions")
        model.eval()
        criterion.eval()

        collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=eval_dataset)
        eval_loader = DataLoader(
            eval_dataset,
            collate_fn=collate_fn,
            batch_size=opt.eval_bsz,
            num_workers=opt.num_workers,
            shuffle=False,
            pin_memory=opt.pin_memory
        )

    loss_meters = defaultdict(AverageMeter)

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
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
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
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
