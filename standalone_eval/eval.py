import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp
from standalone_eval.utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap

import torch
from torchmetrics.classification import BinaryRecall, BinaryAccuracy, BinaryPrecision


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    # TODO look at ap
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"],  # in order to use the API
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    if ap_array.shape[0] == 0:
        return {'average': -1}
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_rk(submission, ground_truth, iou_thds=[0.1, 0.3, 0.5], top_ks=[1, 2, 5, 10], is_nms=False):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""

    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = dict()
    iou_thd2recall_at_k = {}
    for top_k in top_ks:
        for s in submission:
            pred_qid2window[s["qid"]] = [k[0:2] for k in s["pred_relevant_windows"][0:top_k]]  # :2 rm scores

        iou_thd2recall_at_d = []
        for d in ground_truth:
            cur_gt_windows = d["relevant_windows"]
            cur_qid = d["qid"]
            if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
                #if len(pred_qid2window) >= top_k:
                curr_pred_qid2window = np.array(pred_qid2window[cur_qid])
                cur_ious = compute_temporal_iou_batch_cross(
                    curr_pred_qid2window, np.array(d["relevant_windows"])
                )[0]
                iou_thd2recall_at_d.append(cur_ious)

        for thd in iou_thds:
            if len(iou_thd2recall_at_d) != 0:
                if not is_nms:
                    iou_thd2recall_at_k[f'{thd}@{top_k}'] = float(
                        f"{(np.array(iou_thd2recall_at_d)[..., 0] >= thd).any(1).mean() * 100:.2f}")
                else:
                    iou_temp = []
                    for iou in iou_thd2recall_at_d:
                        iou_temp.append((iou >= thd).any())

                    iou_thd2recall_at_k[f'{thd}@{top_k}'] = float(
                        f"{np.mean(iou_temp) * 100:.2f}")

            else:
                iou_thd2recall_at_k[f'{thd}@{top_k}'] = -1

    return iou_thd2recall_at_k


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one


def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range, is_long_nlq):
    """ keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    if is_long_nlq:
        return submission, ground_truth
    is_foreground_idx = [idx for idx, g in enumerate(ground_truth) if g['is_foreground']]
    ground_truth = [g for idx, g in enumerate(ground_truth) if idx in is_foreground_idx]
    submission = [g for idx, g in enumerate(submission) if idx in is_foreground_idx]

    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    shared_qids = pred_qids.intersection(gt_qids)

    if len(gt_qids) != len(shared_qids) or len(pred_qids) != len(shared_qids):
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    min_l, max_l = len_range
    if min_l == 0 and max_l == 200:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        if not any(isinstance(i, list) for i in d["relevant_windows"]):
            d["relevant_windows"] = [d["relevant_windows"]]
        rel_windows_in_range = [
            w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def sort_pos_predicted(submission, ground_truth, n=None):
    # moment retrieval recall is only calculated on positive predicted windows
    pred_proba = torch.tensor([s['pred_cls'] for s in submission]).sigmoid()
    predicted_foreground_idx = (torch.where(pred_proba > 0.5)[0]).cpu()
    if predicted_foreground_idx.shape[0]==0:
        return [], []
    _submission = [submission[i] for i in predicted_foreground_idx]
    # can be any list entry of ground truth, since all are same
    _ground_truth = [ground_truth[0]]

    # predicted windows are sorted by confidence
    _submission_vstack = []
    for _s in _submission:
        _submission_vstack.extend(_s['pred_relevant_windows'])
    _submission_sorted = sorted(_submission_vstack, key=lambda x: x[2], reverse=True)[0:n]
    _submission = [{'qid': _ground_truth[0]['qid'],
                    'pred_relevant_windows': _submission_sorted,
                    'pred_cls':[_s['pred_cls'] for _s in _submission]}]

    return _submission, _ground_truth


def remove_zero_predictions(submission, ground_truth):
    # removing zero windows from submission
    for idx, s in enumerate(submission):
        pred_relevant_windows_wo_zeros = [_s for _s in s['pred_relevant_windows'] if _s[0:2] != [0, 0]]
        submission[idx]['pred_relevant_windows'] = pred_relevant_windows_wo_zeros
        if len(submission[idx]['pred_relevant_windows']) == 0:
            del submission[idx]
            del ground_truth[idx]
    return submission, ground_truth


def eval_moment_retrieval(submission, ground_truth, verbose=True, is_nms=False,
                          is_long_nlq=False, length_ranges=[[0, 10], [10, 20], [20, 30], [0, 200], ],
                          range_names=["short", "middle", "long", "full"],
                          iou_thds=[0.1, 0.3, 0.5], top_ks=[1, 2, 5, 10]):
    range_names = [f'{d}_{length_ranges[idx][0]}_{length_ranges[idx][1]}' if d != 'full' else 'full' for idx, d in
                   enumerate(range_names)]

    ret_metrics = {}
    submission, ground_truth = remove_zero_predictions(submission, ground_truth)

    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()

        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range, is_long_nlq)
        cls_acc, cls_recall, cls_precision = compute_ret_metrics(_submission, _ground_truth)

        if is_long_nlq:
            _submission, _ground_truth = sort_pos_predicted(_submission, _ground_truth)

        if len(_submission) != 0:
            if verbose:
                print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
                      f"{100 * len(_ground_truth) / len(ground_truth):.2f}% examples.")

            iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
            iou_thd2recall_at_k = compute_mr_rk(submission=_submission,
                                                ground_truth=_ground_truth,
                                                is_nms=is_nms,
                                                iou_thds=iou_thds,
                                                top_ks=top_ks)

            if verbose:
                print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
        else:
            iou_thd2average_precision = {"0.5": -1,
                                         "0.55": -1,
                                         "0.6": -1,
                                         "0.65": -1,
                                         "0.70": -1,
                                         "0.75": -1,
                                         "0.8": -1,
                                         "0.85": -1,
                                         "0.9": -1,
                                         "0.95": -1,
                                         "average": -1,
                                         }
            iou_thd2recall_at_k = {'0.1@1': -1,
                                   '0.3@1': -1,
                                   '0.5@1': -1,
                                   '0.1@5': -1,
                                   '0.3@5': -1,
                                   '0.5@5': -1,
                                   '0.1@10': -1,
                                   '0.3@10': -1,
                                   '0.5@10': -1, }

        ret_metrics[name] = {"MR-mAP": iou_thd2average_precision,
                             "MR-RK": iou_thd2recall_at_k,
                             'CLS': {'accuracy': round(cls_acc, 2),
                                     'recall': round(cls_recall, 2),
                                     'precision': round(cls_precision, 2)}
                             }

    return ret_metrics


def compute_ret_metrics(_submission, _ground_truth):

    preds = torch.tensor([s['pred_cls'] for s in _submission]).squeeze()
    targets = torch.tensor([int(gt['is_foreground']) for gt in _ground_truth])

    # accuracy might be high, because of unbalanced data
    binary_accuracy = BinaryAccuracy()
    accuracy = binary_accuracy(preds, targets)
    # recall is TP / (TP + FN), it evaluates the completeness of the positive predictions
    # TP is correctly predicted foreground windows, FN is incorrectly predicted foreground windows
    binary_recall = BinaryRecall()
    recall = binary_recall(preds, targets)
    #precision is TP / (TP + FP), it evaluates the correctness of the positive predictions
    # background windows are also considered
    binary_precision = BinaryPrecision()
    precision = binary_precision(preds, targets)

    return float(accuracy), float(recall), float(precision)


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {k: np.argmax(v["pred_saliency_scores"]) for k, v in qid2preds.items()}
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]  # (#clips, 3)
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    # aggregate scores from 3 separate annotations (3 workers) by taking the max.
    # then average scores from all queries.
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {k: v["pred_saliency_scores"] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds), 3))  # (#preds, 3)
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):  # annotation score idx
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, w_idx, score in pool.imap_unordered(
                    compute_ap_from_tuple, input_tuples, chunksize=chunksize):
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score

    # it's the same if we first average across different annotations, then average across queries
    # since all queries have the same #annotations.
    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        # print(f"len(y_true) < len(y_predict) {len(y_true), len(y_predict)}")
        y_predict = y_predict[:len(y_true)]
    elif len(y_true) > len(y_predict):
        # print(f"len(y_true) > len(y_predict) {len(y_true), len(y_predict)}")
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


def mk_gt_scores(gt_data, clip_length=2):
    """gt_data, dict, """
    num_clips = int(gt_data["duration"] / clip_length)
    saliency_scores_full_video = np.zeros((num_clips, 3))
    # relevant_clip_ids = np.array(gt_data["relevant_clip_ids"])  # (#relevant_clip_ids, )
    # relevant_clip_ids = np.zeros(saliency_scores_full_video.shape[0])
    # saliency_scores_relevant_clips = np.array(gt_data["saliency_scores"])  # (#relevant_clip_ids, 3)
    # saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    return saliency_scores_full_video  # (#clips_in_video, 3)  the scores are in range [0, 4]


def eval_highlight(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    qid2preds = {d["qid"]: d for d in submission}
    qid2gt_scores_full_range = {d["qid"]: mk_gt_scores(d) for d in ground_truth}  # scores in range [0, 4]
    # gt_saliency_score_min: int, in [0, 1, 2, 3, 4]. The minimum score for a positive clip.
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ["Fair", "Good", "VeryGood"]
    highlight_det_metrics = {}
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list, saliency_score_names):
        start_time = time.time()
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full_range.items()}  # scores in [0, 1]
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f"HL-min-{score_name}"] = {"HL-mAP": mean_ap, "HL-Hit1": hit_at_one}
        if verbose:
            print(f"Calculating highlight scores with min score {gt_saliency_score_min} ({score_name})")
            print(f"Time cost {time.time() - start_time:.2f} seconds")
    return highlight_det_metrics


def eval_submission(submission, ground_truth, verbose=True, match_number=False, is_nms=False,
                    is_long_nlq=False, length_ranges=[[0, 10], [10, 20], [20, 30], [0, 200], ],
                    range_names=["short", "middle", "long", "full"],
                    iou_thds=[0.1, 0.3, 0.5], top_ks=[1, 2, 5, 10]):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])

    # TODO set match_number to False
    match_number = False
    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        if len(gt_qids) != len(shared_qids) or len(pred_qids) != len(shared_qids):
            submission = [e for e in submission if e["qid"] in shared_qids]
            ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose, is_nms=is_nms,
            is_long_nlq=is_long_nlq, length_ranges=length_ranges, range_names=range_names,
            iou_thds=iou_thds, top_ks=top_ks)

        eval_metrics.update(moment_ret_scores)
        if is_nms:
            moment_ret_scores_brief = {
                "MR-R1@0.5 (nms)": moment_ret_scores["full"]["MR-RK"]["0.5@1"],
                "MR-R5@0.5 (nms)": moment_ret_scores["full"]["MR-RK"]["0.5@5"],
                "MR-R10@0.5 (nms)": moment_ret_scores["full"]["MR-RK"]["0.5@10"],

            }
        else:
            moment_ret_scores_brief = {
                "CLS-Acc": moment_ret_scores["full"]['CLS']["accuracy"],
                "CLS-Recall": moment_ret_scores["full"]['CLS']["recall"],
                "CLS-Precision": moment_ret_scores["full"]['CLS']["precision"],
                "MR-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],

                "MR-R1@0.5": moment_ret_scores["full"]["MR-RK"]["0.5@1"],
                "MR-R2@0.5": moment_ret_scores["full"]["MR-RK"]["0.5@2"],
                "MR-R5@0.5": moment_ret_scores["full"]["MR-RK"]["0.5@5"],
                "MR-R10@0.5": moment_ret_scores["full"]["MR-RK"]["0.5@10"],
            }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    # TODO no highlight score calculation
    if "pred_saliency_scores" in submission[0] and False:
        highlight_det_scores = eval_highlight(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict([
            (f"{k}-{sub_k.split('-')[1]}", v[sub_k])
            for k, v in highlight_det_scores.items() for sub_k in v])
        eval_metrics_brief.update(highlight_det_scores_brief)

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
