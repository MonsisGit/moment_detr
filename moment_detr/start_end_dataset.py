import traceback
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import copy
import h5py
import pickle
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0, sampling_fps=0.5,
                 sampling_mode='none', lang_feat_path='CLIP_L14_language_tokens_features',
                 dataset_fps=5, v_feat_dim=768, use_exact_ts=False):

        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.lang_feat_path = lang_feat_path
        self.lang_feats = self.get_lang_feats()
        self.using_mat_dataset = True if "mad_dataset" in self.data_path else False
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.sampling_mode = sampling_mode
        self.v_feat_dim = v_feat_dim
        self.data = self.load_data()
        if sampling_mode == 'online':
            self.video_feats = h5py.File(os.path.join(self.v_feat_dirs[0], 'CLIP_L14_frames_features_5fps.h5'), 'r')
            self.normalize_v = True

        self.dataset_fps = dataset_fps
        self.clip_length_in_seconds = self.max_v_l / self.dataset_fps
        self.clip_length_in_frames = self.max_v_l
        self.sampling_fps = sampling_fps
        self.data_keys = list(self.data[0].keys())
        self.use_exact_ts = use_exact_ts
        self.is_val = True if 'val' in self.dset_name else False

        #set seed
        np.random.seed(seed=42)


    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist[0]) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.is_val:
           print('is val')

        model_inputs = dict()
        print(index)

        if self.sampling_mode == 'online' and self.using_mat_dataset:
            qid = self.data_keys[index]
            meta = self.data[0][qid]
            model_inputs["query_feat"] = self._get_query_feat_by_qid(qid)  # (Dq, ) or (Lq, Dq)
        else:
            meta = self.data[index]
            model_inputs["query_feat"] = self._get_query_feat_by_qid(meta['id'])  # (Dq, ) or (Lq, Dq)

        if self.use_video and self.sampling_mode == 'offline' and self.using_mat_dataset:
            model_inputs["video_feat"], meta = self._get_video_feat_by_vid(qid=qid,
                                                                           vid=meta["id"],
                                                                           meta=meta)  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])

        elif self.use_video and self.sampling_mode == 'online' and self.using_mat_dataset:
            model_inputs["video_feat"], meta = self._get_video_feat_by_vid(qid=qid,
                                                                           vid=meta["movie"],
                                                                           meta=meta,
                                                                           index)  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
            if "subs_train" not in self.data_path and not self.using_mat_dataset:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt
        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))

        # this sets the saliency score to zero, since no negative window exists
        if len(neg_pool) == 0:
            return [-1, -1], [-1, -1]
        else:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if self.using_mat_dataset and self.sampling_mode == 'offline':
            q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)

        elif self.using_mat_dataset and self.sampling_mode == 'online':
            q_feat = self.lang_feats[qid]

        else:
            q_feat = np.array(self.lang_feats[qid]).astype(np.float32)

        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def get_lang_feats(self):
        print(f'LOADING: {self.q_feat_dir}{self.lang_feat_path}')
        return h5py.File(f'{self.q_feat_dir}{self.lang_feat_path}', 'r')

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, qid, vid, meta, index):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:

            if self.sampling_mode == 'offline':
                _feat_path = join(_feat_dir, f"{vid}.npz")
                try:
                    _feat = np.load(_feat_path)["features"].astype(np.float32)[:self.max_v_l]
                except Exception as e:
                    print(f'{e}\nFile: {_feat_path}')
                    _feat = np.zeros(shape=(self.max_v_l, 768))

            elif self.sampling_mode == 'online':
                try:
                    _feat, meta = self._online_sampling(qid, vid, meta, index)
                    _feat = _feat.astype(np.float32)[:self.max_v_l]
                except Exception as e:
                    print(f'{e}\n')
                    _feat = np.zeros(shape=(self.max_v_l, self.v_feat_dim))
                    meta = {
                        'relevant_windows': [[0, 2]],
                        'query': meta['sentence'],
                        'duration': self.clip_length_in_seconds,
                    }

            else:
                raise NotImplementedError

            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat), meta

    def _online_sampling(self, qid, vid, meta, index):
        temp_dict = self._compute_annotations(meta)
        video_features, start_moment, stop_moment = self._get_video_features(temp_dict, index)
        meta = {
            'qid': qid,
            'vid': vid,
            'relevant_windows': [[start_moment, stop_moment]],
            'query': meta['sentence'],
            'duration': self.clip_length_in_seconds,
        }
        return video_features, meta

    def _compute_annotations(self, meta):
        movie = meta['movie']
        duration = self.clip_length_in_seconds
        timestamp = meta['ext_timestamps']
        sentence = meta['sentence']

        # Process gt annotations -----------------------------------------------------
        if timestamp[0] < timestamp[1]:
            moment = [max(timestamp[0], 0), min(timestamp[1], duration)]

            start = int(moment[0] * self.dataset_fps)
            stop = int(moment[1] * self.dataset_fps)

            # frames_idx is in frame space
            frames_idx = [start, stop]

            # Save preprocessed annotations ----------------------------------------------
            temp_dict = {
                'movie': movie,
                'moment': moment,
                'frames_idx': frames_idx,
                'sentence': sentence,
                'movie_duration': duration,
            }
        return temp_dict

    def _calc_val_offset(self, index, num_frames):
        if not bool((index/self.clip_length_in_frames)%1):
            offset =  int(max(min(index - index//self.clip_length_in_frames, self.clip_length_in_frames),0))
            return offset - num_frames
        else:
            return index - num_frames

    def _get_video_features(self, meta, index):
        start_idx, stop_idx = meta['frames_idx']
        num_frames = stop_idx - start_idx
        assert num_frames > 0, f"Number of frames is {num_frames}"

        if num_frames < self.clip_length_in_frames:
            if not self.is_val:
                offset = random.sample(range(0, self.clip_length_in_frames - num_frames, 1), 1)[0]
            else:
                offset = self._calc_val_offset(index, num_frames)
            start_window = max(start_idx - offset, 0)

        else:
            center = (start_idx + stop_idx) / 2
            offset = self.clip_length_in_frames / 2
            start_window = max(int(center - offset), 0)

        # Compute features for window
        stop_window = start_window + self.clip_length_in_frames

        if not stop_window <= meta['movie_duration'] * self.dataset_fps:
            stop_window = int(meta['movie_duration'] * self.dataset_fps)
            start_window = stop_window - self.clip_length_in_frames

        feats = self.video_feats[meta['movie']][start_window:stop_window]

        assert feats.shape[0] == self.clip_length_in_frames

        # Compute moment position within the window in seconds
        start_moment = max((start_idx - start_window) / self.dataset_fps, 0)
        stop_moment = min((stop_idx - start_window) / self.dataset_fps, self.clip_length_in_seconds)

        if self.use_exact_ts:
            start_idx = meta['moment'][0] * self.dataset_fps
            stop_idx = meta['moment'][1] * self.dataset_fps
            start_moment = max((start_idx - start_window) / self.dataset_fps, 0)
            stop_moment = min((stop_idx - start_window) / self.dataset_fps, self.clip_length_in_seconds)

        assert 0 <= start_moment <= self.clip_length_in_seconds, f'start moment ({start_moment}) outside clip'
        assert 0 <= stop_moment <= self.clip_length_in_seconds, f'stop moment ({stop_moment}) outside clip'

        return feats, start_moment, stop_moment


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
