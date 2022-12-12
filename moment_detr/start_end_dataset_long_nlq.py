import os
import torch
from torch.utils.data import Dataset
import numpy as np
import time
import h5py
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDatasetLong(Dataset):
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
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0, sampling_fps=5,
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
            self.is_val = True
        else:
            self.is_val = False

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
        # self.data_keys = [d['qid'] for d in self.data]
        self.use_exact_ts = use_exact_ts
        self.val_offset = 0

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            if n_examples == 0:
                n_examples = 1
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        try:
            model_inputs = dict()
            meta = self.data[index]
            model_inputs["query_feat"] = self._get_query_feat_by_qid(meta['qid'])  # (Dq, ) or (Lq, Dq)
            if self.use_video and self.using_mat_dataset:
                model_inputs["video_feat"], meta = self._get_video_feat_by_vid(meta)  # (Lv, Dv)
                ctx_l = len(model_inputs["video_feat"])
                model_inputs['cls_label'] = meta['foreground']

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
                        self.get_saliency_labels_sub_as_query(meta, ctx_l)  # only one gt
            return dict(meta=meta, model_inputs=model_inputs)
        except Exception as e:
            return None

    def get_saliency_labels_sub_as_query(self, meta, ctx_l, max_n=2):
        if not meta['foreground']:
            return [-1, -1], [-1, -1]

        gt_window = meta["relevant_windows"][0]
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
        if len(neg_pool) <= 0:
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

    def _get_video_feat_by_vid(self, meta):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:

            if self.sampling_mode == 'offline':
                _feat_path = join(_feat_dir, f"{meta['vid']}.npz")
                _feat = np.load(_feat_path)["features"].astype(np.float32)[:self.max_v_l]

            elif self.sampling_mode == 'online':
                _feat, meta = self._online_sampling(meta)
                _feat = _feat.astype(np.float32)[:self.max_v_l]

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

    def _online_sampling(self, meta):

        if not self.is_val:
            gt_moment = self._get_gt_moment(meta)
            window, is_foreground = self._sample_window(gt_moment, meta)
            duration = self.clip_length_in_seconds
            if is_foreground:
                start_moment, stop_moment = self._calc_new_moment(window, gt_moment, meta)
            else:
                # if background is sampled, the network should predict zeros
                start_moment = stop_moment = 0

        else:
            window = [max(meta['window'][0], 0), min(meta['window'][1], self.video_feats[meta['vid']].shape[0])]
            start_moment, stop_moment = meta["relevant_windows"][0]
            is_foreground = meta['is_foreground']
            duration = meta['duration']

        _video_feats = self.video_feats[meta['vid']][window[0]:window[1]]
        # TODO this doesnt seem to work in training
        if self.sampling_fps != self.dataset_fps and False:
            _video_feats = _video_feats[::int(self.dataset_fps / self.sampling_fps)]

        meta = {
            'qid': meta['qid'],
            'vid': meta['vid'],
            'relevant_windows': [[start_moment, stop_moment]],
            'query': meta['query'],
            'duration': duration,
            'foreground': is_foreground
        }
        return _video_feats, meta

    def _get_gt_moment(self, meta):

        timestamp = meta['relevant_windows'][0]

        # Process gt annotations -----------------------------------------------------
        if timestamp[0] < timestamp[1]:
            moment = [max(timestamp[0], 0), min(timestamp[1], meta['duration'])]

            start = int(moment[0] * self.dataset_fps)
            stop = int(moment[1] * self.dataset_fps)

            # is in frame space
            return [start, stop]
        else:
            return [0, 0]

    def _sample_window(self, frame_idx, meta):
        start_idx, stop_idx = frame_idx
        num_frames = stop_idx - start_idx
        assert num_frames > 0, f"Number of frames is {num_frames}"

        if num_frames < self.clip_length_in_frames:
            offset = random.sample(range(0, self.clip_length_in_frames - num_frames, 1), 1)[0]
            start_window = max(start_idx - offset, 0)

        else:
            center = (start_idx + stop_idx) / 2
            offset = self.clip_length_in_frames / 2
            start_window = max(int(center - offset), 0)

        # Compute features for window
        stop_window = start_window + self.clip_length_in_frames

        if not stop_window <= meta['duration'] * self.dataset_fps:
            stop_window = int(np.floor(meta['duration'] * self.dataset_fps))
            start_window = stop_window - self.clip_length_in_frames

        # sample negative window
        if random.random() < 0.5:
            neg_start_window, neg_stop_window = self._sample_neg_window(start_window, stop_window, meta)
            is_foreground = False
            return [neg_start_window, neg_stop_window], is_foreground

        else:
            is_foreground = True
            return [start_window, stop_window], is_foreground

    def _sample_neg_window(self, start_window, stop_window, meta, recursion_counter=0):

        sample_windows = [start_window, int(np.floor(meta['duration'] * self.dataset_fps)) - stop_window]
        idx_larger_window = np.argmax(sample_windows)
        if idx_larger_window == 0:
            neg_start_window = max(int(0.5 * random.random() * sample_windows[idx_larger_window]), 0)
        else:
            neg_start_window = max(int(
                sample_windows[idx_larger_window] * random.random() + sample_windows[0] + self.clip_length_in_frames),
                0)
        neg_stop_window = neg_start_window + self.clip_length_in_frames

        if not neg_stop_window <= meta['duration'] * self.dataset_fps:
            neg_stop_window = int(np.floor(meta['duration'] * self.dataset_fps))
            neg_start_window = neg_stop_window - self.clip_length_in_frames

        if recursion_counter > 5:
            logger.info(f'Maximum Recursion depth exceeded: {recursion_counter}')
            return None
        if start_window <= neg_start_window <= stop_window or start_window <= neg_stop_window <= stop_window:
            recursion_counter += 1
            return self._sample_neg_window(start_window, stop_window, meta, recursion_counter)

        return neg_start_window, neg_stop_window

    def _calc_new_moment(self, window, gt_moment, meta):
        start_window, stop_window = window
        start_idx, stop_idx = gt_moment

        # Compute moment position within the window in seconds
        start_moment = max((start_idx - start_window) / self.dataset_fps, 0)
        stop_moment = min((stop_idx - start_window) / self.dataset_fps, self.clip_length_in_seconds)

        if self.use_exact_ts:
            start_idx = meta['relevant_windows'][0][0] * self.dataset_fps
            stop_idx = meta['relevant_windows'][0][1] * self.dataset_fps
            start_moment = max((start_idx - start_window) / self.dataset_fps, 0)
            stop_moment = min((stop_idx - start_window) / self.dataset_fps, self.clip_length_in_seconds)

        # assert 0 <= start_moment <= self.clip_length_in_seconds, f'start moment ({start_moment}) outside clip'
        # assert 0 <= stop_moment <= self.clip_length_in_seconds, f'stop moment ({stop_moment}) outside clip'

        return start_moment, stop_moment


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
        if k == 'cls_label':
            batched_data[k] = torch.tensor([e["model_inputs"][k] for e in batch])
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

    if "cls_label" in batched_model_inputs:
        targets["cls_label"] = batched_model_inputs['cls_label'].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets


def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with another examples sampled randomly.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset which the DataLoader is loading.
            Specify it with functools.partial and pass the resulting partial function that only
            requires 'batch' argument to DataLoader's 'collate_fn' option.

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783

    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return start_end_collate(batch)
