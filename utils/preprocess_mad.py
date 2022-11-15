import copy
import json, os
import numpy as np
from tqdm import tqdm
import h5py
import pickle
from pathlib import Path
import traceback


def run():
    root = '/nfs/data3/goldhofer/mad_dataset'
    annotation_paths = [f'{root}/annotations/MAD_val.json',
                        f'{root}/annotations/MAD_test.json', f'{root}/annotations/MAD_train.json']
    frame_features_path = 'CLIP_L14_frames_features_5fps.h5'
    annotations_filename = "_transformed_exact.json"
    meta_filename = '_meta_log_exact.pkl'
    save_path = 'clip_frame_features_transformed_exact/'

    clip_frame_features = get_video_feats(root, frame_features_path)
    rng = np.random.default_rng(42)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    id_tracker = []
    exact_timestamps = True

    fps = 5
    video_length_seconds = 150

    for annotation_path in annotation_paths:
        print(f'Processing {annotation_path}')
        annotated_data = json.load(open(annotation_path, 'r'))
        meta_cache = {}
        mad_transformed = []
        discarded_datapoints_counter = 0

        for k in tqdm(list(annotated_data.keys())):

            try:
                assert k not in id_tracker, f'duplicated id: {k}'
                id_tracker.append(k)

                if exact_timestamps:
                    lowest_clip = annotated_data[k]["ext_timestamps"][0]
                    highest_clip = annotated_data[k]["ext_timestamps"][1]
                else:
                    lowest_clip = int(annotated_data[k]["ext_timestamps"][0])
                    highest_clip = int(annotated_data[k]["ext_timestamps"][1])

                    if lowest_clip % 2 != 0:
                        lowest_clip -= 1
                    if highest_clip % 2 != 0:
                        highest_clip += 1

                if highest_clip > annotated_data[k]["movie_duration"]:
                    print(
                        f'highest clip higher than movie duration, adjusted from {highest_clip} to {int(np.floor(annotated_data[k]["movie_duration"]))}')
                    highest_clip = int(np.floor(annotated_data[k]["movie_duration"]))

                if highest_clip > lowest_clip:

                    meta = {"qid": k,
                            "query": annotated_data[k]["sentence"],
                            "duration": video_length_seconds,
                            "vid": k,
                            "relevant_windows": [[lowest_clip, highest_clip]], }
                    if not exact_timestamps:
                        meta["relevant_clip_ids"] = [i for i in
                                                     range(int(lowest_clip / 2), int(highest_clip / 2))]
                        meta["saliency_scores"] = [[0, 0, 0] for _ in
                                                   range(int(lowest_clip / 2), int(highest_clip / 2))]

                    old_meta = copy.deepcopy(meta)
                    sliced_frame_features, meta = slice_window(clip_frame_features[annotated_data[k]["movie"]], meta,
                                                               rng, fps, video_length_seconds, exact_timestamps)
                    meta_cache = log_meta(old_meta, meta, annotated_data[k], meta_cache)

                    if check_dict(meta, annotated_data[k], exact_timestamps):
                        mad_transformed.append(meta)
                        np.savez(f'{root}/{save_path}{k}.npz', features=sliced_frame_features)
                    else:
                        discarded_datapoints_counter += 1
                else:
                    discarded_datapoints_counter += 1
            except Exception:
                traceback.print_exc()
                discarded_datapoints_counter += 1

        save_annotations(annotation_path, root, mad_transformed, annotations_filename)
        save_meta(meta_cache, root, annotation_path, meta_filename)
        print(f'Discarded {discarded_datapoints_counter} / {len(list(annotated_data.keys()))} datapoints')


def get_video_feats(root, frame_features_path):
    return h5py.File(f'{root}/{frame_features_path}', 'r')


def check_dict(meta, annotated_data, exact_timestamps):
    try:
        if not exact_timestamps:
            assert len(meta["saliency_scores"]) != 0, "saliency scores are zero"

        assert len(meta["relevant_windows"][0]) != 0
        assert meta["relevant_windows"][0][0] < meta["relevant_windows"][0][1]
        assert 0 <= meta["relevant_windows"][0][0] < meta["relevant_windows"][0][
            1], f'relevant window: {meta["relevant_windows"][0]}\noriginal data:\n{annotated_data}'
        return True
    except Exception:
        traceback.print_exc()


def save_annotations(annotation_path, root, mad_transformed, annotations_filename):
    save_path = root + "/" + annotation_path.split("/")[-1].split(".")[0] + annotations_filename
    with open(save_path, "w") as f:
        f.write("\n".join([json.dumps(e) for e in mad_transformed]))
    print(f'saved to: {save_path}')


def slice_window(frame_features, meta, rng, fps, max_v_l, exact_timestamps):
    f_max_v_l = max_v_l * fps  # qv samples at 0.5FPS, MAD at 5 FPS
    f_relevant_windows = np.multiply(meta["relevant_windows"][0], fps)  # relevant windows seconds -> frames @ 5 FPS
    if exact_timestamps:
        f_relevant_windows = [int(i) for i in f_relevant_windows]
    f_window_length = f_relevant_windows[1] - f_relevant_windows[0]

    # assert f_max_v_l > f_window_length, "moment longer then max sample length"

    random_window_offset = rng.random()
    assert f_max_v_l > f_window_length, f"window length ({f_window_length}) longer than max ({f_max_v_l}), discarding datapoint"
    f_left_offset = int(np.floor(random_window_offset * (f_max_v_l - f_window_length)))
    f_right_offset = int(f_max_v_l - f_window_length - f_left_offset)

    f_right_offset, f_left_offset = check_offsets(f_right_offset,
                                                  f_left_offset,
                                                  f_relevant_windows,
                                                  f_max_v_l,
                                                  frame_features)

    window = frame_features[
             int(f_relevant_windows[0] - f_left_offset):int(f_relevant_windows[1] + f_right_offset),
             :]

    if not exact_timestamps:
        meta = adjust_meta(meta,
                           f_left_offset,
                           f_window_length,
                           fps)

    # window = rng.choice(window, size=max_v_l, replace=False, axis=0, shuffle=False)
    return window, meta


def check_offsets(f_right_offset, f_left_offset, f_relevant_windows, f_max_v_l, frame_features):
    if f_relevant_windows[0] - f_left_offset < 0:
        f_right_offset += f_left_offset
        f_left_offset = 0
    if f_relevant_windows[1] + f_right_offset > frame_features.shape[0]:
        f_left_offset += f_right_offset
        f_right_offset = 0

    return f_right_offset, f_left_offset


def log_meta(old_meta, new_meta, annotated_data, meta_cache):
    meta_cache[old_meta["qid"]] = {"old_meta": old_meta, "new_meta": new_meta, "annotation": annotated_data}
    return meta_cache


def save_meta(meta_cache, root, annotation_path, meta_filename):
    print(f'saving meta log with length: {len(meta_cache)}')
    meta_save_path = f'{root}/{annotation_path.split("/")[-1].split(".")[0]}{meta_filename}'
    with open(meta_save_path, 'wb') as f:
        pickle.dump(meta_cache, f)
    print(f'saved metadata cache to: {meta_save_path}')


def adjust_meta(meta, f_left_offset, f_window_length, fps):
    window_start = int(np.floor(f_left_offset / fps)) if int(np.floor(f_left_offset / fps)) % 2 == 0 else int(
        np.floor(f_left_offset / fps)) - 1
    new_window = [[window_start, int(window_start + f_window_length / fps)]]
    new_clip_ids = [i for i in range(int(new_window[0][0] / 2), int(new_window[0][1] / 2))]

    meta["relevant_windows"] = new_window
    meta["relevant_clip_ids"] = new_clip_ids
    # meta.pop("duration")
    return meta


if __name__ == "__main__":
    run()

    # from utils.basic_utils import load_jsonl
    #
    # root = '/nfs/data3/goldhofer/mad_dataset'
    # clip_frame_features = get_video_feats(root)
    # annotation_paths = [f'{root}/annotations/MAD_val.json',
    #                     f'{root}/annotations/MAD_test.json', f'{root}/annotations/MAD_train.json']
    # for annotation_path in annotation_paths:
    #     save_path = root + "/annotations/" + annotation_path.split("/")[-1].split(".")[0] + "_transformed.json"
    #     meta = load_jsonl(save_path)
    #
    #     for m in tqdm(meta):
    #         if m["relevant_windows"][0][1] - m["relevant_windows"][0][0] >= 120:
    #             print(m["relevant_windows"])

    # with open(root + "/annotations/" + annotation_path.split("/")[-1].split(".")[0] + "_transformed.json", "w") as f:
    #    f.write("\n".join([json.dumps(e) for e in meta]))
    # print(f'saved to: {save_path}')
