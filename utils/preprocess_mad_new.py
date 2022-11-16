import json
import h5py
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse


class MADdataset():

    def __init__(self, root, video_feat_file,
                 generated_feats_save_folder, log_folder,
                 dataset_fps):

        self.sampling_fps = None
        self.sampling_mode = None
        self.clip_length_in_frames = None
        self.clip_length_in_seconds = None
        self.dataset_fps = dataset_fps

        self.root = root
        self.rng = np.random.default_rng(42)
        self.generated_feats_save_path = generated_feats_save_folder
        Path(f'{root}{generated_feats_save_folder}').mkdir(parents=True, exist_ok=True)

        self.log_folder = log_folder
        self.annos = []
        self.old_annos = []
        print(f'Loading {root}{video_feat_file}')
        self.video_feats = h5py.File(f'{root}/{video_feat_file}', 'r')
        self.discarded_data_counter = 0

    def compute_annotations(self, anno_path, anno_save_path,
                            clip_length_in_seconds, l2_normalize,
                            process_fraction, sampling_mode, sampling_fps):
        '''
            The function processes the query features.
            Construct the moment annotations for training.
            Processed the language to obtain syntactic dependencies.
            Dump everything in the pickle file for speading up following run.
            INPUTS:
            annos: annotations loaded from json files
            cache: path to pickle file where to dump preprocessed annotations
            OUTPUTS:
            None.
        '''
        # compute the annotation data and dump it in a pickle file
        self.annos = []
        self.old_annos = []
        self.clip_length_in_seconds = clip_length_in_seconds
        assert (clip_length_in_seconds * self.dataset_fps) % 1 == 0, \
            f'with dataset FPS of {self.dataset_fps}, frames can only be extracted at {1 / self.dataset_fps} increments'
        self.clip_length_in_frames = int(clip_length_in_seconds * self.dataset_fps)
        self.discarded_data_counter = 0
        self.sampling_mode = sampling_mode
        self.sampling_fps = sampling_fps
        if self.sampling_mode == "None":
            print(f'###################################################################################################'
                  f'\nSampling mode is {self.sampling_mode},'
                  f' sampling FPS ({self.sampling_fps}) has no influence.'
                  f' If you want to sample, set sampling mode to something else, e.g. fixed'
                  f'\n###################################################################################################')

        annos = json.load(open(f'{self.root}{anno_path}', 'r'))

        print(f'\nUsing sampling mode: {self.sampling_mode},'
              f'\nSaving npz files to {self.root}{self.generated_feats_save_path}')
        print(f'\n\nProcessing {anno_path} ..')

        for k, anno in tqdm(list(annos.items())[0:int(len(annos.items()) * process_fraction)]):
            # Unpack Info ----------------------------------------------------------------

            movie = anno['movie']
            duration = anno['movie_duration']
            timestamp = anno['ext_timestamps']
            sentence = anno['sentence']

            # Process gt annotations -----------------------------------------------------
            if timestamp[0] < timestamp[1]:
                moment = [max(timestamp[0], 0), min(timestamp[1], duration)]

                start = int(moment[0] * self.dataset_fps)
                stop = int(moment[1] * self.dataset_fps)

                # frames_idx is in frame space
                frames_idx = [start, stop]

                # Save preprocessed annotations ----------------------------------------------
                temp_dict = {
                    'id': k,
                    'movie': movie,
                    'moment': moment,
                    'frames_idx': frames_idx,
                    'sentence': sentence,
                    'movie_duration': duration,
                }
                video_features, start_moment, stop_moment = self._get_video_features(temp_dict, l2_normalize)
                dump_dict = {
                    'id': k,
                    'relevant_windows': [[start_moment, stop_moment]],
                    'query': sentence,
                    'duration': self.clip_length_in_seconds,
                }

                self.old_annos.append(temp_dict)
                self.annos.append(dump_dict)
                np.savez(f'{self.root}{self.generated_feats_save_path}{k}.npz', features=video_features)

            else:
                self.discarded_data_counter += 1

                # save to file
        self._save_annos(anno_save_path)
        print(f'Discarded {self.discarded_data_counter} / {len(list(annos.items()))} data')

    def _save_annos(self, anno_path):
        anno_save_path = f'{self.root}{anno_path}'
        with open(anno_save_path, "w") as f:
            f.write("\n".join([json.dumps(e) for e in self.annos]))
        print(f'Saved annotations to: {anno_save_path}')

        Path(f'{self.root}{self.log_folder}').mkdir(parents=True, exist_ok=True)
        anno_save_path = f'{self.root}{self.log_folder}/{anno_path.split("/")[-1].split(".json")[0]}_log.json'
        with open(anno_save_path, "w") as f:
            f.write("\n".join([json.dumps(e) for e in self.old_annos]))
        print(f'Saved old annotations log to: {anno_save_path}')

    def _get_video_features(self, anno, l2_normalize):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features
            OUTPUTS:
            feat: movie features
            start_moment: start moment in seconds
            end_moment: end moment in seconds
        '''

        start_idx, stop_idx = anno['frames_idx']
        num_frames = stop_idx - start_idx

        if num_frames < self.clip_length_in_frames:
            offset = random.sample(range(0, self.clip_length_in_frames - num_frames, 1), 1)[0]
        else:
            center = (start_idx + stop_idx) / 2
            offset = int(round(center / 2))

        # Compute features for window
        start_window = max(start_idx - offset, 0)
        stop_window = start_window + self.clip_length_in_frames

        if not stop_window <= anno['movie_duration'] * self.dataset_fps:
            stop_window = int(anno['movie_duration'] * self.dataset_fps)
            start_window = stop_window - self.clip_length_in_frames

        feats = self.video_feats[anno['movie']][start_window:stop_window]

        assert feats.shape[0] == self.clip_length_in_frames

        # Compute moment position within the window in seconds
        start_moment = max((start_idx - start_window) / self.dataset_fps, 0)
        stop_moment = min((stop_idx - start_window) / self.dataset_fps, self.clip_length_in_seconds)

        if l2_normalize:
            feats = self._l2_normalize_np_array(feats)

        feats = self._sampling(feats)
        return feats, start_moment, stop_moment

    def _sampling(self, feats):
        if self.sampling_mode == 'None':
            return feats
        elif self.sampling_mode == 'fixed':
            feats = feats[::int(5 / self.sampling_fps)]
        elif self.sampling_mode == 'random':
            num_frames = int(feats.shape[0] / (5 / self.sampling_fps))
            feats = self.rng.choice(feats, size=num_frames, replace=False, axis=0, shuffle=False)
        elif self.sampling_mode == 'pooling':
            num_frames_to_pool = int(5 / self.sampling_fps)
            first_dim = feats.shape[0] / num_frames_to_pool
            feats = feats.reshape(int(first_dim), num_frames_to_pool, -1).mean(axis=1)
        return feats

    def _l2_normalize_np_array(self, feats, eps=1e-5):
        """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='/nfs/data3/goldhofer/mad_dataset/')
    parser.add_argument("--video_feat_file", type=str, default='CLIP_L14_frames_features_5fps.h5')
    parser.add_argument("--generated_feats_save_folder", default="clip_frame_features/")
    parser.add_argument("--log_folder", type=str, default='meta_log')
    parser.add_argument("--anno_path", type=str, default="annotations/MAD_val.json")
    parser.add_argument("--anno_save_path", default="annotations/MAD_val_transformed.json")

    parser.add_argument("--dataset_fps", type=int, default=5)
    parser.add_argument("--clip_length_in_seconds", type=float, default=150.0)
    parser.add_argument("--l2_normalize", type=bool, default=False)

    parser.add_argument("--process_fraction", type=float, default=1.0)
    parser.add_argument("--sampling_fps", type=float, default=0.5)
    parser.add_argument("--sampling_mode", type=str, default='None', choices=["None", "random", "fixed", "pooling"])
    args = parser.parse_args()

    preprocessor = MADdataset(root=args.root,
                              video_feat_file=args.video_feat_file,
                              generated_feats_save_folder=args.generated_feats_save_folder,
                              log_folder=args.log_folder,
                              dataset_fps=args.dataset_fps)

    preprocessor.compute_annotations(anno_path=args.anno_path,
                                     anno_save_path=args.anno_save_path,
                                     clip_length_in_seconds=args.clip_length_in_seconds,
                                     l2_normalize=args.l2_normalize,
                                     process_fraction=args.process_fraction,
                                     sampling_fps=args.sampling_fps,
                                     sampling_mode=args.sampling_mode)
