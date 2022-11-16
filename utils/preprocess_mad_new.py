import json
import h5py
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path


class MADdataset():

    def __init__(self, root, video_feat_file,
                 generated_feats_save_folder, log_folder):

        self.sampling_fps = None
        self.sampling_mode = None
        self.clip_length_in_frames = None
        self.clip_length_in_seconds = None
        self.dataset_fps = None

        self.root = root
        self.rng = np.random.default_rng(42)
        self.generated_feats_save_path = generated_feats_save_folder
        Path(generated_feats_save_folder).mkdir(parents=True, exist_ok=True)

        self.log_folder = log_folder
        self.annos = []
        self.old_annos = []
        print(f'Loading {root}/{video_feat_file}')
        self.video_feats = h5py.File(f'{root}/{video_feat_file}', 'r')
        self.discarded_data_counter = 0

    def compute_annotations(self, anno_path, anno_save_path, dataset_fps,
                            clip_length_in_seconds, l2_normalize,
                            process_fraction, sampling_mode, sampling_fps):
        '''
            The function processes the annotations computing language tokenizationa and query features.
            Construct the moment annotations for training and the target iou2d map.
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
        self.dataset_fps = dataset_fps
        self.clip_length_in_frames = clip_length_in_seconds * dataset_fps
        self.discarded_data_counter = 0
        self.sampling_mode = sampling_mode
        self.sampling_fps = sampling_fps

        annos = json.load(open(f'{self.root}{anno_path}', 'r'))

        print(f'Processing {anno_path} ..')
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
                    'moment': moment,
                    'relevant_windows': [[start_moment, stop_moment]],
                    'query': sentence,
                    'duration': self.clip_length_in_seconds,
                }

                self.old_annos.append(temp_dict)
                self.annos.append(dump_dict)
                np.savez(f'{self.root}/{self.generated_feats_save_path}{k}.npz', features=video_features)

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
        anno_save_path = f'{self.root}{self.log_folder}{anno_path.split(".json")[0]}_log.json'
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
    preprocessor = MADdataset(root='/nfs/data3/goldhofer/mad_dataset/',
                              video_feat_file='CLIP_L14_frames_features_5fps.h5',
                              generated_feats_save_folder='clip_frame_features_transformed_exact/',
                              log_folder='meta_log/')

    preprocessor.compute_annotations(anno_path=f'annotations/MAD_val.json',
                                     anno_save_path='annotations/MAD_val_transformed.json',
                                     dataset_fps=5,
                                     clip_length_in_seconds=150,
                                     l2_normalize=False,
                                     process_fraction=0.01,
                                     sampling_fps=10,
                                     sampling_mode='None')
