import os
import json
import h5py
import random
import numpy as np
import pickle as pk
from tqdm import tqdm
import traceback

class MADdataset():

    def __init__(self, root, video_feat_file,
                 fps, generated_feats_save_path, log_folder, clip_length_in_seconds):

        self.max_words = 0
        self.fps = fps
        self.root = root
        self.generated_feats_save_path = generated_feats_save_path
        self.log_folder = log_folder
        self.clip_length_in_seconds = clip_length_in_seconds
        self.clip_length_in_frames = self.clip_length_in_seconds * self.fps
        # load annotation file
        self.annos = []
        self.old_annos = []
        assert os.path.exists(video_feat_file)
        self.video_feats = h5py.File(f'{root}/{video_feat_file}', 'r')
        self.discarded_data_counter = 0

    def compute_annotations(self, anno_path, anno_save_path):
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
        self.discarded_data_counter = 0

        annos = json.load(open(f'{self.root}{anno_path}', 'r'))

        print(f'Processing {anno_path} ..')
        for k, anno in tqdm(annos.items()):
            # Unpack Info ----------------------------------------------------------------

            movie = anno['movie']
            duration = anno['movie_duration']
            timestamp = anno['ext_timestamps']
            sentence = anno['sentence']

            # Process gt annotations -----------------------------------------------------
            if timestamp[0] < timestamp[1]:
                moment = [max(timestamp[0], 0), min(timestamp[1], duration)]

                start = int(moment[0] * self.fps)
                stop = int(moment[1] * self.fps)

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
                video_features, start_moment, stop_moment = self._get_video_features(temp_dict, movie)
                dump_dict = {
                    'id': k,
                    'moment': moment,
                    'relevant_windows': [[start_moment, stop_moment]],
                    'query': sentence,
                    'duration': self.clip_length_in_seconds,
                    "relevant_clip_ids": [i for i in
                                          range(int(start_moment / 2), int(stop_moment / 2))],
                    "saliency_scores": [[0, 0, 0] for _ in
                                        range(int(start_moment / 2), int(stop_moment / 2))]
                }

                if self._check_dict(temp_dict,dump_dict):
                    self.old_annos.append(temp_dict)
                    self.annos.append(dump_dict)
                    np.savez(f'{self.root}/{self.generated_feats_save_path}{k}.npz', features=video_features)
                else:
                    self.discarded_data_counter += 1
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

        anno_save_path = f'{self.root}{anno_path.split(".json")[0]}_log.json'
        with open(anno_save_path, "w") as f:
            f.write("\n".join([json.dumps(e) for e in self.old_annos]))
        print(f'Saved old annotations log to: {anno_save_path}')


    def _check_dict(self, old_annos, annos, exact_timestamps=False):
        try:
            if not exact_timestamps:
                assert len(old_annos["saliency_scores"]) != 0, "saliency scores are zero"

            assert len(old_annos["relevant_windows"][0]) != 0
            assert old_annos["relevant_windows"][0][0] < old_annos["relevant_windows"][0][1]
            assert 0 <= old_annos["relevant_windows"][0][0] < old_annos["relevant_windows"][0][
                1], f'relevant window: {old_annos["relevant_windows"][0]}\noriginal data:\n{annos}'
            return True
        except Exception:
            traceback.print_exc()
            return False


    def _get_video_features(self, anno, movie):
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

        if not stop_window <= anno['movie_duration'] * self.fps:
            stop_window = int(anno['movie_duration'] * self.fps)
            start_window = stop_window - self.clip_length_in_frames

        feats = self.video_feats[movie][start_window:stop_window]

        assert feats.shape[0] == self.clip_length_in_frames

        # Compute moment position within the window
        start_moment = max((start_idx - start_window) / self.fps, 0)
        stop_moment = min((stop_idx - start_window) / self.fps, self.clip_length_in_seconds)

        return feats, start_moment, stop_moment


if __name__ == "__main__":
    preprocessor = MADdataset(root='/nfs/data3/goldhofer/mad_dataset/',
                              video_feat_file='CLIP_L14_frames_features_5fps.h5',
                              generated_feats_save_path='clip_frame_features_transformed_exact/',
                              log_folder='meta_log/',
                              fps=5,
                              clip_length_in_seconds=150)

    preprocessor.compute_annotations(anno_path=f'annotations/MAD_val.json',
                                     anno_save_path='annotations/MAD_val_transformed.json')
