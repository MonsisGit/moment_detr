import os
import json
import h5py
import random
import numpy as np
import pickle as pk
from tqdm import tqdm


class MADdataset():

    def __init__(self, anno_file, anno_save_file, root, feat_file,
                 fps, generated_feats_save_path, log_folder, clip_length_in_seconds):

        self.max_words = 0
        self.fps = fps
        self.root = root
        self.generated_feats_save_path = generated_feats_save_path
        self.log_folder = log_folder
        self.clip_length_in_seconds = clip_length_in_seconds
        self.clip_length_in_frames = self.clip_length_in_seconds * self.fps
        self.anno_save_file = anno_save_file
        # load annotation file
        self.annos = []
        self.old_annos = []
        annos = json.load(open(f'{root}{anno_file}', 'r'))
        self._compute_annotations(annos)

        # Get correct data for language
        if lang_feat_type == 'clip' and os.path.exists(lang_feat_file):
            self.load_clip_lang_feats(lang_feat_file)
        else:
            raise ValueError('Select a correct type of lang feat - Glove is deprecated.')

        self.movies = {a['movie']: a['movie_duration'] for a in self.annos}
        self.feats = movie2feats(feat_file, self.movies.keys())

        if self.max_words > 50:
            self.max_words = 50

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['movie_duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        '''
            return moment duration in seconds
        '''
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['movie']

    def get_number_of_windows(self, idx):
        movie = self.annos[idx]['movie']
        return len(self.windows[movie])

    def load_clip_lang_feats(self, file):
        with h5py.File(file, 'r') as f:
            for i, anno in enumerate(self.annos):
                lang_feat = f[anno['id']][:]
                self.annos[i]['query'] = torch.from_numpy(lang_feat).float()
                self.annos[i]['wordlen'] = len(lang_feat)

        self.max_words = max([a['wordlen'] for a in self.annos])

    def _compute_annotations(self, annos):
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

                self.old_annos.append(temp_dict)
                self.annos.append(dump_dict)
                np.savez(f'{self.root}/{self.generated_feats_save_path}{k}.npz', features=video_features)

        # save to file
        self._save_annos()

    def _save_annos(self):
        anno_save_path = f'{self.root}{self.anno_save_file}'
        with open(anno_save_path, "w") as f:
            f.write("\n".join([json.dumps(e) for e in self.annos]))
        print(f'saved annotations to: {anno_save_path}')

        anno_save_path = f'{self.root}{self.anno_save_file.split(".json")[0]}_log.json'
        with open(anno_save_path, "w") as f:
            f.write("\n".join([json.dumps(e) for e in self.old_annos]))
        print(f'saved old annotations log to: {anno_save_path}')

    def _get_video_features(self, anno, movie):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features
            OUTPUTS:
            feat: movie features
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

        feats = self.feats[movie][start_window:stop_window]

        assert feats.shape[0] == self.clip_length_in_frames

        # Compute moment position within the window
        start_moment = max((start_idx - start_window) / self.fps, 0)
        stop_moment = min((stop_idx - start_window) / self.fps, self.clip_length_in_seconds)

        return feats, start_moment, stop_moment


if __name__ == "__main__":
    preprocessor = MADdataset(anno_file=f'annotations/MAD_val.json',
                              anno_save_file=f'annotations/MAD_val_t.json',
                              root='/nfs/data3/goldhofer/mad_dataset/',
                              feat_file='CLIP_L14_frames_features_5fps.h5',
                              generated_feats_save_path='clip_frame_features_transformed_exact/',
                              log_folder='meta_log/',
                              fps=5,
                              clip_length_in_seconds=150)
