sampling_mode=None
l2_normalize=True
sampling_fps=0.5
clip_length_in_seconds=25
process_fraction=1

PYTHONPATH=$PYTHONPATH:. python utils/preprocess_mad_new.py \
--process_fraction ${process_fraction} \
--anno_path annotations/MAD_val.json \
--clip_length_in_seconds ${clip_length_in_seconds} \
--l2_normalize ${l2_normalize} \
--sampling_fps ${sampling_fps} \
--sampling_mode ${sampling_mode} \
--generated_feats_save_folder clip_frame_features_150_dense/ \
--anno_save_path annotations/MAD_val_SM${sampling_mode}_FPS${sampling_fps}_CL${clip_length_in_seconds}_L2${l2_normalize}.json \
