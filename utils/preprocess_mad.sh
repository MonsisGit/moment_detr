dataset_fps=5
sampling_mode=fixed
l2_normalize=True
sampling_fps=1
clip_length_in_seconds=150
process_fraction=1
split=test

if [[ ${sampling_mode} = None ]] && [[ ${sampling_fps} != ${dataset_fps} ]]
then
  sampling_fps=${dataset_fps}
  echo "Sampling fps equals dataset fps since sampling is set to None"
fi


PYTHONPATH=$PYTHONPATH:. python utils/preprocess_mad.py \
--process_fraction ${process_fraction} \
--anno_path annotations/MAD_${split}.json \
--clip_length_in_seconds ${clip_length_in_seconds} \
--l2_normalize ${l2_normalize} \
--sampling_fps ${sampling_fps} \
--sampling_mode ${sampling_mode} \
--generated_feats_save_folder clip_frame_features_150_1FPS/ \
--anno_save_path annotations/MAD_${split}_SM${sampling_mode}_FPS${sampling_fps}_CL${clip_length_in_seconds}_L2${l2_normalize}.json \
--dataset_fps ${dataset_fps} \
