PYTHONPATH=$PYTHONPATH:. python utils/preprocess_mad_new.py \
--process_fraction 0.01 \
--sampling_mode fixed \
--anno_path=annotations/MAD_val_transformed.json \
--anno_save_path=annotations/MAD_val.json \
--clip_length_in_seconds=150 \
--l2_normalize=False \
--process_fraction=0.01 \
---sampling_fps=0.5 \
--sampling_mode=None \