dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip
exp_id=exp

######## data paths
root=/nfs/data3/goldhofer/mad_dataset/
train_path=${root}annotations/MAD_train_SMNone_FPS5_CL30_exts_balanced.json
eval_path=${root}annotations/MAD_val_SMNone_FPS5_CL30_exts_balanced.json
eval_path_long_nlq=${root}annotations/MAD_test.json
#set
eval_results_dir=CLIP_L14_bsz256_lr1e-4_dr1_wl30_fps5_lws4_lloss4_closs4_ret_tok_prop
ckpt_path=${root}momentDETR_results/${eval_results_dir}/model_best.ckpt
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/)
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
results_root=${root}momentDETR_results
eval_split_name=test

######## setup video+text features
v_feat_dim=768
t_feat_dim=768
device=0
sampling_fps=5
nms_thd=0.3
data_ratio=0.005
num_workers=8

PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
  --dset_name ${dset_name} \
  --ctx_mode ${ctx_mode} \
  --train_path ${train_path} \
  --eval_path ${eval_path} \
  --eval_split_name ${eval_split_name} \
  --v_feat_dirs ${v_feat_dirs[@]} \
  --v_feat_dim ${v_feat_dim} \
  --t_feat_dir ${t_feat_dir} \
  --t_feat_dim ${t_feat_dim} \
  --results_root ${results_root} \
  --num_workers ${num_workers} \
  --exp_id ${exp_id} \
  --resume ${ckpt_path} \
  --device ${device} \
  --eval_results_dir ${eval_results_dir} \
  --sampling_fps ${sampling_fps} \
  --use_exact_ts \
  --nms_thd ${nms_thd} \
  --data_ratio ${data_ratio} \
  --eval_path_long_nlq ${eval_path_long_nlq}

${@:1}
