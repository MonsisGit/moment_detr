dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
exp_id=exp

######## paths
root=/nfs/data3/goldhofer/mad_dataset/
train_path=${root}annotations/MAD_train_SMNone_FPS5_CL25.6_L2True.json
eval_path=${root}annotations/MAD_val_SMNone_FPS5_CL25.6_L2True.json
results_root=${root}momentDETR_results
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/clip_frame_features_25.6_5FPS/)
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
lang_feat_path=CLIP_L14_language_tokens_features.h5


#### training
eval_split_name=val
v_feat_dim=768
t_feat_dim=768
bsz=256
cuda_visible_devices=0
lw_saliency=4
data_ratio=1
num_workers=8
n_epoch=100
lr=8e-4
lr_drop=100
clip_length=0.2
max_q_l=32
max_v_l=200
sheduler=cosnl_wrmp
##set for results tracking!
window_length=25.6
sampling_mode=none
fps=1/${clip_length}
eval_results_dir=${lang_feat_path:0:8}_bsz${bsz}_lr${lr}_lrd${lr_drop}_dr${data_ratio}_wl${window_length}_sm${sampling_mode}_fps${fps}_lws${lw_saliency}_${sheduler}


PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--num_workers ${num_workers} \
--n_epoch ${n_epoch} \
--exp_id ${exp_id} \
--eval_results_dir ${eval_results_dir} \
--lw_saliency ${lw_saliency} \
--data_ratio ${data_ratio} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--lang_feat_path ${lang_feat_path} \
--no_norm_vfeat \
--clip_length ${clip_length} \
--max_q_l ${max_q_l} \
--max_v_l ${max_v_l} \
--resume /nfs/data3/goldhofer/mad_dataset/momentDETR_results/CLIP_L14_bsz256_lr8e-4_lrd100_dr1_wl25.6_smnone_fps5_lws4/model_best.ckpt \
--scheduler=${sheduler}
${@:1}
