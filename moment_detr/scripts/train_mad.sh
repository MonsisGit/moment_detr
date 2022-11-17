dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
exp_id=exp

######## data paths
root=/nfs/data3/goldhofer/mad_dataset/
train_path=${root}annotations/MAD_train_SMfixed_FPS0.5_CL180_L2True.json
eval_path=${root}annotations/MAD_val_SMfixed_FPS0.5_CL180_L2True.json
results_root=${root}momentDETR_results

eval_split_name=val

######## setup video+text features
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/clip_frame_features_180_0.5FPS/)
v_feat_dim=768
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
t_feat_dim=768
#### training
bsz=256
cuda_visible_devices=0
lw_saliency=0
data_ratio=1
num_workers=8
n_epoch=100
lr=8e-4
lr_drop=20
lang_feat_path=CLIP_L14_language_tokens_features.h5

##set for results tracking!
clip_length=180
sampling_mode=fixed
fps=0.5
eval_results_dir=${lang_feat_path:0:8}_bsz${bsz}_lr${lr}_lrd${lr_drop}_dr${data_ratio}_cl${clip_length}_sm${sampling_mode}_fps${fps}


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
--cuda_visible_devices ${cuda_visible_devices} \
--eval_results_dir ${eval_results_dir} \
--lw_saliency ${lw_saliency} \
--data_ratio ${data_ratio} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--lang_feat_path ${lang_feat_path} \
--no_norm_vfeat
${@:1}
