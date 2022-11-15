dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
results_root=results
exp_id=exp

######## data paths
train_path=/nfs/data3/goldhofer/mad_dataset/annotations/MAD_train_transformed.json
eval_path=/nfs/data3/goldhofer/mad_dataset/annotations/MAD_val_transformed.json
eval_split_name=val

######## setup video+text features
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/clip_frame_features_transformed_dense/)
v_feat_dim=768
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
t_feat_dim=768
#### training
bsz=128
cuda_visible_devices=1
eval_results_dir=L14_5FPS
lw_saliency=0
data_ratio=0.05
num_workers=16
n_epoch=400
lr=0.001
lr_drop=50

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
${@:1}
