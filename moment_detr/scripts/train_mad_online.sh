dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
exp_id=exp

######## paths
root=/nfs/data3/goldhofer/mad_dataset/
train_path=${root}annotations/MAD_train_SMNone_FPS5_CL30_L2False_extsTrue.json
eval_path=${root}annotations/MAD_val_SMNone_FPS5_CL30_L2False_extsTrue.json
results_root=${root}momentDETR_results
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/)
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
lang_feat_path=CLIP_L14_language_tokens_features.h5


#### training
eval_split_name=val
v_feat_dim=768
t_feat_dim=768
bsz=128
cuda_visible_devices=0
data_ratio=1
num_workers=8
n_epoch=100
lr=4e-4
lr_drop=15
clip_length=0.2
max_q_l=100
#this must be fps * window length
max_v_l=150
sheduler=step_lr_warmup
max_es_cnt=10

## Losses
lw_saliency=4
set_cost_class=4   #"Class coefficient in the matching cost"
label_loss_coef=4
##set for results tracking!
window_length=30
sampling_mode=online
sampling_fps=5
eval_results_dir=${lang_feat_path:0:8}_bsz${bsz}_lr${lr}_lrd${lr_drop}_dr${data_ratio}_wl${window_length}_sm${sampling_mode}_fps${sampling_fps}_lws${lw_saliency}_lloss${label_loss_coef}_${sheduler}

if [ ${window_length} -gt ${max_v_l} ]; then
    echo "Window length larger than max_v_l"
    exit 1
fi


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
--clip_length ${clip_length} \
--max_q_l ${max_q_l} \
--max_v_l ${max_v_l} \
--scheduler=${sheduler} \
--label_loss_coef ${label_loss_coef} \
--set_cost_class ${set_cost_class} \
--eval_bsz ${bsz} \
--sampling_mode ${sampling_mode} \
--cuda_visible_devices ${cuda_visible_devices} \
--use_exact_ts \
--sampling_fps ${sampling_fps} \
--max_es_cnt ${max_es_cnt} \
${@:1}
