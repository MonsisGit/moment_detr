dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
exp_id=exp

######## paths
root=/nfs/data3/goldhofer/mad_dataset/
train_path=${root}annotations/MAD_train.json
eval_path=${root}annotations/MAD_val.json
results_root=${root}momentDETR_results_wCLIP
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/)
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
lang_feat_path=CLIP_L14_language_tokens_features.h5


#### training
eval_split_name=val
v_feat_dim=768
t_feat_dim=768
bsz=256
cuda_visible_devices=2
data_ratio=1
data_ratio_long_nlq=1
data_ratio_long_nlq_val_test=0.05
clip_topk=100
num_workers=8
n_epoch=400
lr=1e-4
lr_drop=15
clip_length=0.2
max_q_l=100
#this must be fps * window length
max_v_l=150
sheduler=reduce_plateau
max_es_cnt=100 #early stopping patience
use_warmup=True
nms_thd=0.3
num_queries=10
neg_window_ratio=0.05

## Losses
lw_saliency=4
set_cost_class=4   #"Class coefficient in the matching cost"
label_loss_coef=4
lw_cls=4
##set for results tracking!
window_length=30
sampling_mode=online
sampling_fps=5
eval_results_dir=${lang_feat_path:0:8}_bsz${bsz}_lr${lr}_wCLIP_topk${clip_topk}

if [ ${window_length} -gt ${max_v_l} ]; then
    echo "Window length larger than max_v_l"
    exit 1
fi


PYTHONPATH=$PYTHONPATH:. python moment_detr/clip_training.py \
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
--sampling_fps ${sampling_fps} \
--max_es_cnt ${max_es_cnt} \
--use_exact_ts \
--use_warmup \
--nms_thd ${nms_thd} \
--num_queries ${num_queries} \
--max_before_nms ${num_queries} \
--max_after_nms ${num_queries} \
--lw_cls ${lw_cls} \
--neg_window_ratio ${neg_window_ratio} \
--data_ratio_long_nlq ${data_ratio_long_nlq} \
--data_ratio_long_nlq_val_test ${data_ratio_long_nlq_val_test} \
--clip_topk ${clip_topk} \

${@:1}