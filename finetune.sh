# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

# option 1, download model automatically
model_name_or_model_dir="/data/SenseVoice/model"

# option 2, download model by git
# local_path_root=${workspace}/modelscope_models
# mkdir -p ${local_path_root}/${model_name_or_model_dir}
# git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
# model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}
# /data/FunASR/funasr/bin/train_ds.py

# data dir, which contains: train.json, val.json
train_data=${workspace}/data/train.jsonl
val_data=${workspace}/data/val.jsonl

# exp output dir
output_dir="./outputs"
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/deepspeed_conf/ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes 1 \
    --nproc_per_node 1 \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"
# DISTRIBUTED_ARGS="
#     --nnodes 1 \
#     --nproc_per_node 1 \
#     --node_rank ${RANK:-0} \
#     --master_addr ${MASTER_ADDR:-127.0.0.1} \
#     --master_port ${MASTER_PORT:-26669}
#     --rdzv_endpoint localhost:29411
# "

echo $DISTRIBUTED_ARGS

# funasr trainer path
train_tool="../FunASR/funasr/bin/train_ds.py"

torchrun $DISTRIBUTED_ARGS \
${train_tool} \
++model="${model_name_or_model_dir}" \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=512  \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=20 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++train_conf.use_deepspeed=false \
++train_conf.deepspeed_config=${deepspeed_config} \
++optim_conf.lr=0.0002 \
++output_dir="${output_dir}" &> ${log_file}