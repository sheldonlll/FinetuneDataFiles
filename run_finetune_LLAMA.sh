#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=llama_finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

dataset_path=${project_dir}/data/MIT/train

mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  /root/LMFlow/examples/finetune.py \
    --model_name_or_path /root/autodl-tmp/v3/llama-7b/ \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 0.01 \
    --learning_rate 2e-5 \
    --block_size 128 \
    --per_device_train_batch_size 1 \
    --deepspeed examples/ds_config.json \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err



# accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml /root/LMFlow/examples/finetune.py     --model_name_or_path /root/autodl-tmp/v3/llama-7b/ --dataset_path /root/LMFlow/data/MIT/train --output_dir /root/LMFlow/output_models/llama_finetune --overwrite_output_dir     --num_train_epochs 0.01     --learning_rate 2e-5     --block_size 128     --per_device_train_batch_size 1 --bf16     --run_name finetune     --validation_split_percentage 0     --logging_steps 20     --do_train     --ddp_timeout 72000     --save_steps 5000     --dataloader_num_workers 1 | tee /root/LMFlow/log/llama_finetune//train.log 2> /root/LMFlow/log/llama_finetune//train.err