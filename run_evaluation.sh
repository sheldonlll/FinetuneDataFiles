#!/bin/bash
# /root/LMFlow/output_models/finetune
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path /root/LMFlow/output_models/finetune \
    --dataset_path data/MIT/test \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy