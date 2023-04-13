#!/usr/bin/env bash

export ENFLAME_ENABLE_TF32=true
export ONEFLOW_DEBUG_MODE=1

python3 -u projects/MAE/train_net.py \
        --config-file projects/MAE/configs/mae_finetune.py \
        --device gcu \
        --num_accumulation_steps 1 \
        --log_period 10 \
        --warmup_ratio 0.0