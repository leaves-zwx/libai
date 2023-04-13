#!/usr/bin/env bash

export ENFLAME_ENABLE_TF32=true
export ONEFLOW_DEBUG_MODE=1

python3 -u projects/MAE/train_net_test_case.py \
        --config-file projects/MAE/configs/mae_finetune.py \
        --device gcu \
        --warmup_ratio 0.0 \
        --is_test_case \
        --train_iter 1 \
        --pretrined_weight_path ./test_case/checkpoint/ori.ckp
