#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0
export ENFLAME_LOG_LEVEL=DEBUG
# export ENFLAME_LOG_DEBUG_MOD=OP,HLIR,Factor
export COMPILE_OPTIONS_MLIR_DBG="-pass-timing -pass-statistics -print-ir-before=hlir-first-pass -print-ir-after=hlir-last-pass -mlir-elide-elementsattrs-if-larger=409600 -log-output-path=./irdump/"
export ONEFLOW_DEBUG_MODE=1

python3 -u projects/MAE/train_net.py \
        --config-file projects/MAE/configs/mae_finetune.py \
        --device gcu \
        --num_accumulation_steps 1 \
        --n_gpus 1 \
        --log_period 10 \
        --warmup_ratio 0.0