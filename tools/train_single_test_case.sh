#!/usr/bin/env bash

export COMPILE_OPTIONS_MLIR_DBG="-pass-timing -pass-statistics -print-ir-before=hlir-first-pass -print-ir-after=hlir-last-pass -mlir-elide-elementsattrs-if-larger=409600 -log-output-path=./irdump/"
export ONEFLOW_DEBUG_MODE=1

python3 -u projects/MAE/train_net_test_case.py \
        --config-file projects/MAE/configs/mae_finetune.py \
        --device gcu \
        --warmup_ratio 0.0 \
        --is_test_case \
        --train_iter 1 \
        --pretrined_weight_path ./test_case/checkpoint/ori.ckp
