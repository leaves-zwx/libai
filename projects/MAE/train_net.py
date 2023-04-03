# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import sys

import numpy as np
import oneflow as flow
from utils.weight_convert import load_torch_checkpoint

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer

sys.path.append(".")
logger = logging.getLogger("libai.mae." + __name__)


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if try_get_key(cfg, "finetune") is not None:
            if cfg.finetune.enable is True:
                logger.info("Loading pretrained weight for finetuning")
                assert cfg.finetune.weight_style in ["oneflow", "pytorch"]
                if cfg.finetune.weight_style == "oneflow":
                    Checkpointer(model).load(cfg.finetune.path)
                elif cfg.finetune.weight_style == "pytorch":
                    model = load_torch_checkpoint(model, cfg, path=cfg.finetune.path, strict=False)
                else:
                    raise NotImplementedError(
                        "Only support loading oneflow & pytorch pretrained weight now."
                    )
        return model


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.eval_only = args.eval_only
    cfg.is_test_case = args.is_test_case

    cfg.dataloader.train.dataset[0].root = args.train_data_path
    cfg.dataloader.test[0].dataset.root = args.test_data_path

    cfg.finetune.path = args.pretrined_weight_path
    cfg.finetune.weight_style = args.pretrined_weight_style
    cfg.train.input_placement_device = 'cpu'
    cfg.train.dist.device_type = 'cpu'
    cfg.train.train_micro_batch_size = args.train_micro_batch_size
    cfg.train.num_accumulation_steps = args.num_accumulation_steps
    cfg.train.test_micro_batch_size = args.test_micro_batch_size
    cfg.train.dist.data_parallel_size = args.data_parallel_size
    cfg.train.dist.tensor_parallel_size = args.tensor_parallel_size
    cfg.train.dist.pipeline_parallel_size = args.pipeline_parallel_size
    cfg.train.evaluation.enabled = args.do_eval
    cfg.train.amp.enabled = args.amp
    cfg.train.activation_checkpoint.enabled = args.activation_checkpoint
    cfg.train.log_period = args.log_period
    cfg.train.warmup_ratio = args.warmup_ratio
    cfg.train.train_epoch = args.train_epoch
    cfg.train.train_iter = args.train_iter

    cfg.model.device = args.device

    default_setup(cfg, args)

    if args.fast_dev_run:
        cfg.train.train_epoch = 0
        cfg.train.checkpointer.period = 5
        cfg.train.train_iter = 10
        cfg.train.evaluation.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        cfg.eval_only = True
        tokenizer = None
        if try_get_key(cfg, "tokenization.setup", default=False):
            tokenizer = Trainer.build_tokenizer(cfg)
        model = Trainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.load_weight, resume=args.resume
        )
        if try_get_key(cfg, "train.graph.enabled", default=False):
            model = Trainer.build_graph(cfg, model, is_train=False)
        test_loader = Trainer.build_test_loader(cfg, tokenizer)
        if len(test_loader) == 0:
            logger.info("No dataset in dataloader.test, please set dataset for dataloader.test")
        _ = Trainer.test(cfg, test_loader, model)
        return

    # manual different seed for each rank
    seed_for_rank = cfg.train.seed + flow.env.get_rank()
    flow.manual_seed(seed_for_rank)
    if flow.cuda.is_available():
        flow.cuda.manual_seed(seed_for_rank)
    np.random.seed(seed_for_rank)
    random.seed(seed_for_rank)

    trainer = Trainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
