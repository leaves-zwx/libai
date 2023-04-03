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
import os

import numpy as np
import oneflow as flow
from utils.weight_convert import load_torch_checkpoint

from libai.config import LazyConfig, default_argument_parser, try_get_key
from libai.engine import DefaultTrainer, default_setup
from libai.utils.checkpoint import Checkpointer
from libai.data.structures import DistTensorData

sys.path.append(".")
logger = logging.getLogger("libai.mae." + __name__)


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if try_get_key(cfg, "finetune") is not None:
            if cfg.finetune.enable is True:
                logger.info("Loading pretrained weight for finetuning")
                Checkpointer(model).load(cfg.finetune.path)
        return model

def cosSim(x, y):
    '''
    consin similarity
    '''
    tmp = np.sum(x*y)
    non = np.linalg.norm(x)*np.linalg.norm(y)
    return np.round(tmp/(float(non)+1e-30), decimals=9)

def save_dict(dict_to_save, path, step):
    if not os.path.exists(path):
        os.makedirs(path)
    for key, value in dict_to_save.items():
        np.save('{}/{}_{}.npy'.format(path, key, step), value)

def compare_dict(gcu_dict, path, step):
    if not os.path.exists(path):
        raise('error: values to be compared are not exists.')
    for key, gcu_value in gcu_dict.items():
        gcu_value = gcu_value.numpy()
        cuda_value = np.load('{}/{}_{}.npy'.format(path, key, step))
        try:
            np.testing.assert_allclose(cuda_value, gcu_value,
                                    rtol=1e-5,
                                    atol=1e-5,
                                    equal_nan=False)

        except Exception as e:
            print("error: {} compared with cuda failed at step {}.".format(key, step))
            cos_sim = cosSim(cuda_value, gcu_value)
            print("The cos sim of cuda and gcu:", cos_sim)
            if cos_sim < 0.99:
                raise(e)
            else:
                print("Pass for cosin similarity is high enough")
        else:
            print("{} compared with cuda successfully at step {}.".format(key, step))


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

    # manual different seed for each rank
    seed_for_rank = cfg.train.seed + flow.env.get_rank()
    flow.manual_seed(seed_for_rank)
    if flow.cuda.is_available():
        flow.cuda.manual_seed(seed_for_rank)
    np.random.seed(seed_for_rank)
    random.seed(seed_for_rank)

    trainer = Trainer(cfg)
    graph = trainer._trainer.graph
    model = graph.model
    module = model.to(flow.nn.Module)
    params = dict(reversed(list(module.named_parameters())))
    np_images = np.random.randn(32, 3, 224, 224).astype(np.float32)
    np_labels = np.random.randn(32, 1000).astype(np.float32)
    images = flow.tensor(np_images).to_global(
        placement=flow.placement(type=args.device, ranks=[0]),
        sbp=(flow.sbp.broadcast,))
    labels = flow.tensor(np_labels).to_global(
        placement=flow.placement(type=args.device, ranks=[0]),
        sbp=(flow.sbp.broadcast,))
    inputs = {'images': images, 'labels': labels}
    if args.device == 'cuda':
        Checkpointer(module, save_dir='./test_case/checkpoint').save('ori.ckp')
        save_dict(params, path = './test_case/params', step = 0)
        save_dict(inputs, path = './test_case/inputs', step = 0)
    logger.info('BEFORE RUN: compare params with cuda.')
    compare_dict(params, path = './test_case/params', step = 0)
    logger.info('BEFORE RUN: compare inputs with cuda.')
    compare_dict(inputs, path = './test_case/inputs', step = 0)
    logger.info('RUN ONE STEP...')
    loss_dict = graph(**inputs)
    if args.device == 'cuda':
        save_dict(params, path = './test_case/params', step = 1)
        save_dict(loss_dict, path = './test_case/losses', step = 1)
    logger.info('AFTER RUN: compare loss_dict with cuda.')
    compare_dict(loss_dict, path = './test_case/losses', step = 1)
    logger.info('AFTER RUN: compare params with cuda.')
    compare_dict(params, path = './test_case/params', step = 1)
    logger.info('OK: all compared successfully.')

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
