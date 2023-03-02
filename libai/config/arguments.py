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


import argparse
import sys


def default_argument_parser(epilog=None):
    """Create a parser with some common arguments used by libai users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    {sys.argv[0]} --config-file cfg.yaml

Change some config options:
    $ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 12345 \
    {sys.argv[0]} --config-file cfg.yaml train.load_weight=/path/to/weight.pth optim.lr=0.001

Run on multiple machines:
    (machine0)$ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr <URL> --master_port 12345 \
    {sys.argv[0]} --config-file cfg.yaml

    $ python3 -m oneflow.distributed.launch \
    --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr <URL> --master_port 12345 \
    {sys.argv[0]} --config-file cfg.yaml
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run several batches of train, eval and test to find any bugs, "
        "(ie: a sort of unit test)",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "path.key value" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--train_data_path',
        default='/data/datasets/imagenet_raw/',
        help='train dataset path')
    parser.add_argument(
        '--test_data_path',
        default='/data/datasets/imagenet_raw/',
        help='test dataset path')
    parser.add_argument(
        '--pretrined_weight_path',
        default='mae_pretrain_vit_base.pth',
        help='pretrined weight path')
    parser.add_argument(
        '--pretrined_weight_style',
        default='pytorch',
        choices=["pytorch", "oneflow"],
        help='pretrined weight style')
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "gcu"],
        help='device'
    )
    parser.add_argument(
        "--train_micro_batch_size",
        default=32,
        type=int,
        help='train batch size per device'
    )
    parser.add_argument(
        "--test_micro_batch_size",
        default=32,
        type=int,
        help='test batch size per device'
    )
    parser.add_argument(
        "--num_accumulation_steps",
        default=4,
        type=int,
        help='num accumulation steps to update'
    )
    parser.add_argument(
        "--n_gpus",
        default=8,
        type=int,
        help='num devices'
    )
    parser.add_argument(
        "--log_period",
        default=20,
        type=int,
        help='log frequency'
    )
    parser.add_argument(
        "--do_eval",
        default=False,
        action='store_true',
        help="skip evaulation")
    parser.add_argument(
        "--amp",
        default=False,
        action='store_true',
        help="enable amp training")
    return parser
