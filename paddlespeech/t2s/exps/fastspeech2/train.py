# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import os
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import yaml
from paddle import DataParallel
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.am_batch_fn import fastspeech2_multi_spk_batch_fn
from paddlespeech.t2s.datasets.am_batch_fn import fastspeech2_single_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Evaluator
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Updater
from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.extensions.visualizer import VisualDL
from paddlespeech.t2s.training.optimizer import build_optimizers
from paddlespeech.t2s.training.seeding import seed_everything
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.t2s.utils import str2bool


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    if (not paddle.is_compiled_with_cuda()) or args.ngpu == 0:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")
    world_size = paddle.distributed.get_world_size()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    print(
        f"rank: {dist.get_rank()}, pid: {os.getpid()}, parent_pid: {os.getppid()}",
    )
    fields = [
        "text", "text_lengths", "speech", "speech_lengths", "durations",
        "pitch", "energy"
    ]
    converters = {"speech": np.load, "pitch": np.load, "energy": np.load}
    spk_num = None
    if args.speaker_dict is not None:
        print("multiple speaker fastspeech2!")
        collate_fn = fastspeech2_multi_spk_batch_fn
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        fields += ["spk_id"]
    elif args.voice_cloning:
        print("Training voice cloning!")
        collate_fn = fastspeech2_multi_spk_batch_fn
        fields += ["spk_emb"]
        converters["spk_emb"] = np.load
    else:
        print("single speaker fastspeech2!")
        collate_fn = fastspeech2_single_spk_batch_fn
    print("spk_num:", spk_num)

    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for training and validation
    with jsonlines.open(args.train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=fields,
        converters=converters, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=fields,
        converters=converters, )

    # collate function and dataloader

    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True)

    print("samplers done!")

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_mels
    model = FastSpeech2(
        idim=vocab_size, odim=odim, spk_num=spk_num, **config["model"])
    if args.pretrained_model != '':
        state_dict = paddle.load(args.pretrained_model)
        state_dict = state_dict.get('main_params', state_dict)
        model.set_state_dict(state_dict)
        print("pretrained fastspeech:", args.pretrained_model)
    trainable_model = model
    if world_size > 1:
        model = DataParallel(model)
    print("model done!")

    if config.get("train_diffusion_only", False):
        assert args.pretrained_model != '', "'train_diffusion_only' need a pretrained fastspeech2"
        trainable_model = trainable_model.diffusion
        if world_size > 1:
            trainable_model = DataParallel(trainable_model)
    else:
        trainable_model = model
    optimizer = build_optimizers(trainable_model, **config["optimizer"])
    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        config_name = args.config.split("/")[-1]
        # copy conf to output_dir
        shutil.copyfile(args.config, output_dir / config_name)

    if "enable_speaker_classifier" in config.model:
        enable_spk_cls = config.model.enable_speaker_classifier
    else:
        enable_spk_cls = False

    updater = FastSpeech2Updater(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        output_dir=output_dir,
        enable_spk_cls=enable_spk_cls,
        **config["updater"], )

    use_epoch = "max_epoch" in config
    if use_epoch:
        trainer_trigger = (config.max_epoch, "epoch")
        evaluator_trigger = (1, "epoch")
        snapshot_trigger = (1, "epoch")
    else:
        trainer_trigger = (config.train_max_steps, "iteration")
        evaluator_trigger = (config.eval_interval_steps, "iteration")
        snapshot_trigger = (config.save_interval_steps, "iteration")

    trainer = Trainer(updater, trainer_trigger, output_dir)

    evaluator = FastSpeech2Evaluator(
        model,
        dev_dataloader,
        output_dir=output_dir,
        enable_spk_cls=enable_spk_cls,
        **config["updater"], )

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=evaluator_trigger)
        trainer.extend(VisualDL(output_dir), trigger=(1, "iteration"))
    trainer.extend(
        Snapshot(max_size=config.num_snapshots), trigger=snapshot_trigger)
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a FastSpeech2 model.")
    parser.add_argument("--config", type=str, help="fastspeech2 config file.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker-dict",
        type=str,
        default=None,
        help="speaker id map file for multiple speaker model.")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default='',
        help="pretrained fastspeech2 model path.")

    parser.add_argument(
        "--voice-cloning",
        type=str2bool,
        default=False,
        help="whether training voice cloning model.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)
    print(
        f"master see the word size: {dist.get_world_size()}, from pid: {os.getpid()}"
    )

    # dispatch
    if args.ngpu > 1:
        dist.spawn(train_sp, (args, config), nprocs=args.ngpu)
    else:
        train_sp(args, config)


if __name__ == "__main__":
    main()
