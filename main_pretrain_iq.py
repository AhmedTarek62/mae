# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from dataset_classes.iq_dataset import IQDataset, pad_collate, IQResampler, IQDatasetH5
from torch.utils.data import DataLoader, RandomSampler, random_split, SequentialSampler

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_iq
from engine_pretrain_iq import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training on I/Q', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations '
                             '(for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_iq_debug', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--segment_duration', default=1e-3, type=float,
                        help='duration of every segment')

    parser.add_argument('--segment_len', default=4096, type=int,
                        help='length used in segmenting IQ signal')

    parser.add_argument('--hop', default=4096, type=int,
                        help='hop size used in segmenting IQ signal')

    parser.add_argument('--mask_ratio', default=0.1, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--use_ant_mask', action='store_true', default=False)

    parser.add_argument('--norm_seg_loss', action='store_true', default=False, help='Normalize per-segment loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='',
                        type=str, help='dataset path(s)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--local_rank', default=-1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--dist_on_itp', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dist_url', default='env://', help=argparse.SUPPRESS)

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # resampler = IQResampler(segment_len=args.segment_len, segment_duration=args.segment_duration)
    # dataset = IQDataset(args.data_path, resampler=resampler, stats='stats.json')
    # dataset_train, dataset_val = random_split(dataset, [0.8, 0.2])
    dataset = IQDatasetH5(args.data_path)
    dataset, _ = random_split(dataset, [0.5, 0.5])
    dataset_train, dataset_val = random_split(dataset, [0.7, 0.3])
    print(dataset_train)

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=pad_collate,  # <<—— time-only padding + time_mask
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=pad_collate,  # <<—— time-only padding + time_mask
    )

    # define the model
    model = models_mae_iq.__dict__[args.model](norm_seg_loss=args.norm_seg_loss, use_ant_mask=args.use_ant_mask)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        val_stats = evaluate(data_loader_val, model, device, args.mask_ratio)
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
