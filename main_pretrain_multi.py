# main_pretrain_multi.py
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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

import torchvision
import torchvision.transforms as transforms

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# datasets
from dataset_classes.spectrogram_images import SpectrogramImages
from dataset_classes.iq_dataset import IQDatasetH5, pad_collate

# model registry (presets live here)
import models_multimodal_mae

# multimodal engine
from engine_pretrain_multi import train_one_epoch_multi, evaluate_iq, evaluate_vision


def get_args_parser():
    p = argparse.ArgumentParser('Multimodal MAE pre-training (Vision+IQ)', add_help=False)

    # Training
    p.add_argument('--epochs', default=400, type=int)
    p.add_argument('--accum_iter', default=2, type=int)
    p.add_argument('--batch_size_vis', default=256, type=int)
    p.add_argument('--batch_size_iq', default=256, type=int)

    # Model (keyword only)
    p.add_argument('--model', default='mae_vit_multi_micro', type=str,
                   help='Preset in models_multimodal_mae (e.g., mae_vit_multi_small/base/large)')

    # Common toggles passed into preset
    p.add_argument('--norm_pix_loss', action='store_true', default=False)
    p.add_argument('--norm_seg_loss', action='store_true', default=False)
    p.add_argument('--use_ant_mask', action='store_true', default=False)

    # Mask ratios (per modality)
    p.add_argument('--mask_ratio_vis', default=0.75, type=float)
    p.add_argument('--mask_ratio_iq', default=0.5, type=float)

    # Optimizer / LR schedule
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--blr', type=float, default=1e-3)
    p.add_argument('--min_lr', type=float, default=0.)
    p.add_argument('--warmup_epochs', type=int, default=40)

    # Data paths
    p.add_argument('--spect_paths',
                   default=['../datasets/spectrogram_dataset', '../datasets/spectrogram_iqengine_dataset'],
                   type=str, nargs='+',
                   help='List of spectrogram roots')
    p.add_argument('--iq_path', default='../datasets/train_256_100_256_22.h5', type=str)

    # Logging / env
    p.add_argument('--output_dir', default='./output_dir_multi')
    p.add_argument('--log_dir', default='./output_dir_multi')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--resume', default='')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--num_workers', default=0, type=int)
    p.add_argument('--pin_mem', action='store_true')
    p.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    p.set_defaults(pin_mem=True)

    # Distributed (kept for compatibility with your utils)
    p.add_argument('--world_size', default=1, type=int, help=argparse.SUPPRESS)
    p.add_argument('--local_rank', default=-1, type=int, help=argparse.SUPPRESS)
    p.add_argument('--dist_on_itp', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--dist_url', default='env://', help=argparse.SUPPRESS)
    return p


def build_spect_loader(args, img_size=224):
    transform_train = transforms.Compose([
        transforms.functional.pil_to_tensor,
        transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
        transforms.Lambda(lambda x: (x + 120) / (-0.5 + 120)),
        transforms.Resize((img_size, img_size), antialias=True,
                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.451], std=[0.043]),
    ])
    ds = SpectrogramImages(args.spect_paths, transform=transform_train)
    sampler = RandomSampler(ds)
    dl = DataLoader(
        ds, sampler=sampler, batch_size=args.batch_size_vis,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    return ds, dl


def build_iq_loaders(args):
    ds = IQDatasetH5(args.iq_path)
    ds, _ = random_split(ds, [0.5, 0.5])
    ds_tr, ds_val = random_split(ds, [0.7, 0.3])

    samp_tr = RandomSampler(ds_tr)
    samp_val = SequentialSampler(ds_val)

    dl_tr = DataLoader(
        ds_tr, sampler=samp_tr, batch_size=args.batch_size_iq,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=pad_collate,
    )
    dl_val = DataLoader(
        ds_val, sampler=samp_val, batch_size=args.batch_size_iq,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=pad_collate,
    )
    return (ds_tr, dl_tr), (ds_val, dl_val)


def main(args):
    misc.init_distributed_mode(args)
    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # --- data ---
    ds_vis, dl_vis = build_spect_loader(args)
    (ds_iq_tr, dl_iq_tr), (ds_iq_val, dl_iq_val) = build_iq_loaders(args)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # --- model from preset keyword ---
    # Pass only the common toggles; preset handles architecture specifics.
    model = models_multimodal_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        norm_seg_loss=args.norm_seg_loss,
        iq_use_ant_mask=args.use_ant_mask,
    ).to(device)
    model_without_ddp = model
    print("Model =", model_without_ddp)

    # --- optimizer & lr ---
    eff_batch_size = (args.batch_size_vis + args.batch_size_iq) * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size (sum): {eff_batch_size}")

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    t0 = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch_multi(
            model=model,
            loader_vis=dl_vis,
            loader_iq=dl_iq_tr,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            mask_ratio_vis=args.mask_ratio_vis,
            mask_ratio_iq=args.mask_ratio_iq,
            accum_iter=args.accum_iter,
            args=args,
        )

        val_iq = evaluate_iq(dl_iq_val, model, device, mask_ratio=args.mask_ratio_iq)
        val_vis = evaluate_vision(dl_vis, model, device, mask_ratio=args.mask_ratio_vis)

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_iq_{k}': v for k, v in val_iq.items()},
            **{f'val_vis_{k}': v for k, v in val_vis.items()},
            'epoch': epoch
        }
        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total = time.time() - t0
    print('Training time', str(datetime.timedelta(seconds=int(total))))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
