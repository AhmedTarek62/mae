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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split, Subset

import torchvision
import torchvision.transforms as transforms

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lr_decay import build_wd_groups

from util.grad_probe import sample_pairs, summarize, log_summary, probe_full_encoder

# datasets
from dataset_classes.spectrogram_images import SpectrogramImages
from dataset_classes.iq_dataset import IQDatasetH5, IQDatasetH5Sharded, pad_collate

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
    p.add_argument('--equal_exposure', action='store_true', default=False)
    p.add_argument('--train_mode', choices=['multi', 'vis', 'iq'], default='multi',
                   help="Train on both (multi), vision-only (vis), or IQ-only (iq)")
    p.add_argument('--use_pcgrad', action='store_true', default=False,
                        help='PCGrad on shared encoder (assumes 2-task round-robin)')
    p.add_argument('--clip_grad', type=float, default=None,
                   help='max global grad norm (None to disable)')

    # Model (keyword only)
    p.add_argument('--model', default='mae_vit_multi_micro', type=str,
                   help='Preset in models_multimodal_mae (e.g., mae_vit_multi_small/base/large)')

    # Common toggles passed into preset
    p.add_argument('--norm_pix_loss', action='store_true', default=False)
    p.add_argument('--norm_seg_loss', action='store_true', default=False)
    p.add_argument('--use_ant_mask', action='store_true', default=False)
    p.add_argument('--separate_decoders', action='store_true', default=False)

    # Mask ratios (per modality)
    p.add_argument('--mask_ratio_vis', default=0.75, type=float)
    p.add_argument('--mask_ratio_iq', default=0.5, type=float)

    # Optimizer / LR schedule
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--blr', type=float, default=1e-3)
    p.add_argument('--min_lr', type=float, default=0.)
    p.add_argument('--warmup_epochs', type=int, default=0)

    # Differential LR (optional)
    p.add_argument('--lr_diff', action='store_true', default=False,
                   help='Use different LR for encoder vs. decoders/adapters')
    p.add_argument('--lr_enc_scale', type=float, default=0.2,
                   help='Encoder LR = base LR * lr_enc_scale when --lr_diff')
    p.add_argument('--lr_dec_scale', type=float, default=1.0,
                   help='Decoder LR = base LR * lr_dec_scale when --lr_diff')

    # Data paths
    p.add_argument('--spect_paths',
                   default=['../datasets/spectrogram_dataset', '../datasets/spectrogram_iqengine_dataset'],
                   type=str, nargs='+',
                   help='List of spectrogram roots')
    p.add_argument('--iq_path', default='../datasets/train_iq', type=str)

    # WandB args
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='WavesFM')
    p.add_argument('--wandb_entity', type=str, default='waves-lab')  # optional
    p.add_argument('--wandb_group', type=str, default="pretrain")  # e.g., "vit-small-vs-sd"
    p.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    p.add_argument('--run_name', type=str, default=None)  # optional readable name
    p.add_argument('--full_probe_every_epochs', type=int, default=0)
    p.add_argument('--full_probe_batches', type=int, default=1)

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


def h5_worker_init_fn(_):
    ds = torch.utils.data.get_worker_info().dataset
    if hasattr(ds, "_h5") and ds._h5 is not None:
        try: ds._h5.close()
        except Exception: pass
        ds._h5 = None
        ds._cur_path = None


def fixed_subset(dataset, k, seed):
    k = min(k, len(dataset))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:k].tolist()
    return Subset(dataset, idx)


class EmptyDataLoader:
    def __len__(self): return 0

    def __iter__(self):
        if False:  # never yields
            yield None
        return


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
    gen_split = torch.Generator().manual_seed(args.seed)
    ds_tr, ds_val = random_split(ds, [0.8, 0.2], generator=gen_split)
    gen_tr = torch.Generator().manual_seed(args.seed + 1)
    sampler_tr = RandomSampler(ds_tr, generator=gen_tr)
    sampler_val = SequentialSampler(ds_val)

    common = dict(num_workers=args.num_workers, pin_memory=args.pin_mem, batch_size=args.batch_size_vis)
    if args.num_workers == 0:
        dl_tr = DataLoader(ds_tr, sampler=sampler_tr, drop_last=True, **common)
        dl_val = DataLoader(ds_val, sampler=sampler_val, drop_last=False, **common)
    else:
        dl_tr = DataLoader(ds_tr, sampler=sampler_tr, persistent_workers=True, prefetch_factor=4, drop_last=True,
                           **common)
        dl_val = DataLoader(ds_val, sampler=sampler_val, persistent_workers=True, prefetch_factor=4, drop_last=False,
                            **common)

    return (ds_tr, dl_tr), (ds_val, dl_val)


def build_iq_loaders(args, target_train_size=None):
    ds = IQDatasetH5Sharded(args.iq_path)

    if target_train_size is None:
        gen_a = torch.Generator().manual_seed(args.seed + 2)
        gen_b = torch.Generator().manual_seed(args.seed + 3)
        ds, _ = random_split(ds, [0.5, 0.5], generator=gen_a)
        ds_tr, ds_val = random_split(ds, [0.7, 0.3], generator=gen_b)
    else:
        # make a fixed subset then split 80/20 for proper validation
        sub = fixed_subset(ds, target_train_size, seed=args.seed + 4)
        gen_c = torch.Generator().manual_seed(args.seed + 5)
        ds_tr, ds_val = random_split(sub, [0.8, 0.2], generator=gen_c)

    gen_tr = torch.Generator().manual_seed(args.seed + 6)
    samp_tr = RandomSampler(ds_tr, generator=gen_tr)
    samp_val = SequentialSampler(ds_val)

    common = dict(num_workers=args.num_workers, pin_memory=args.pin_mem, worker_init_fn=h5_worker_init_fn,
                  batch_size=args.batch_size_iq)
    if args.num_workers == 0:
        dl_tr = DataLoader(ds_tr, sampler=samp_tr, drop_last=True, **common)
        dl_val = DataLoader(ds_val, sampler=samp_val, drop_last=False, **common)
    else:
        dl_tr = DataLoader(ds_tr, sampler=samp_tr, persistent_workers=True, prefetch_factor=4, drop_last=True,
                           **common)
        dl_val = DataLoader(ds_val, sampler=samp_val, persistent_workers=True, prefetch_factor=4, drop_last=False,
                            **common)
    return (ds_tr, dl_tr), (ds_val, dl_val)


def main(args):
    cudnn.deterministic = True
    cudnn.benchmark = False
    misc.init_distributed_mode(args)
    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- data ---
    if args.train_mode in ('multi', 'vis'):
        (ds_vis_tr, dl_vis_tr), (ds_vis_val, dl_vis_val) = build_spect_loader(args)
    else:
        ds_vis_tr, dl_vis_tr, ds_vis_val, dl_vis_val = None, EmptyDataLoader(), None, EmptyDataLoader()

    if args.train_mode in ('multi', 'iq'):
        (ds_iq_tr, dl_iq_tr), (ds_iq_val, dl_iq_val) = build_iq_loaders(
            args, target_train_size=(len(ds_vis_tr) + len(ds_vis_val) if args.equal_exposure else None)
        )
    else:
        ds_iq_tr, dl_iq_tr, ds_iq_val, dl_iq_val = None, EmptyDataLoader(), None, EmptyDataLoader()

    print(
        f"[Mode={args.train_mode}] "
        f"[IQ Train/Val: {0 if ds_iq_tr is None else len(ds_iq_tr)}/{0 if ds_iq_val is None else len(ds_iq_val)}] "
        f"[Vis Train/Val: {0 if ds_vis_tr is None else len(ds_vis_tr)}/{0 if ds_vis_val is None else len(ds_vis_val)}]"
    )

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # --- wandb init (main process only)
    wandb = None
    if args.use_wandb and misc.is_main_process():
        import wandb as _wandb
        wandb = _wandb
        cfg = vars(args).copy()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            mode=args.wandb_mode,
            job_type="pretrain",
            name=args.run_name,
            config=cfg,
        )
        # set step metrics
        wandb.define_metric("epoch")
        wandb.define_metric("pretrain/*", step_metric="epoch")
        wandb.define_metric("pretrain/train/*", step_metric="epoch")
        wandb.define_metric("pretrain/val_iq/*", step_metric="epoch")
        wandb.define_metric("pretrain/val_vis/*", step_metric="epoch")
        wandb.define_metric("pretrain/probe_full/*", step_metric="epoch")

    if wandb is not None:
        print(f"W&B → project={args.wandb_project} group={args.wandb_group} name={args.run_name}")

    # --- model from preset keyword ---
    # Pass only the common toggles; preset handles architecture specifics.
    model = models_multimodal_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        norm_seg_loss=args.norm_seg_loss,
        iq_use_ant_mask=args.use_ant_mask,
        separate_decoders=args.separate_decoders
    ).to(device)
    model_without_ddp = model
    print("Model =", model_without_ddp)
    encoder_params = sum(p.numel() for p in model.blocks.parameters())
    decoder_params = sum(p.numel() for p in model.decoder_blocks.parameters())
    print(f"[Encoder: {encoder_params / 1.e6:.2f}] "
          f"[Decoder: {decoder_params / 1.e6:2f}] "
          f"[Ratio: {encoder_params / decoder_params:.2f}]")

    # --- optimizer & lr ---
    if args.train_mode == 'multi':
        eff_batch_size = (args.batch_size_vis + args.batch_size_iq) * args.accum_iter
    elif args.train_mode == 'vis':
        eff_batch_size = args.batch_size_vis * args.accum_iter
    else:  # 'iq'
        eff_batch_size = args.batch_size_iq * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if not args.lr_diff:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        # ---- Differential LR param groups ----
        enc_named, dec_named, rest_named = [], [], []
        for n, p in model_without_ddp.named_parameters():
            if not p.requires_grad:
                continue
            lname = n.lower()
            is_enc = lname.startswith("encoder") or lname.startswith("blocks") or lname.startswith("norm")
            is_dec = lname.startswith("decoder")
            if is_dec:
                dec_named.append((n, p))
            elif is_enc:
                enc_named.append((n, p))
            else:
                rest_named.append((n, p))  # any leftovers

        lr_enc = args.lr * args.lr_enc_scale
        lr_dec = args.lr * args.lr_dec_scale

        groups = []
        groups += build_wd_groups(enc_named, lr_enc, args.weight_decay)
        groups += build_wd_groups(dec_named, lr_dec, args.weight_decay)
        # put leftovers with decoder LR by default
        groups += build_wd_groups(rest_named, lr_dec, args.weight_decay)

        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))

        # print counts & LRs
        def _cnt(named):
            return sum(p.numel() for _, p in named)

        print("[LR-diff] "
              f"enc lr={lr_enc:.2e} (n={_cnt(enc_named)}) | "
              f"dec lr={lr_dec:.2e} (n={_cnt(dec_named)}) | "
              f"rest n={_cnt(rest_named)}")

    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    t0 = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch_multi(
            model=model,
            loader_vis=dl_vis_tr,
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

        # --- eval ---
        val_stats = {}
        if len(dl_iq_val) > 0:
            val_stats.update({f'val_iq_{k}': v for k, v in
                              evaluate_iq(dl_iq_val, model, device, mask_ratio=args.mask_ratio_iq).items()})
        if len(dl_vis_val) > 0:
            val_stats.update({f'val_vis_{k}': v for k, v in
                              evaluate_vision(dl_vis_val, model, device, mask_ratio=args.mask_ratio_vis).items()})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **val_stats, 'epoch': epoch}

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb is not None:
            payload = {'epoch': epoch}
            payload.update({f"pretrain/train/{k}": v for k, v in train_stats.items()})
            payload.update({f"pretrain/val_iq/{k.replace('val_iq_', '')}": v
                            for k, v in val_stats.items() if k.startswith('val_iq_')})
            payload.update({f"pretrain/val_vis/{k.replace('val_vis_', '')}": v
                            for k, v in val_stats.items() if k.startswith('val_vis_')})
            wandb.log(payload)

        if args.full_probe_every_epochs and ((epoch + 1) % args.full_probe_every_epochs == 0):
            num_batches = max(1, args.full_probe_batches)
            pairs = sample_pairs(dl_vis_tr, dl_iq_tr, K=num_batches)
            rows = []
            for b_vis, b_iq in pairs:
                r = probe_full_encoder(model, b_vis, b_iq, device, args.mask_ratio_vis, args.mask_ratio_iq)
                if r:
                    rows.append(r)
            summary = summarize(rows)
            log_summary(summary, epoch, log_writer=log_writer, wandb=wandb, prefix="pretrain/probe_full")

    total = time.time() - t0
    print('Training time', str(datetime.timedelta(seconds=int(total))))
    if wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
