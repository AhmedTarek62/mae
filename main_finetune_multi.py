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

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

import models_vit_multi
from advanced_finetuning.lora import create_lora_model  # keep LoRA only

from engine_finetune_multi import train_one_epoch, evaluate
from dataset_classes import IQDatasetH5Sharded, RML, RFPrintDataset, Positioning5G, RadioSignal, CSISensingDataset
from dataset_classes.rml import make_snr_sampler


def get_args_parser():
    parser = argparse.ArgumentParser('Multimodal ViT fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str,
                        choices=['aoa', 'amc', 'rml', 'rfp', 'pos', 'sensing', 'rfs'])
    # Model parameters
    parser.add_argument('--model', default='vit_multi_micro', type=str, metavar='MODEL')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')  # kept arg for compatibility
    parser.add_argument('--frozen_blocks', type=int)
    parser.add_argument('--tanh', action='store_true', default=False)
    parser.add_argument('--use_conditional_ln', action='store_true', default=False)
    parser.add_argument('--strict_probe', action='store_true', default=False,
                        help='freeze tokenizer & condLN')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')

    # LoRA
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=1)

    # Augmentation / Mixup
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    # * Finetuning params
    parser.add_argument('--finetune', default='')
    parser.add_argument('--global_pool', default='token')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--nb_classes', default=0, type=int)

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default='')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # WandB parameters
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='WavesFM')
    parser.add_argument('--wandb_entity', type=str, default='waves-lab')  # optional
    parser.add_argument('--wandb_group', type=str, default=None)  # e.g., "vit-small-vs-sd"
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--run_name', type=str, default=None)

    # distributed (kept)
    parser.add_argument('--world_size', default=1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--local_rank', default=-1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--dist_on_itp', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dist_url', default='env://', help=argparse.SUPPRESS)

    return parser


def _build_datasets(args):
    assert args.mode in ('aoa', 'amc', 'rml', 'rfp', 'pos', 'sensing', 'rfs')
    if args.mode in ('amc', 'aoa'):
        _, dataset = random_split(IQDatasetH5Sharded(args.data_path, mode=args.mode), [0.5, 0.5])
        dataset_train, dataset_val = random_split(dataset, [0.7, 0.3])
    elif args.mode == 'rml':
        dataset = RML(args.data_path, version="2022")
        dataset_train, dataset_val = random_split(dataset, [0.7, 0.3])
    elif args.mode == 'rfp':
        dataset = RFPrintDataset(args.data_path)
        dataset_train, dataset_val = random_split(dataset, [0.7, 0.3])
    elif args.mode == 'pos':
        dataset_train = Positioning5G(Path(os.path.join(args.data_path, f'outdoor/train')))
        dataset_val = Positioning5G(Path(os.path.join(args.data_path, f'outdoor/test')))
    elif args.mode == 'sensing':
        dataset_train = CSISensingDataset(os.path.join(args.data_path, 'train'))
        dataset_val = CSISensingDataset(os.path.join(args.data_path, 'test'))
    elif args.mode == 'rfs':
        dataset_train = RadioSignal(os.path.join(args.data_path, 'train'))
        dataset_val = RadioSignal(os.path.join(args.data_path, 'test'))
    else:
        raise ValueError(f"Unrecognized mode: {args.mode}")
    return dataset_train, dataset_val


def _build_samplers(args, dataset_train, dataset_val):
    if args.mode == 'rml':
        sampler_train = make_snr_sampler(dataset_train, policy='gaussian')
        sampler_val = SequentialSampler(dataset_val)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)
    return sampler_train, sampler_val


def _infer_task_and_modality(args):
    # nb_classes + modality + in_chans for vision
    in_chans = None
    if args.mode == 'amc':
        return 7, 'iq', in_chans
    if args.mode == 'aoa':
        return 3, 'iq', in_chans
    if args.mode == 'rml':
        return 11, 'iq', in_chans
    if args.mode == 'rfp':
        return 4, 'iq', in_chans
    if args.mode == 'pos':
        return 3, 'vision', 4
    if args.mode == 'sensing':
        return 6, 'vision', 3
    if args.mode == 'rfs':
        return 20, 'vision', 1
    raise ValueError(f"Unrecognized mode: {args.mode}")


def main(args):
    misc.init_distributed_mode(args)

    print('job dir:', os.path.dirname(os.path.realpath(__file__)))
    print(str(args).replace(', ', ',\n'))

    # seeds
    print(f"seed is {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    cudnn.benchmark = True

    # --- wandb init (main process only)
    wandb = None
    is_main = misc.is_main_process()
    if args.use_wandb and is_main:
        import wandb as _wandb
        wandb = _wandb
        cfg = vars(args).copy()
        protocol = "lora" if getattr(args, "lora", False) else "linear-probe"
        run_name = args.run_name or f"{args.mode}/{protocol}"
        group = args.wandb_group if args.wandb_group else f"finetune-{args.mode}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group,
            mode=args.wandb_mode,
            name=run_name,
            config=cfg,
        )
        # set step metrics
        wandb.define_metric("epoch")
        wandb.define_metric("finetune/*", step_metric="epoch")
        wandb.define_metric("finetune/train/*", step_metric="epoch")
        wandb.define_metric("finetune/val/*", step_metric="epoch")

    # datasets & loaders
    dataset_train, dataset_val = _build_datasets(args)
    sampler_train, sampler_val = _build_samplers(args, dataset_train, dataset_val)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # model
    nb_classes, modality, in_chans = _infer_task_and_modality(args)

    model = models_vit_multi.__dict__[args.model](modality=modality,
                                                  global_pool=args.global_pool,
                                                  num_classes=nb_classes,
                                                  tanh=args.tanh,
                                                  vis_in_chans_actual=in_chans,
                                                  use_conditional_ln=args.use_conditional_ln)
    # optional LoRA
    if args.lora:
        model = create_lora_model(model, args.lora_rank, args.lora_alpha)

    # load weights
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("Load checkpoint:", args.resume)
        msg = model.load_state_dict(checkpoint['model'], strict=True)
        print(msg)
    elif args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Finetune from:", args.finetune)
        ckpt = checkpoint['model']
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)
        if hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
            trunc_normal_(model.head.weight, std=2e-5)

    # freezing
    if args.lora and hasattr(model, 'freeze_encoder_lora'):
        model.freeze_encoder_lora()
    elif args.frozen_blocks is not None:
        model.freeze_encoder(args.frozen_blocks)
    else:
        model.freeze_encoder()

    if not args.strict_probe:
        model.unfreeze_tokenizer()
        model.unfreeze_conditional_ln()

    model = model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model =", model_without_ddp)
    print('model params (M): %.6f' % (sum(p.numel() for p in model.parameters()) / 1.e6))
    print('fine-tuned params (M): %.6f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations:", args.accum_iter)
    print("effective batch size:", eff_batch_size)

    # optimizer with layer decay
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # criterion
    classification_modes = {'amc', 'rml', 'rfp', 'sensing', 'rfs'}
    if args.mode == 'rfs':
        criterion = torch.nn.CrossEntropyLoss(weight=dataset_train.class_weights.to(device))
    elif args.smoothing > 0. and args.mode in classification_modes:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss() if args.mode in classification_modes else torch.nn.MSELoss()
    print("criterion =", criterion)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    track_pca = args.mode in classification_modes
    best_metric = float('-inf') if track_pca else float('inf')
    best_epoch = -1
    best_key = 'pca' if track_pca else 'mae'
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer, args=args
        )

        if args.output_dir and (epoch % 10 == 0 or (epoch + 1) == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        test_stats = evaluate(data_loader_val, model, criterion, device)

        # --- best tracking ---
        cur = test_stats.get(best_key, None)
        if cur is not None:
            improved = (cur > best_metric) if track_pca else (cur < best_metric)
            if improved:
                best_metric = float(cur)
                best_epoch = epoch
                print(f"[BEST] epoch={best_epoch} {best_key}={best_metric:.4f}")
                if wandb is not None and is_main:
                    wandb.run.summary['best_epoch'] = int(best_epoch)
                    wandb.run.summary['best_metric'] = float(best_metric)
                    wandb.run.summary['best_metric_key'] = best_key

        # classification vs regression logging
        if 'pca' in test_stats:  # classification
            print(f"Mean per-class acc on {len(dataset_val)} val samples: {test_stats['pca']:.3f}%")
            if log_writer is not None:
                log_writer.add_scalar('perf/test_pca', test_stats.get('pca', 0.0), epoch)
                log_writer.add_scalar('perf/test_acc1', test_stats.get('acc1', 0.0), epoch)
                log_writer.add_scalar('perf/test_acc3', test_stats.get('acc3', 0.0), epoch)
                log_writer.add_scalar('perf/test_loss', test_stats.get('loss', 0.0), epoch)
        else:  # regression (AoA)
            print(f"Val MAE: {test_stats.get('mae', 0.0):.4f}  RMSE: {test_stats.get('rmse', 0.0):.4f}")
            if log_writer is not None:
                log_writer.add_scalar('perf/val_mae', test_stats.get('mae', 0.0), epoch)
                log_writer.add_scalar('perf/val_rmse', test_stats.get('rmse', 0.0), epoch)
                log_writer.add_scalar('perf/val_loss', test_stats.get('loss', 0.0), epoch)

        # flat log
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if wandb is not None and is_main:
            payload = {"epoch": epoch,
                       "finetune/train/loss": train_stats.get("loss", 0.0),
                       "finetune/lr": train_stats.get("lr")}
            for k in ("acc1", "acc3", "pca", "loss", "mae", "rmse"):
                if k in test_stats:
                    payload[f"finetune/val/{k}"] = test_stats[k]
            wandb.log(payload)

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time', total_time_str)
    if wandb is not None and is_main:
        wandb.run.summary['best_epoch'] = int(best_epoch)
        wandb.run.summary['best_metric'] = float(best_metric)
        wandb.run.summary['best_metric_key'] = 'pca' if track_pca else 'mae'
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
