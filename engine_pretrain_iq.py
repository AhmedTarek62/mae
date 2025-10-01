# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x_pad, time_mask, lengths) in (
            enumerate(metric_logger.log_every(data_loader, print_freq, header))):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x_pad = x_pad.to(device, non_blocking=True)            # (N, 2, C, T_max)
        time_mask = time_mask.to(device, non_blocking=True)    # (N, T_max), bool

        with torch.cuda.amp.autocast():
            loss, _, _, _ = model(x_pad, time_mask, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader: Iterable, model: torch.nn.Module,
                      device: torch.device, mask_ratio: float = 0.75):
    """
    Validation loop for MAE pre-training on IQ.
    Expects data_loader to yield (x_pad, time_mask, lengths):
      x_pad:     (N, 2, C, T_max)
      time_mask: (N, T_max)  bool
      lengths:   list[int]   (unused here)
    Returns dict of averaged metrics.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Val:"

    model.eval()
    total_masked_real = 0
    total_real = 0

    for x_pad, time_mask, lengths in metric_logger.log_every(data_loader, 10, header):
        x_pad = x_pad.to(device, non_blocking=True)
        time_mask = time_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mae_mask, token_mask = model(x_pad, time_mask, mask_ratio=mask_ratio)

        loss_val = float(loss.item())
        metric_logger.update(loss=loss_val)

        # bookkeeping: how many tokens were actually masked among real (non-pad) tokens
        masked_real = ((mae_mask > 0) & token_mask).sum().item()
        real_tokens = token_mask.sum().item()
        total_masked_real += masked_real
        total_real += real_tokens

    # sync across processes (if DDP)
    metric_logger.synchronize_between_processes()

    masked_real_pct = (100.0 * total_masked_real / total_real) if total_real > 0 else 0.0
    print(f"* Val loss {metric_logger.loss.global_avg:.4f}  "
          f"masked_real% {masked_real_pct:.1f}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["masked_real_pct"] = masked_real_pct
    return stats
