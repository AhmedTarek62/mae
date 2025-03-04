# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_best_chechpoint(args, output_dir, epoch, model):    
    checkpoint_path = Path(output_dir) / ('best-checkpoint.pth')
    to_save = {
        'model': model.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    torch.save(to_save, checkpoint_path)

def save_model(args, output_dir, epoch, model, optimizer, loss_scaler):    
    checkpoint_path = Path(output_dir) / ('checkpoint-%s.pth' % str(epoch))
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    torch.save(to_save, checkpoint_path)

# def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
#     output_dir = Path(args.output_dir)
#     epoch_name = str(epoch)
#     if loss_scaler is not None:
#         checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
#         for checkpoint_path in checkpoint_paths:
#             to_save = {
#                 'model': model_without_ddp.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'scaler': loss_scaler.state_dict(),
#                 'args': args,
#             }
#             torch.save(to_save, checkpoint_path)
#     else:
#         client_state = {'epoch': epoch}
#         model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

import math

def calculate_score(validation_length, model_n_parameters, final_val_loss, final_val_accuracy):
    assert final_val_loss or final_val_accuracy, "Neither reported loss nor accuracy"

    """
    The score is calculated as follows:
    score = w1 * log (1 + validation_length) - w2 * loss + w3 * accuracy - w4 * log(1 + model_n_parameters)
    """
    assert final_val_loss or final_val_accuracy, print("Neither reported loss nor accuracy")

    # Every +100 data sample gets 10 points (maximum 100)
    validation_score = min(validation_length // 100 * 10, 100)

    # Loss/Accuracy score
    if final_val_loss:
        # Excepted values are close to 0.1 --> (best is 0)
        perf_score = 100 - final_val_loss*100
    else:
        # Expected values are close to 0.9 --> (best is 1)
        perf_score = final_val_accuracy*100

    # Parameters Score
    # Every +200K parameters deduct 5 points (maximum 100)
    paras_score = 100 - min(max(model_n_parameters // 200000 * 5, 0), 100)

    # Define weights
    w1, w2, w3 = 0.25, 0.5, 0.25 # validation, performance, parameters
    
    # Compute the score
    score = (
        w1 * validation_score
        + w2 * perf_score
        + w3 * paras_score
    )

    return score

def report_score(config, model, dataset_val, final_val_loss, final_val_accuracy):
    # Extract parameters
    validation_length = len(dataset_val)
    model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    score_value = calculate_score(validation_length, model_n_parameters, final_val_loss, final_val_accuracy)
    print(f"Score value = {score_value}")
    report ={"task": config.task, 
             "loss": final_val_loss,
             "accuracy": final_val_accuracy,
             "validation_length": validation_length,
             "model_n_parameters": model_n_parameters,
             "score": score_value}

    return report 