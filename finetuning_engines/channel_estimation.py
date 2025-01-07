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
from typing import Iterable, Optional

import torch

from timm.data import Mixup
import util.lr_sched as lr_sched
from snr_weighted_loss import WeightedLoss
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, args=None):
    """
    Train a model for one epoch.

    This function trains a given model for one epoch using a specified data loader, criterion, optimizer, 
    and learning rate scheduler. It supports mixed precision training and optional mixup for data augmentation.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): Loss function used to compute the training loss.
        data_loader (Iterable): Data loader providing batches of input samples and targets.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device on which the model and data are located.
        epoch (int): The current epoch number (used for logging).
        loss_scaler (Callable): Utility to scale and backpropagate the loss, supporting mixed precision.
        max_norm (float, optional): Maximum gradient norm for gradient clipping. Defaults to 0 (no clipping).
        mixup_fn (Optional[Mixup], optional): Function for applying mixup augmentation to samples and targets. Defaults to None.
        args (Namespace): Additional arguments, including hyperparameters like total epochs and configuration
                          for learning rate scheduling. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - "average_loss" (float): Average loss over the epoch.
            - "average_lr" (float): Average learning rate over the epoch.
            - "losses" (list): List of individual loss values for each batch.
            - "learning_rates" (list): List of learning rates used for each batch.

    Notes:
        - The function uses `torch.amp.autocast` for mixed precision training.
        - The learning rate is adjusted dynamically at each iteration using the provided scheduler.
        - Loss values and learning rates are logged and returned for analysis.
    """
    assert args, print("Namespace (args) cannot be skipped.")

    model.train(True)
    optimizer.zero_grad()
    losses = []          # To store loss values
    learning_rates = []  # To store learning rates
    

    with tqdm(data_loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
        for data_iter_step, (samples, targets, snr) in enumerate(pbar):
            # we use a per iteration (instead of per epoch) lr scheduler
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
            
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if isinstance(criterion, WeightedLoss):
                snr = snr.to(device, non_blocking=True)
            

            if mixup_fn:
                samples, targets = mixup_fn(samples, targets)
            
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                targets = targets.view(-1)
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
                if isinstance(criterion, WeightedLoss):
                    loss = criterion(outputs, targets, snr)
                else:
                    loss = criterion(outputs, targets)
            
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False, update_grad=True)
            pbar.set_postfix({'loss': loss_value})

            optimizer.zero_grad()

            # Store loss value
            losses.append(loss_value)
            # Capture current learning rates
            for group in optimizer.param_groups:
                learning_rates.append(group["lr"])

        # Compute the average loss and learning rates
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_lr = sum(learning_rates) / len(learning_rates) if learning_rates else 0

        return {
            "avg_loss": avg_loss,
            "avg_lr": avg_lr,
            "losses": losses,
            "learning_rates": learning_rates
        }


@torch.no_grad()
def evaluate(data_loader, model, criterion, device):

    losses = []          # To store loss values

    # switch to evaluation mode
    model.eval()
    with tqdm(data_loader, desc=f'Validation..', unit='batch') as pbar:
        for data_iter_step, (samples, targets, snr) in enumerate(pbar):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if isinstance(criterion, WeightedLoss):
                snr = snr.to(device, non_blocking=True)

            # compute output
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                targets = targets.view(-1)
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
                if isinstance(criterion, WeightedLoss):
                    loss = criterion(outputs, targets, snr)
                else:
                    loss = criterion(outputs, targets)

            batch_size = samples.shape[0]
            
            losses.append(loss.item())
    
    # Compute the average loss and learning rates
    avg_loss = sum(losses) / len(losses) if losses else 0

    return {
        "avg_loss": avg_loss,
        "avg_acc": None
        }