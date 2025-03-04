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
import numpy as np
import torch

from timm.data import Mixup
import util.lr_sched as lr_sched
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, n_epochs:int, loss_scaler, 
                    lr: float, lr_args:dict, max_norm: float = 0):
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
    model.train(True)
    optimizer.zero_grad()
    losses = []          # To store loss values
    learning_rates = []  # To store learning rates
    

    with tqdm(data_loader, desc=f'Epoch {epoch}/{n_epochs}', unit='batch') as pbar:
        for data_iter_step, (samples, targets) in enumerate(pbar):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            pbar.set_postfix({'loss': loss_value})
            # Store loss value
            losses.append(loss_value)
            # Capture current learning rates
            for group in optimizer.param_groups:
                learning_rates.append(group["lr"])

            loss.backward()
            optimizer.step()

        # Compute the average loss and learning rates
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_lr = sum(learning_rates) / len(learning_rates) if learning_rates else 0

        return {
            "avg_loss": avg_loss,
            "avg_lr": avg_lr,
            "losses": losses,
            "learning_rates": learning_rates
        }

def reverse_normalize(x, coord_min, coord_max):
    return (x + 1) / 2 * (coord_max - coord_min) + coord_min

@torch.no_grad()
def evaluate(data_loader, model, criterion, device, coords=None):

    losses = []          # To store loss values
    distances = []
    # switch to evaluation mode
    model.eval()
    with tqdm(data_loader, desc=f'Validation..', unit='batch') as pbar:
        for data_iter_step, (samples, targets) in enumerate(pbar):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                loss = criterion(outputs, targets)
                if coords:
                    pred_position = reverse_normalize(outputs.cpu(),
                                                      coords['min'], coords['max'])
                    position = reverse_normalize(targets.cpu(),
                                                 coords['min'], coords['max'])
                    distances.extend(torch.sqrt(torch.sum((pred_position - position) ** 2, dim=1)))

            batch_size = samples.shape[0]
            
            losses.append(loss.item())
    
    # Compute the average loss and learning rates
    avg_loss = sum(losses) / len(losses) if losses else 0
    distances = np.array(distances)
    return {
        "avg_loss": avg_loss,
        "avg_acc": None,
        "distances": distances
        } 