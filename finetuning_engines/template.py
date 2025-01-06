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
from timm.utils import accuracy

import util.lr_sched as lr_sched
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
    
    avg_loss, avg_lr = None, None 
    with tqdm(data_loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
        for data_iter_step, (samples, targets) in enumerate(pbar):
            # TODO: Compelte your own training loop
            raise NotImplementedError
        
        return {
            "avg_loss": avg_loss,
            "avg_lr": avg_lr,
            "losses": losses,
            "learning_rates": learning_rates
        }


@torch.no_grad()
def evaluate(data_loader, model, criterion, device):

    losses = []          # To store loss values
    accuracies = []      # To store accuracies values

    # switch to evaluation mode
    model.eval()

    avg_loss, avg_acc = None, None

    with tqdm(data_loader, desc=f'Validation..', unit='batch') as pbar:
        for data_iter_step, (samples, targets) in enumerate(pbar):
            # TODO: Compelte your own evaluation loop
            raise NotImplementedError

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc
        }