# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(data_loader, model, criterion, device):

    losses = []          # To store loss values
    accuracies = []      # To store accuracies values

    # switch to evaluation mode
    model.eval()
    with tqdm(data_loader, desc=f'Validation..', unit='batch') as pbar:
        for data_iter_step, (samples, targets) in enumerate(pbar):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.amp.autocast('cuda'):
                outputs = model(samples)
                targets = targets.view(-1)
                outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
                loss = criterion(outputs, targets)

            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = torch.sum(torch.argmax(outputs, dim=-1) == targets) / targets.shape[0] * 100
            batch_size = samples.shape[0]
            
            losses.append(loss.item())
            accuracies.append(acc1.item())
    
    # Compute the average loss and learning rates
    avg_loss = sum(losses) / len(losses) if losses else 0
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc
        }