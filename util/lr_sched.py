# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, n_epochs, lr, lr_args:dict):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < lr_args['warmup_epochs']:
        lr = lr * epoch / lr_args['warmup_epochs'] 
    else:
        lr = lr_args['min_lr'] + (lr_args['lr'] - lr_args['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - lr_args['warmup_epochs']) / (n_epochs - lr_args['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
