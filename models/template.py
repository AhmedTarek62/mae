# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.layers import trunc_normal_

class CLASS_TEMPLATE:
    """ Vision Transformer with support for global average pooling
    """
    # TODO: Complete your own model class for your specific task!
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def load_model_checkpoint(self, checkpoint_path:str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        # TODO: Complete this function to load the pre-trained model checkpoint weights into your model

        raise NotImplementedError


def new_custom_arch(**kwargs):
    # TODO: Initiate a version (architecture) of your model here for easy establishment
    raise NotImplementedError

