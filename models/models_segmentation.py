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
import torch.nn as nn
from .tasks.segmentation import SegmentationViT


def seg_vit_small_patch16_dec512d8b(**kwargs):
    model = SegmentationViT(
        patch_size=16, embed_dim=512, in_chans=1, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def seg_vit_medium_patch16_dec512d8b(**kwargs):
    model = SegmentationViT(
        patch_size=16, embed_dim=768, in_chans=1, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def seg_vit_large_patch16_dec512d8b(**kwargs):
    model = SegmentationViT(
        patch_size=16, embed_dim=1024, in_chans=1, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def new_custom_arch(**kwargs):
    # TODO: In case you need to design a new architecture of the same SegmentationViT model (changing number of layers, embedding dimension, etc.),
    # please write this function like the 3 previous examples

    # Note: You can to also set a new for it in the recommended archs below
    pass

# Some recommended archs
seg_vit_small_patch16 = seg_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
seg_vit_medium_patch16 = seg_vit_medium_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
seg_vit_large_patch16 = seg_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

# TODO: set a name for your custom architecture here like the 3 lines above


