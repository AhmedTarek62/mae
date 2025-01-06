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

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool, tanh=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.tanh = tanh

    def freeze_encoder(self, num_blocks=None):
        if num_blocks is None:
            for param in self.blocks.parameters():
                param.requires_grad = False
        else:
            for param in self.blocks[:num_blocks].parameters():
                param.requires_grad = False

        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        if self.tanh:
            return torch.tanh(x)
        return x
    
    def load_model_checkpoint(self, checkpoint_path:str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        
        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias', 'pos_embed']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        checkpoint_model['patch_embed.proj.weight'] = checkpoint_model['patch_embed.proj.weight'].expand(-1, 3, -1, -1)
        msg = self.load_state_dict(checkpoint_model, strict=False)
        trunc_normal_(self.head.weight, std=2e-5)
        return msg


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

seg_vit_small_patch16 = vit_small_patch16  # decoder: 512 dim, 8 blocks

def vit_medium_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

seg_vit_medium_patch16 = vit_medium_patch16  # decoder: 512 dim, 8 blocks

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

seg_vit_large_patch16 = vit_large_patch16  # decoder: 512 dim, 8 blocks

# TODO: In case you need to design a new architecture of the same SegmentationViT model (changing number of layers, embedding dimension, etc.),
# please write this function like the 3 previous examples
def new_custom_arch(**kwargs):
    # Note: You can to also set a new for it in the recommended archs below
    pass

# TODO: Set an alias name to your architecture

