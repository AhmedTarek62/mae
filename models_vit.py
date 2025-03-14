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
from timm.models._manipulate import checkpoint_seq
from advanced_finetuning.prefix import PrefixEncoder


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool, tanh=False, head_layers=1, prefix_tuning=False, num_prefix_tokens=10, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.tanh = tanh
        num_classes = kwargs['num_classes']
        layers = []
        for i in range(head_layers - 1):
            layers.extend([nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()])
        layers.append(nn.Linear(self.embed_dim, num_classes))
        self.head = nn.Sequential(*layers) if head_layers > 1 else nn.Linear(self.embed_dim, num_classes)
        if prefix_tuning:
            self.prefix_encoder = PrefixEncoder(num_prefix_tokens, self.embed_dim, len(self.blocks))
        else:
            self.prefix_encoder = None

    def unfreeze_patch_embed(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = True

    def freeze_encoder(self, num_blocks=None):
        if num_blocks is None:
            for param in self.blocks.parameters():
                param.requires_grad = False
        else:
            for param in self.blocks[:num_blocks].parameters():
                param.requires_grad = False

        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False

    def freeze_encoder_lora(self):
        # Freeze all params
        for param in self.blocks.parameters():
            param.requires_grad = False

        # Unfreeze LoRA layers
        for block in self.blocks:
            for param in block.attn.qkv.lora_q.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_v.parameters():
                param.requires_grad = True

        # Unfreeze classifier layer
        for param in self.head.parameters():
            param.requires_grad = True

    def freeze_encoder_prefix(self):
        # Freeze all params
        for param in self.blocks.parameters():
            param.requires_grad = False

        for param in self.prefix_encoder.parameters():
            param.requires_grad = True

        # Unfreeze classifier layer
        for param in self.head.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        elif self.prefix_encoder is not None:
            B = x.shape[0]
            prefix_tokens = self.prefix_encoder(B)
            for i, block in enumerate(self.blocks):
                prefix_k, prefix_v = torch.split(
                    prefix_tokens[:, :, i * self.embed_dim:(i + 2) * self.embed_dim], self.embed_dim, dim=-1)
                x = block(x, prefix_k, prefix_v)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        if self.tanh:
            return torch.tanh(x)
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_medium_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
