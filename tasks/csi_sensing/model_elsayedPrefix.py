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


class TaskModel(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool, tanh=False, head_layers=1,**kwargs):
        super(TaskModel, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.tanh = tanh
        # Task Head
        num_classes = kwargs['num_classes']
        layers = []
        for i in range(head_layers - 1):
            layers.extend([nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()])
        layers.append(nn.Linear(self.embed_dim, num_classes))
        self.head = nn.Sequential(*layers) if head_layers > 1 else nn.Linear(self.embed_dim, num_classes)

        prefix_dim = 30
        self.prefix_embedding = nn.Parameter(torch.randn(1, 1, prefix_dim))
        self.prefix_proj = nn.Sequential(
            nn.Linear(self.embed_dim + prefix_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        # self.freeze_encoder()

    def freeze_encoder(self, num_blocks=None):
        if num_blocks is None:
            for param in self.blocks.parameters():
                param.requires_grad = False
        else:
            for param in self.blocks[:num_blocks].parameters():
                param.requires_grad = False

        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False


    def forward_features_prefix(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # Convert image to patch embeddings
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        
        # Expand and concatenate prefix embeddings
        prefix_tokens = self.prefix_embedding.expand(B, x.shape[1], -1)  # (B, num_prefix_tokens, embed_dim)
        x = torch.cat((prefix_tokens, x), dim=-1)  # Prepend prefix tokens to the input
        x = self.prefix_proj(x)

        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features_prefix(x)
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
        if 'patch_embed.proj.weight' in checkpoint_model.keys():
            checkpoint_model['patch_embed.proj.weight'] = checkpoint_model['patch_embed.proj.weight'].expand(-1, 3, -1, -1)
        else:
            checkpoint_model['patch_embed.proj.weight'] = checkpoint_model['patch_embed.1.proj.weight']
        msg = self.load_state_dict(checkpoint_model, strict=False)
        trunc_normal_(self.head.weight, std=2e-5)
        return msg
    
def vit_small_patch16(**kwargs):
    model = TaskModel(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patch16(**kwargs):
    model = TaskModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
