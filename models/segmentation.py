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

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class SegmentationViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone.
    This class is a custom implementation of a Vision Transformer (ViT) tailored for segmentation tasks. 
    It incorporates components from Masked Autoencoders (MAE) to handle image patches and reconstruction tasks.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 out_chans=3, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        ## 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        ## 2. Class tokenization (for the entire image)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        ## 3. Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        ## 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])
        
        # 5. Normalization
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        ## 1. Decoder Embedding (from encoder embedding dimenssion --> decoder embeddnig dimension)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        ## 2. Decoder Position Embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        ## 3. Transformer Blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True) for _ in range(decoder_depth)])

        ## 4. Normalization
        self.decoder_norm = norm_layer(decoder_embed_dim)

        ## 5. Prediction Head (decoder to patch)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        c = x.shape[-1] // p ** 2
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x[:, 1:]

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)
        return self.unpatchify(pred)

    def freeze_encoder(self):
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.patch_embed.proj.parameters():
            param.requires_grad = True

    def load_model_checkpoint(self, checkpoint_path:str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']

        del checkpoint_model['decoder_pred.weight']
        del checkpoint_model['decoder_pred.bias']
        del checkpoint_model['mask_token']
        # Set the model state
        msg = self.load_state_dict(checkpoint_model, strict=False)
        return msg
    

def vit_small_patch16_dec512d8b(**kwargs):
    model = SegmentationViT(
        patch_size=16, embed_dim=512, in_chans=1, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patch16(**kwargs):
    model = SegmentationViT(
        patch_size=16, embed_dim=768, in_chans=1, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = SegmentationViT(
        patch_size=16, embed_dim=1024, in_chans=1, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# TODO: In case you need to design a new architecture of the same SegmentationViT model (changing number of layers, embedding dimension, etc.),
# please write this function like the 3 previous examples
def new_custom_arch(**kwargs):
    pass