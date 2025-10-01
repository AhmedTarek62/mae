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
import torch.nn.functional as F

from timm.models.vision_transformer import Block

from util.pos_embed import get_1d_sincos_pos_embed


class MaskedAutoencoderIQViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 segment_len=4096,
                 hop=0,
                 max_tokens=256,
                 max_antennas=64,
                 embed_dim=512,
                 depth=4, num_heads=8,
                 decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_seg_loss=False, use_ant_mask=False):
        super().__init__()

        # MAE preprocessing specifics
        # --------------------------------------------------------------------------
        self.segment_len = segment_len
        self.hop = hop if hop != 0 else segment_len
        self.max_tokens = max_tokens
        self.max_antennas = max_antennas
        # antenna masking toggle + how many antennas to mask ---
        self.use_ant_mask = use_ant_mask  # set True to enable antenna-stream masking
        self.ant_mask_k: int = 1
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.segment_embed = nn.Conv1d(in_channels=2, out_channels=embed_dim, kernel_size=segment_len, stride=1)
        self.time_pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, embed_dim), requires_grad=False)
        self.ant_embed = nn.Embedding(max_antennas, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_time_pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, decoder_embed_dim),
                                                   requires_grad=False)
        self.decoder_ant_embed = nn.Embedding(max_antennas, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 2 * segment_len, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_seg_loss = norm_seg_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.embed_dim, self.max_tokens, cls_token=True)
        self.time_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pos_embed = get_1d_sincos_pos_embed(self.decoder_embed_dim, self.max_tokens, cls_token=True)
        self.decoder_time_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.ant_embed.weight, std=.02)
        torch.nn.init.normal_(self.decoder_ant_embed.weight, std=.02)

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

    def segment(self, x, pad_tail=True):
        """`
        x: (n, 2, c, l)
        returns:
          x_seg:   (n, s, 2, c, m)
          mask_in: (n, s, m)  True=real, False=pad
        """
        assert x.dim() == 4 and x.shape[1] == 2, "expected (n, 2, c, l)"
        n, _, c, l = x.shape
        m = int(self.segment_len)
        h = int(self.hop)
        assert 0 < m <= l and h > 0

        # optional right pad so the last window fits exactly
        pad = 0
        if pad_tail:
            rem = (l - m) % h
            pad = 0 if rem == 0 else (h - rem)
        xp = F.pad(x, (0, pad)) if pad > 0 else x

        # unfold along time -> (n, 2, c, s, m) -> (n, s, 2, c, m)
        win = xp.unfold(dimension=3, size=m, step=h)
        x_seg = win.permute(0, 3, 1, 2, 4).contiguous()
        s = x_seg.size(1)

        # within-segment mask (all True; maybe trim tail of last segment)
        mask_in = torch.ones((n, s, m), dtype=torch.bool, device=x.device)
        if pad > 0:
            last_start = (s - 1) * h
            last_real = max(0, min(m, l - last_start))
            if last_real < m:
                mask_in[:, -1, last_real:] = False

        return x_seg, mask_in

    def unsegment(self, x_seg, mask_in):
        """
        x_seg:   (n, s, 2, c, m)
        mask_in: (n, s, m)   True=real, False=pad (only tail may be False)
        orig_len: int
        returns: x: (n, 2, c, orig_len)
        """
        n, s, _, c, m = x_seg.shape
        h = int(self.hop)
        l_ola = m + (s - 1) * h

        # ---- data fold (overlap-add by summation) ----
        # (n, s, 2, c, m) -> (n, 2*c*m, s)
        patches = x_seg.permute(0, 2, 3, 4, 1).reshape(n, 2 * c * m, s)
        out = F.fold(patches, output_size=(1, l_ola), kernel_size=(1, m), stride=(1, h))
        # out: (n, 2*c, 1, l_ola) -> (n, 2, c, l_ola)
        out = out.view(n, 2, c, 1, l_ola).squeeze(3)

        # ---- count fold (to average overlaps) ----
        # mask patches: (n, s, m) -> (n, m, s)
        mask_patches = mask_in.transpose(1, 2).to(out.dtype)  # (n, m, s)
        cnt = F.fold(mask_patches, output_size=(1, l_ola),
                     kernel_size=(1, m), stride=(1, h))  # (n, 1, 1, l_ola)
        cnt = cnt.clamp_min(1e-8)

        # broadcast divide and crop
        out = out / cnt
        # true per-item lengths: (s-1)*h + real samples in last segment
        last_real = mask_in[:, -1, :].sum(dim=-1).to(torch.long)  # (n,)
        lengths = (s - 1) * h + last_real
        uniq = torch.unique(lengths)
        assert uniq.numel() == 1, "batch contains different original lengths"
        t = int(uniq.item())

        return out[..., :t]

    def random_masking(self, x, mask_ratio, token_mask=None):
        """
        Per-sample random masking that ignores pad tokens.

        Args:
          x:           (N, L, D)
          mask_ratio:  float in [0,1)
          token_mask:  (N, L) bool, True for real tokens, False for padding.
                       If None, all tokens are treated as real.

        Returns:
          x_masked:  (N, L_keep, D) where L_keep = min_i floor(real_i * (1 - mask_ratio))
          mask:      (N, L) with 0=kept, 1=removed (pads are marked removed)
          ids_restore: (N, L) indices to restore original order after masking
        """
        n, l, d = x.shape
        if token_mask is None:
            token_mask = torch.ones((n, l), dtype=torch.bool, device=x.device)

        real_counts = token_mask.sum(dim=1)  # (N,)
        # keep count per sample, then pick a single batch-wide L_keep to keep shapes rectangular
        keep_per = (real_counts.to(torch.float32) * (1.0 - mask_ratio)).floor().to(torch.long)
        l_keep = int(keep_per.min().clamp_min(1).item())  # ensure at least 1 if any real tokens exist

        # random noise for shuffling; send pads to the end by giving them +inf noise
        noise = torch.rand(n, l, device=x.device)
        noise = noise.masked_fill(~token_mask, float('inf'))

        # sort by noise (asc): real tokens come first, pads last
        ids_shuffle = torch.argsort(noise, dim=1)  # (N, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # (N, L)

        # keep the first l_keep indices (guaranteed real for every sample by construction)
        ids_keep = ids_shuffle[:, :l_keep]  # (N, L_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))

        # build binary mask over original L: 0 for kept, 1 for removed (pads count as removed)
        base = torch.ones((n, l), device=x.device, dtype=x.dtype)
        base[:, :l_keep] = 0
        mask = torch.gather(base, dim=1, index=ids_restore).to(x.dtype)

        return x_masked, mask, ids_restore

    # --- per-antenna masking (mask whole streams across all time tokens) ---
    def antenna_masking(self, tok, token_mask, num_segments: int, num_antennas: int):
        """
        Args:
          tok:        (N, L, D) tokens in time-major then antenna order (L = s*c)
          token_mask: (N, L) bool, True for real (non-pad) tokens
          num_segments: s
          num_antennas: c
        Returns:
          x_masked:   (N, L_keep, D)
          mae_mask:   (N, L)  0=kept, 1=removed (pads remain 'removed')
          ids_restore:(N, L)  identity (no shuffle needed)
        """
        N, L, D = tok.shape
        s, c = int(num_segments), int(num_antennas)
        k = max(1, min(int(self.ant_mask_k), c - 0))   # clamp; typically 1 or 2

        device = tok.device
        # build per-sample "removed" mask for selected antennas
        remove = torch.zeros((N, L), dtype=torch.bool, device=device)
        seg_idx = torch.arange(s, device=device)
        for i in range(N):
            ants = torch.randperm(c, device=device)[:k]      # choose k antennas
            # indices for these antennas across all segments: t*c + a
            idx = (seg_idx.unsqueeze(1) * c + ants.view(1, -1)).reshape(-1)  # (k*s,)
            remove[i, idx] = True

        # don't keep pads; keep = real tokens AND not removed
        keep_bool = token_mask & (~remove)
        keep_counts = keep_bool.sum(dim=1)                    # per-sample keep
        L_keep = int(keep_counts.min().clamp_min(1).item())   # rectangularize

        # gather first L_keep kept tokens per sample (preserve order)
        ids_keep_list = []
        for i in range(N):
            ids_i = torch.nonzero(keep_bool[i], as_tuple=False).squeeze(1)
            ids_keep_list.append(ids_i[:L_keep])
        ids_keep = torch.stack(ids_keep_list, dim=0)          # (N, L_keep)

        x_masked = torch.gather(tok, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mae_mask: 1 everywhere, 0 at kept indices (pads & removed remain 1)
        mae_mask = torch.ones((N, L), device=device, dtype=tok.dtype)
        mae_mask.scatter_(1, ids_keep, 0.0)

        # identity restore (we didn't shuffle)
        ids_restore = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)
        return x_masked, mae_mask, ids_restore

    def forward_encoder(self, x, time_mask, mask_ratio):
        """
        x:         (n, 2, c, t_max)  batch of resampled streams (time-padded)
        time_mask: (n, t_max)        True where samples are real (no pad)
        """
        n, _, c, t_max = x.shape
        m = int(self.segment_len)
        h = int(self.hop)

        # ---- segment whole batch (no per-item loops) ----
        # x_seg: (n, s, 2, c, m)
        x_seg, _ = self.segment(x, pad_tail=True)
        s = x_seg.shape[1]
        l_ola = m + (s - 1) * h  # length after right pad inside segment()

        # ---- build segment mask from time_mask (no loops) ----
        # pad time_mask to l_ola, then window it like the signal
        pad_tail = max(0, l_ola - t_max)
        tm = F.pad(time_mask, (0, pad_tail))  # (n, l_ola)
        seg_mask = tm.unfold(dimension=1, size=m, step=h)  # (n, s, m) bool
        # token_mask says which (time-window, antenna) tokens are real
        token_mask = seg_mask.any(dim=2).repeat_interleave(c, dim=1)  # (n, s*c)

        # ---- optional cap to max_tokens by truncating time windows ----
        if s * c > self.max_tokens:
            s_keep = self.max_tokens // c
            x_seg = x_seg[:, :s_keep]
            seg_mask = seg_mask[:, :s_keep]
            s = s_keep
            token_mask = seg_mask.any(dim=2).repeat_interleave(c, dim=1)

        # ---- per-antenna tokenization via conv over (2, m) ----
        # (n, s, 2, c, m) -> (n*s*c, 2, m) -> conv -> (n, s*c, embed_dim)
        x_tokens_2m = x_seg.permute(0, 1, 3, 2, 4).reshape(n * s * c, 2, m)
        tok = self.segment_embed(x_tokens_2m).squeeze(-1).view(n, s * c, self.embed_dim)

        # ---- add time PE + antenna embedding ----
        pe_time = self.time_pos_embed[:, 1:1 + s, :]  # (1, s, d)
        pe_time = pe_time.repeat_interleave(c, dim=1)  # (1, s*c, d)
        ant_ids = torch.arange(c, device=x.device).repeat(s)  # (s*c,)
        pe_ant = self.ant_embed(ant_ids).unsqueeze(0)  # (1, s*c, d)
        tok = tok + pe_time + pe_ant  # (n, s*c, d)

        # ---- random masking that ignores pad tokens ----
        # tok_masked, mae_mask, ids_restore = self.random_masking(tok, mask_ratio, token_mask=token_mask)
        if self.use_ant_mask:
            tok_masked, mae_mask, ids_restore = self.antenna_masking(tok, token_mask, num_segments=s, num_antennas=c)
        else:
            tok_masked, mae_mask, ids_restore = self.random_masking(tok, mask_ratio, token_mask=token_mask)

        # ---- prepend cls and run encoder blocks ----
        cls = (self.cls_token + self.time_pos_embed[:, :1, :]).expand(tok_masked.size(0), -1, -1)
        x_enc = torch.cat([cls, tok_masked], dim=1)
        for blk in self.blocks:
            x_enc = blk(x_enc)
        x_enc = self.norm(x_enc)

        return x_enc, mae_mask, ids_restore, token_mask

    def forward_decoder(self, x, ids_restore, num_segments, num_antennas):
        """
        x:            (n, 1 + L_keep, embed_dim)  encoder output (with cls)
        ids_restore:  (n, L) indices to restore masked+pad order (L = s*c)
        num_segments: int, s
        num_antennas: int, c
        returns:
          pred: (n, L, 2*segment_len)
        """
        n = x.size(0)
        s = int(num_segments)
        c = int(num_antennas)
        L = ids_restore.size(1)
        assert s * c == L, f"s*c ({s}*{c}) != L ({L})"

        # project to decoder dim
        x = self.decoder_embed(x)  # (n, 1+L_keep, d_dec)

        # append mask tokens and unshuffle to full length
        need = L + 1 - x.size(1)
        mask_tokens = self.mask_token.expand(n, need, -1)  # (n, need, d_dec)
        x_full = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (n, L, d_dec)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x = torch.cat([x[:, :1, :], x_full], dim=1)  # (n, 1+L, d_dec)

        # build decoder positional embeddings (time + antenna); cls gets time-cls only
        cls_pe = self.decoder_time_pos_embed[:, :1, :]  # (1,1,d_dec)
        time_pe = self.decoder_time_pos_embed[:, 1:1 + s, :]  # (1,s,d_dec)
        time_pe = time_pe.repeat_interleave(c, dim=1)  # (1,L,d_dec)
        ant_ids = torch.arange(c, device=x.device).repeat(s)  # (L,)
        ant_pe = self.decoder_ant_embed(ant_ids).unsqueeze(0)  # (1,L,d_dec)
        pe = torch.cat([cls_pe, time_pe + ant_pe], dim=1)  # (1,1+L,d_dec)

        # add pe, run decoder blocks
        x = x + pe
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predict per-token (I,Q) segment
        pred = self.decoder_pred(x)  # (n, 1+L, 2*K)
        pred = pred[:, 1:, :]  # drop cls → (n, L, 2*K)
        return pred

    def forward_loss(self, target_tokens, pred, mae_mask, token_mask):
        """
        Token-space MAE loss on masked real tokens.

        Args:
          target_tokens: (n, L, 2*K)  flattened (I,Q) per token
          pred:          (n, L, 2*K)  decoder outputs
          mae_mask:      (n, L)       0=kept, 1=removed (pads are 'removed' too)
          token_mask:    (n, L) bool  True for real tokens, False for pad tokens

        Returns:
          scalar loss (tensor)
        """
        per_token_mse_input = pred  # (n, L, 2*K)

        if self.norm_seg_loss:
            mu = target_tokens.mean(dim=-1, keepdim=True)  # (n,L,1)
            var = target_tokens.var(dim=-1, keepdim=True, unbiased=False)  # (n,L,1)
            std = (var + 1e-6).sqrt()
            target_n = (target_tokens - mu) / std
            pred_n = (per_token_mse_input - mu) / std
            per_token_mse = (pred_n - target_n).pow(2).mean(dim=-1)  # (n,L)
        else:
            per_token_mse = (per_token_mse_input - target_tokens).pow(2).mean(dim=-1)

        # effective mask: only masked & real tokens contribute
        effective_mask = (mae_mask > 0).to(per_token_mse.dtype) * token_mask.to(per_token_mse.dtype)  # (n, L)

        denom = effective_mask.sum().clamp_min(1e-8)
        loss = (per_token_mse * effective_mask).sum() / denom
        return loss

    def forward(self, x, time_mask, mask_ratio=0.75):
        """
        Args:
          x:         (n, 2, c, t_max)   resampled, time-padded streams
          time_mask: (n, t_max)         True where samples are real
        Returns:
          loss (scalar), pred (n, L, 2*K), mae_mask (n, L), token_mask (n, L)
        """
        n, _, c, _ = x.shape
        # ----- encoder -----
        x_enc, mae_mask, ids_restore, token_mask = self.forward_encoder(x, time_mask, mask_ratio)
        L = ids_restore.shape[1]
        s = L // c  # segments per item used after any truncation

        # ----- build token targets (n, L, 2*K) in the same (time-major, then antenna) order -----
        # segment once for the whole batch (no loops)
        x_seg, seg_mask_in = self.segment(x, pad_tail=True)  # (n, s_all, 2, c, m)
        m = int(self.segment_len)

        # if encoder truncated segments to respect max_tokens, match that here
        s_all = x_seg.shape[1]
        if s < s_all:
            x_seg = x_seg[:, :s]  # (n, s, 2, c, m)

        # flatten per (time, antenna): (n, s, 2, c, m) -> (n, s*c, 2*m)
        target_tokens = x_seg.permute(0, 1, 3, 2, 4).reshape(n, s * c, 2 * m)  # (n, L, 2*K)

        # ----- decoder -----
        pred = self.forward_decoder(x_enc, ids_restore, num_segments=s, num_antennas=c)  # (n, L, 2*K)

        # ----- loss on masked real tokens -----
        loss = self.forward_loss(target_tokens, pred, mae_mask, token_mask)

        return loss, pred, mae_mask, token_mask


def mae_vit_iq_debug(**kwargs):
    """
    ~1–2M params. Fastest smoke-test / overfit-a-batch.
    """
    return MaskedAutoencoderIQViT(
        segment_len=kwargs.pop("segment_len", 16),
        hop=kwargs.pop("hop", 16),
        max_tokens=kwargs.pop("max_tokens", 1024),
        max_antennas=kwargs.pop("max_antennas", 8),
        embed_dim=32, depth=2, num_heads=4,                 # 32/4 = 8 dim/head
        decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def mae_vit_iq_micro(**kwargs):
    """
    ~7–9M params (depending on heads/MLP); fast smoke tests & ablations.
    """
    return MaskedAutoencoderIQViT(
        segment_len=kwargs.pop("segment_len", 16),
        hop=kwargs.pop("hop", 16),
        max_tokens=kwargs.pop("max_tokens", 1024),
        max_antennas=kwargs.pop("max_antennas", 8),
        embed_dim=128, depth=4, num_heads=4,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def mae_vit_iq_tiny(**kwargs):
    """
    ~12–15M params; still lightweight, a good default for fast iteration.
    """
    return MaskedAutoencoderIQViT(
        segment_len=kwargs.pop("segment_len", 4096),
        hop=kwargs.pop("hop", 4096),
        max_tokens=kwargs.pop("max_tokens", 192),
        max_antennas=kwargs.pop("max_antennas", 8),
        embed_dim=192, depth=6, num_heads=3,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def mae_vit_iq_small(**kwargs):
    """
    ~20–30M params; balances capacity & speed. Good starting point.
    """
    return MaskedAutoencoderIQViT(
        segment_len=kwargs.pop("segment_len", 4096),
        hop=kwargs.pop("hop", 4096),
        max_tokens=kwargs.pop("max_tokens", 256),
        max_antennas=kwargs.pop("max_antennas", 16),
        embed_dim=384, depth=8, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def mae_vit_iq_base(**kwargs):
    """
    ~40–60M params; use if you need more headroom.
    """
    return MaskedAutoencoderIQViT(
        segment_len=kwargs.pop("segment_len", 4096),
        hop=kwargs.pop("hop", 4096),
        max_tokens=kwargs.pop("max_tokens", 256),
        max_antennas=kwargs.pop("max_antennas", 32),
        embed_dim=512, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

# ---- tiny config so shapes are easy to eyeball ----
# torch.manual_seed(0)
# n = 7
# c = 2
# segment_len = 16   # K
# hop = 16           # no overlap
# s_target = 3       # segments per stream (before any max_tokens trunc)
# t_max = segment_len + (s_target - 1) * hop  # 16 + 2*16 = 48
#
# # choose different true lengths to exercise time_mask logic
# t_true = torch.tensor([t_max, 35, t_max, t_max, 33, 21, 25])  # item 0 full length, item 1 shorter
#
# # ---- fake batch (n, 2, c, t_max) + time_mask (n, t_max) ----
# x = torch.randn(n, 2, c, t_max)
# time_mask = torch.zeros(n, t_max, dtype=torch.bool)
# for i in range(n):
#     x[i, :, :, t_true[i]:] = 0.0           # zero-out padded tail (not strictly required)
#     time_mask[i, : t_true[i]] = True       # mark real samples
#
# # ---- build a small model (make sure your class implements the updated methods) ----
# model = MaskedAutoencoderIQViT(
#     segment_len=segment_len,
#     hop=hop,
#     max_tokens=256,              # force truncation: s*c = 3*2 = 6 -> keep first 4 tokens
#     embed_dim=64,
#     depth=1, num_heads=4,
#     decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4,
# )
# model.eval()  # no dropout etc.
#
# # ---- run a forward pass ----
# with torch.no_grad():
#     loss, pred, mae_mask, token_mask = model(x, time_mask, mask_ratio=0.5)
#
# print("t_true:", t_true.tolist())
# print("x shape:", tuple(x.shape))                          # (n, 2, c, t_max)
# print("time_mask shape:", tuple(time_mask.shape))          # (n, t_max)
# print("pred shape (n, L, 2K):", tuple(pred.shape))         # expect (n, L, 2*segment_len)
# print("mae_mask shape (n, L):", tuple(mae_mask.shape))
# print("token_mask shape (n, L):", tuple(token_mask.shape))
# print("loss:", float(loss))
#
# # quick sanity checks
# n_, L, twoK = pred.shape
# assert twoK == 2 * segment_len, "decoder head size mismatch"
# assert mae_mask.shape == (n, L)
# assert token_mask.shape == (n, L)
# assert torch.isfinite(loss), "loss exploded to inf/NaN"
#
# # how many tokens kept after masking?
# kept_per_item = (mae_mask == 0).sum(dim=1)
# print("kept tokens per item:", kept_per_item.tolist())
