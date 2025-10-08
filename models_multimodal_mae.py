from functools import partial
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed


class MultimodalMAE(nn.Module):
    def __init__(
        self,
        # Shared backbone
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        # Decoder trunk
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        # Vision adapter config
        vis_img_size: int = 224,
        vis_patch: int = 16,
        vis_in_chans: int = 3,
        # IQ adapter config
        iq_segment_len: int = 4096,
        iq_hop: int = 4096,
        iq_max_tokens: int = 256,
        iq_max_antennas: int = 16,
        iq_use_ant_mask: bool = False,
        # Behavior
        use_conditional_ln: bool = True,
        norm_pix_loss: bool = False,
        norm_seg_loss: bool = False,
        separate_decoders: bool = False
    ):
        super().__init__()

        # --------------------------
        # Shared encoder
        # --------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.separate_decoders = bool(separate_decoders)

        # --------------------------
        # Decoder trunk(s)
        # --------------------------
        if not self.separate_decoders:
            # original shared decoder
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_blocks = nn.ModuleList(
                [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True) for _ in range(decoder_depth)]
            )
            self.decoder_norm = norm_layer(decoder_embed_dim)
        else:
            # per-modality decoder parts
            self.decoder_embed = nn.ModuleDict({
                'vision': nn.Linear(embed_dim, decoder_embed_dim, bias=True),
                'iq':     nn.Linear(embed_dim, decoder_embed_dim, bias=True),
            })
            self.decoder_mask_token = nn.ParameterDict({
                'vision': nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)),
                'iq':     nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)),
            })

            self.decoder_blocks = nn.ModuleDict({
                'vision': nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
                                         for _ in range(decoder_depth)]),
                'iq':     nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
                                         for _ in range(decoder_depth)]),
            })
            self.decoder_norm = nn.ModuleDict({
                'vision': norm_layer(decoder_embed_dim),
                'iq':     norm_layer(decoder_embed_dim),
            })

        # --------------------------
        # Vision adapter & decoder head
        # --------------------------
        self.vis_patch_size = vis_patch
        self.vis_in_chans = vis_in_chans
        self.vis_patch_embed = PatchEmbed(
            vis_img_size, vis_patch, vis_in_chans, embed_dim
        )
        num_patches = self.vis_patch_embed.num_patches
        # fixed sin-cos PE
        self.vis_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.vis_dec_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.vis_decoder_pred = nn.Linear(
            decoder_embed_dim, vis_patch ** 2 * vis_in_chans, bias=True
        )

        # --------------------------
        # IQ adapter & decoder head
        # --------------------------
        self.iq_segment_len = iq_segment_len
        self.iq_hop = iq_hop
        self.iq_max_tokens = iq_max_tokens
        self.iq_max_antennas = iq_max_antennas
        self.iq_use_ant_mask = iq_use_ant_mask

        self.iq_segment_embed = nn.Conv1d(
            in_channels=2, out_channels=embed_dim,
            kernel_size=iq_segment_len, stride=1
        )
        self.iq_time_pos_embed = nn.Parameter(
            torch.zeros(1, iq_max_tokens + 1, embed_dim), requires_grad=False
        )
        self.iq_ant_embed = nn.Embedding(iq_max_antennas, embed_dim)

        self.iq_dec_time_pos_embed = nn.Parameter(
            torch.zeros(1, iq_max_tokens + 1, decoder_embed_dim), requires_grad=False
        )
        self.iq_dec_ant_embed = nn.Embedding(iq_max_antennas, decoder_embed_dim)

        self.iq_decoder_pred = nn.Linear(
            decoder_embed_dim, 2 * iq_segment_len, bias=True
        )

        # --------------------------
        # Conditional LayerNorm (tiny FiLM)
        # --------------------------
        self.use_conditional_ln = use_conditional_ln
        if use_conditional_ln:
            self.mod_ln_scale = nn.ParameterDict({
                'vision': nn.Parameter(torch.ones(embed_dim)),
                'iq': nn.Parameter(torch.ones(embed_dim))
            })
            self.mod_ln_bias = nn.ParameterDict({
                'vision': nn.Parameter(torch.zeros(embed_dim)),
                'iq': nn.Parameter(torch.zeros(embed_dim))
            })

        # --------------------------
        # Loss flags
        # --------------------------
        self.norm_pix_loss = norm_pix_loss
        self.norm_seg_loss = norm_seg_loss

        # --------------------------
        # Init
        # --------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # -----------------------------
        # Vision sin-cos tables (frozen)
        # -----------------------------
        num_patches = self.vis_patch_embed.num_patches
        g = int(num_patches ** 0.5)
        assert g * g == num_patches, "vision expects square grid H==W"
        pe_enc = get_2d_sincos_pos_embed(self.vis_pos_embed.shape[-1], g, cls_token=True)
        self.vis_pos_embed.data.copy_(torch.from_numpy(pe_enc).float().unsqueeze(0))

        pe_dec = get_2d_sincos_pos_embed(self.vis_dec_pos_embed.shape[-1], g, cls_token=True)
        self.vis_dec_pos_embed.data.copy_(torch.from_numpy(pe_dec).float().unsqueeze(0))

        # -----------------------------
        # IQ sin-cos tables (frozen)
        # -----------------------------
        pe_time_enc = get_1d_sincos_pos_embed(self.iq_time_pos_embed.shape[-1],
                                              self.iq_max_tokens, cls_token=True)
        self.iq_time_pos_embed.data.copy_(torch.from_numpy(pe_time_enc).float().unsqueeze(0))

        pe_time_dec = get_1d_sincos_pos_embed(self.iq_dec_time_pos_embed.shape[-1],
                                              self.iq_max_tokens, cls_token=True)
        self.iq_dec_time_pos_embed.data.copy_(torch.from_numpy(pe_time_dec).float().unsqueeze(0))

        # -----------------------------
        # Conv / proj initializations
        # -----------------------------
        # PatchEmbed’s conv: init like Linear
        w = self.vis_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.vis_patch_embed.proj.bias is not None:
            nn.init.constant_(self.vis_patch_embed.proj.bias, 0)

        # IQ segment conv: init like Linear
        w = self.iq_segment_embed.weight.data  # (out, in=2, k)
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.iq_segment_embed.bias is not None:
            nn.init.constant_(self.iq_segment_embed.bias, 0)

        # Embeddings
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.iq_ant_embed.weight, std=0.02)
        torch.nn.init.normal_(self.iq_dec_ant_embed.weight, std=0.02)
        if not self.separate_decoders:
            torch.nn.init.normal_(self.mask_token, std=0.02)
        else:
            for k in ('vision', 'iq'):
                torch.nn.init.normal_(self.decoder_mask_token[k], std=0.02)

        # Optional conditional LN params
        if getattr(self, "use_conditional_ln", False):
            for k in ("vision", "iq"):
                nn.init.ones_(self.mod_ln_scale[k])
                nn.init.zeros_(self.mod_ln_bias[k])

        # -----------------------------
        # Generic init for Linear / LN
        # -----------------------------
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _decoder_parts(self, modality: str):
        """Return (embed, mask_token, blocks, norm) for the selected decoder."""
        if not self.separate_decoders:
            return self.decoder_embed, self.mask_token, self.decoder_blocks, self.decoder_norm
        return (self.decoder_embed[modality], self.decoder_mask_token[modality],
                self.decoder_blocks[modality], self.decoder_norm[modality])

    # ----------------------------
    # Vision helpers
    # ----------------------------
    def vis_patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, C, H, W) with H==W and divisible by self.vis_patch_size
        returns: (N, L, P*P*C) with L=(H/P)*(W/P)
        """
        p = int(self.vis_patch_size)
        N, C, H, W = imgs.shape
        assert H == W and (H % p == 0), "H and W must be equal and divisible by patch size"
        h = w = H // p
        x = imgs.reshape(N, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(N, h * w, p * p * C)
        return x

    def vis_unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (N, L, P*P*C)
        returns: imgs: (N, C, H, W) with H=W and L=(H/P)*(W/P)
        """
        p = int(self.vis_patch_size)
        N, L, PPc = tokens.shape
        C = PPc // (p * p)
        h = w = int(L ** 0.5)
        assert h * w == L, "L must be a perfect square for vision unpatchify"
        x = tokens.reshape(N, h, w, p, p, C)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(N, C, h * p, w * p)
        return imgs

    # ----------------------------
    # IQ helpers
    # ----------------------------
    def iq_segment(
            self, x: torch.Tensor, pad_tail: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, 2, C, T)
        returns:
          x_seg:   (N, S, 2, C, M)  segments along time
          mask_in: (N, S, M)        True for real samples within each segment (tail may be False)
        """
        assert x.dim() == 4 and x.shape[1] == 2, "expected (N, 2, C, T)"
        N, _, C, T = x.shape
        M = int(self.iq_segment_len)
        H = int(self.iq_hop) if int(self.iq_hop) > 0 else M
        assert 0 < M <= T and H > 0

        # optional right-pad so last window fits exactly
        pad = 0
        if pad_tail:
            rem = (T - M) % H
            pad = 0 if rem == 0 else (H - rem)
        xp = F.pad(x, (0, pad)) if pad > 0 else x  # pad on time dim

        # unfold over time: (N, 2, C, S, M) -> (N, S, 2, C, M)
        win = xp.unfold(dimension=3, size=M, step=H)
        x_seg = win.permute(0, 3, 1, 2, 4).contiguous()
        S = x_seg.size(1)

        # within-segment valid mask (all True except tail of last segment if padded)
        mask_in = torch.ones((N, S, M), dtype=torch.bool, device=x.device)
        if pad > 0:
            last_start = (S - 1) * H
            last_real = max(0, min(M, T - last_start))
            if last_real < M:
                mask_in[:, -1, last_real:] = False

        return x_seg, mask_in

    def iq_unsegment(self, x_seg: torch.Tensor, mask_in: torch.Tensor) -> torch.Tensor:
        """
        Overlap-add with averaging using mask counts.
        x_seg:   (N, S, 2, C, M)
        mask_in: (N, S, M)  True = real, False = pad (only tail)
        returns: x: (N, 2, C, T_orig)
        """
        N, S, _, C, M = x_seg.shape
        H = int(self.iq_hop) if int(self.iq_hop) > 0 else M
        # ola length before cropping
        T_ola = M + (S - 1) * H

        # data fold (sum overlaps)
        patches = x_seg.permute(0, 2, 3, 4, 1).reshape(N, 2 * C * M, S)  # (N, 2*C*M, S)
        data = F.fold(patches, output_size=(1, T_ola), kernel_size=(1, M), stride=(1, H))
        data = data.view(N, 2, C, 1, T_ola).squeeze(3)  # (N, 2, C, T_ola)

        # count fold (for averaging)
        cnt_patches = mask_in.transpose(1, 2).to(data.dtype)  # (N, M, S)
        cnt = F.fold(cnt_patches, output_size=(1, T_ola), kernel_size=(1, M), stride=(1, H))
        cnt = cnt.clamp_min(1e-8)  # (N, 1, 1, T_ola)

        # average and crop back to original T
        out = data / cnt
        last_real = mask_in[:, -1, :].sum(dim=-1).to(torch.long)  # (N,)
        T_orig = (S - 1) * H + last_real  # per item
        # assume same T in batch (as in training); crop by the max T_orig, but ensure per-item crop
        if torch.unique(T_orig).numel() == 1:
            T = int(T_orig[0].item())
            return out[..., :T]
        # different lengths: pad to max, caller can slice per item
        T_max = int(T_orig.max().item())
        return out[..., :T_max]

    # ----------------------------
    # Vision masking
    # ----------------------------
    def vis_random_masking(
            self,
            x: torch.Tensor,  # (N, L, D)
            mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-sample random masking à la MAE (no padding to worry about).
        Returns:
          x_masked:   (N, L_keep, D)
          mae_mask:   (N, L)   0=kept, 1=removed
          ids_restore:(N, L)
        """
        N, L, D = x.shape
        L_keep = max(1, int(L * (1.0 - mask_ratio)))

        noise = torch.rand(N, L, device=x.device)  # [0,1)
        ids_shuffle = torch.argsort(noise, dim=1)  # asc
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :L_keep]  # (N, L_keep)
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mae_mask = torch.ones((N, L), device=x.device, dtype=x.dtype)
        mae_mask[:, :L_keep] = 0.0
        mae_mask = torch.gather(mae_mask, 1, ids_restore)
        return x_masked, mae_mask, ids_restore

    # ----------------------------
    # IQ masking: random (ignores pad tokens)
    # ----------------------------
    def iq_random_masking(
            self,
            tokens: torch.Tensor,  # (N, L, D)
            token_mask: torch.Tensor,  # (N, L) bool: True=real, False=pad
            mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-sample random masking that *never* keeps pad tokens.
        Pads are treated as 'removed' in mae_mask.
        Returns:
          x_masked:   (N, L_keep, D)
          mae_mask:   (N, L)  0=kept, 1=removed (pads are 1)
          ids_restore:(N, L)
        """
        N, L, D = tokens.shape
        device = tokens.device

        real_counts = token_mask.sum(dim=1)  # (N,)
        # if any sample has 0 real tokens, keep at least 1 to avoid empty gather
        L_keep_each = (real_counts.to(torch.float32) * (1.0 - mask_ratio)).floor().to(torch.long)
        L_keep = int(L_keep_each.clamp_min(1).min().item())

        # Noise for real tokens; +inf for pads so they go to the end
        noise = torch.rand(N, L, device=device)
        noise = noise.masked_fill(~token_mask, float('inf'))

        ids_shuffle = torch.argsort(noise, dim=1)  # real first, pads last
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :L_keep]  # guaranteed real
        x_masked = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        base = torch.ones((N, L), device=device, dtype=tokens.dtype)
        base[:, :L_keep] = 0.0
        mae_mask = torch.gather(base, 1, ids_restore)  # pads remain 1
        return x_masked, mae_mask, ids_restore

    # ----------------------------
    # IQ masking: antenna streams (mask whole antennas across all time tokens)
    # ----------------------------
    def iq_antenna_masking(
            self,
            tokens: torch.Tensor,  # (N, L, D), L = S*C in (time-major then antenna) order
            token_mask: torch.Tensor,  # (N, L) bool
            segments: int,  # S
            antennas: int,  # C
            k: int = 1
    ):
        """
        Vectorized antenna-stream masking:
          1) pick k antennas per sample via top-k on random noise (N,C)
          2) broadcast selection across segments to (N,S,C) -> flatten to (N,L)
          3) keep order by sorting masked indices via an order-preserving trick
        Returns:
          x_masked:    (N, L_keep, D)
          mae_mask:    (N, L)  0=kept, 1=removed
          ids_restore: (N, L)  identity (no shuffle)
        """
        N, L, D = tokens.shape
        S, C = int(segments), int(antennas)
        assert S * C == L, f"L={L} must equal S*C={S * C}"
        device = tokens.device

        k = max(1, min(int(k), C))  # clamp

        # --- (1) choose k antennas per sample (N,C) ---
        noise = torch.rand(N, C, device=device)
        _, topk_idx = torch.topk(noise, k, dim=1, largest=True)
        sel = torch.zeros(N, C, dtype=torch.bool, device=device)
        sel.scatter_(1, topk_idx, True)  # True for antennas to remove

        # --- (2) expand across segments and flatten to (N,L=S*C) ---
        remove = sel.unsqueeze(1).expand(-1, S, -1).reshape(N, L)  # True = remove
        keep_bool = token_mask & (~remove)  # keep only real, non-removed tokens

        # Per-sample keep counts and batch-wide rectangular L_keep
        keep_counts = keep_bool.sum(dim=1)  # (N,)
        L_keep = int(keep_counts.clamp_min(1).min().item())

        # --- (3) order-preserving gather without loops ---
        # Give non-kept indices a large penalty so they sort to the end,
        # kept indices keep their original order (ascending by index).
        arangeL = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)  # (N,L)
        penalty = (~keep_bool).to(arangeL.dtype) * (L * 2)
        ranks = arangeL + penalty  # kept: 0..L-1, removed: big
        ids_sorted = torch.argsort(ranks, dim=1)  # kept first, in-order
        ids_keep = ids_sorted[:, :L_keep]  # (N,L_keep)

        x_masked = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mae_mask = torch.ones((N, L), device=device, dtype=tokens.dtype)
        mae_mask.scatter_(1, ids_keep, 0.0)  # kept → 0

        ids_restore = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)  # identity
        return x_masked, mae_mask, ids_restore

    # ============================
    # Shared encode/decode trunk
    # ============================

    def _encode(
            self,
            tokens_masked: torch.Tensor,  # (N, L_keep, D)
            cls_pe: torch.Tensor,
            modality: str  # 'vision' | 'iq' (for optional conditional LN)
    ) -> torch.Tensor:
        # prepend CLS (already position-encoded in pe_enc[ :, :1 ])
        cls = self.cls_token.expand(tokens_masked.size(0), -1, -1)
        x = torch.cat([cls, tokens_masked], dim=1)  # (N, 1+L_keep, D)
        x[:, :1, :] = x[:, :1, :] + cls_pe
        # (optional) conditional LN FiLM at the input (light touch)
        if getattr(self, "use_conditional_ln", False):
            g = self.mod_ln_scale[modality]
            b = self.mod_ln_bias[modality]
            x = x * g + b
        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def _decode_trunk(
            self,
            modality: str,
            x_latent: torch.Tensor,  # (N, 1+L_keep, D)
            ids_restore: torch.Tensor,  # (N, L)
            pe_dec: torch.Tensor  # (1, 1+L, D_dec)
    ) -> torch.Tensor:
        # fetch per-modality (or shared) decoder parts
        dec_embed, mask_token, dec_blocks, dec_norm = self._decoder_parts(modality)

        N = x_latent.size(0)
        L = ids_restore.size(1)

        # project to decoder dim
        x = dec_embed(x_latent)  # (N, 1+L_keep, D_dec)

        # append mask tokens to reach full length L (drop CLS for this part)
        need = L + 1 - x.size(1)
        mask_tokens = mask_token.expand(N, need, -1)  # (N, need, D_dec)
        x_full = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (N, L, D_dec)

        # unshuffle to original order
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x = torch.cat([x[:, :1, :], x_full], dim=1)  # (N, 1+L, D_dec)

        # add decoder PE and run blocks
        x = x + pe_dec
        for blk in dec_blocks:
            x = blk(x)
        x = dec_norm(x)
        return x

    # ============================
    # Vision path
    # ============================
    def vision_targets(self, imgs: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) -> (N, L, p*p*C)
        return self.vis_patchify(imgs)

    def vision_loss(
            self,
            imgs: torch.Tensor,
            pred_tokens: torch.Tensor,  # (N, L, p*p*C)
            mae_mask: torch.Tensor  # (N, L) 0=kept, 1=removed
    ) -> torch.Tensor:
        target = self.vision_targets(imgs)  # (N, L, PP*C)
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mu) / (var + 1e-6).sqrt()
        per_patch = (pred_tokens - target).pow(2).mean(dim=-1)  # (N, L)
        loss = (per_patch * mae_mask).sum() / mae_mask.sum().clamp_min(1e-8)
        return loss

    def forward_vision(
            self,
            imgs: torch.Tensor,  # (N, C, H, W)
            mask_ratio: float = 0.75
    ):
        # tokenize
        tok = self.vis_patch_embed(imgs)  # (N, L, D)
        # add encoder positional embeddings
        tok = tok + self.vis_pos_embed[:, 1:, :]
        # mask
        tok_masked, mae_mask, ids_restore = self.vis_random_masking(tok, mask_ratio)
        # encoder PE (CLS)
        cls_pe = self.vis_pos_embed[:, :1, :]  # (1,1,D)
        # encode
        x_latent = self._encode(tok_masked, cls_pe, modality='vision')
        # decoder PE uses full length (1+L)
        L = ids_restore.size(1)
        pe_dec = torch.cat([self.vis_dec_pos_embed[:, :1, :], self.vis_dec_pos_embed[:, 1:1 + L, :]], dim=1)
        # decode trunk
        x_dec = self._decode_trunk('vision', x_latent, ids_restore, pe_dec)  # (N, 1+L, D_dec)
        # predict tokens, drop CLS
        pred_tokens = self.vis_decoder_pred(x_dec[:, 1:, :])  # (N, L, PP*C)
        # loss on masked patches
        loss = self.vision_loss(imgs, pred_tokens, mae_mask)
        extras = dict(ids_restore=ids_restore, L=L)
        return loss, pred_tokens, mae_mask, extras

    # ============================
    # IQ path
    # ============================
    def iq_targets(
            self,
            x: torch.Tensor,  # (N, 2, C, T)
            segments_keep: int,  # S actually used by encoder (after truncation)
            antennas: int  # C
    ) -> torch.Tensor:
        # Build token targets in (time-major then antenna) order: (N, S*C, 2*M)
        x_seg, _ = self.iq_segment(x, pad_tail=True)  # (N, S_all, 2, C, M)
        N, S_all, _, C, M = x_seg.shape
        S = min(segments_keep, S_all)
        x_seg = x_seg[:, :S]  # (N, S, 2, C, M)
        target_tokens = x_seg.permute(0, 1, 3, 2, 4).reshape(N, S * C, 2 * M)
        return target_tokens

    def iq_loss(
            self,
            target_tokens: torch.Tensor,  # (N, L, 2*M)
            pred_tokens: torch.Tensor,  # (N, L, 2*M)
            mae_mask: torch.Tensor,  # (N, L) 0=kept, 1=removed
            token_mask: torch.Tensor  # (N, L) bool True=real
    ) -> torch.Tensor:
        if self.norm_seg_loss:
            mu = target_tokens.mean(dim=-1, keepdim=True)
            var = target_tokens.var(dim=-1, keepdim=True, unbiased=False)
            std = (var + 1e-6).sqrt()
            target_n = (target_tokens - mu) / std
            pred_n = (pred_tokens - mu) / std
            per_tok = (pred_n - target_n).pow(2).mean(dim=-1)  # (N, L)
        else:
            per_tok = (pred_tokens - target_tokens).pow(2).mean(dim=-1)

        eff = (mae_mask > 0).to(per_tok.dtype) * token_mask.to(per_tok.dtype)
        loss = (per_tok * eff).sum() / eff.sum().clamp_min(1e-8)
        return loss

    def forward_iq(
            self,
            x: torch.Tensor,  # (N, 2, C, T)
            time_mask: torch.Tensor,  # (N, T) True=real
            mask_ratio: float = 0.75,
            use_ant_mask: Optional[bool] = None
    ):
        N, _, C, _ = x.shape
        M = int(self.iq_segment_len)
        H = int(self.iq_hop) if int(self.iq_hop) > 0 else M

        # 1) Segment & build token_mask
        x_seg, seg_mask_in = self.iq_segment(x, pad_tail=True)  # (N, S, 2, C, M)
        S_all = x_seg.size(1)

        # Construct token_mask from time_mask via unfolding
        # first align time_mask to overlap-add length
        T_ola = M + (S_all - 1) * H
        pad_tail = max(0, T_ola - time_mask.size(1))
        tm = F.pad(time_mask, (0, pad_tail))  # (N, T_ola)
        seg_mask = tm.unfold(1, M, H)  # (N, S_all, M) bool
        token_mask = seg_mask.any(dim=2).repeat_interleave(C, dim=1)  # (N, S_all*C)

        # 2) Truncate to respect max_tokens
        if S_all * C > self.iq_max_tokens:
            S = self.iq_max_tokens // C
            x_seg = x_seg[:, :S]
            seg_mask = seg_mask[:, :S]
            token_mask = seg_mask.any(dim=2).repeat_interleave(C, dim=1)
        else:
            S = S_all

        # 3) Tokenize per (time, antenna)
        x_tokens_2m = x_seg.permute(0, 1, 3, 2, 4).reshape(N * S * C, 2, M)  # (N*S*C, 2, M)
        tok = self.iq_segment_embed(x_tokens_2m).squeeze(-1).view(N, S * C, -1)  # (N, L, D) L=S*C

        # 4) Add PE: time + antenna
        pe_time = self.iq_time_pos_embed[:, 1:1 + S, :].repeat_interleave(C, dim=1)  # (1, L, D)
        ant_ids = torch.arange(C, device=x.device).repeat(S)  # (L,)
        pe_ant = self.iq_ant_embed(ant_ids).unsqueeze(0)  # (1, L, D)
        tok = tok + pe_time + pe_ant

        # 5) Masking (random vs antenna)
        use_ant = self.iq_use_ant_mask if use_ant_mask is None else bool(use_ant_mask)
        if use_ant:
            tok_masked, mae_mask, ids_restore = self.iq_antenna_masking(tok, token_mask, S, C, k=1)
        else:
            tok_masked, mae_mask, ids_restore = self.iq_random_masking(tok, token_mask, mask_ratio)

        # 6) Encoder PE (CLS + kept tokens)
        cls_pe = self.iq_time_pos_embed[:, :1, :]  # CLS time slot

        # 7) Encode
        x_latent = self._encode(tok_masked, cls_pe, modality='iq')

        # 8) Decoder PE uses full L
        pe_time_dec = self.iq_dec_time_pos_embed[:, :1, :].clone()
        pe_time_slice_d = self.iq_dec_time_pos_embed[:, 1:1 + S, :].repeat_interleave(C, dim=1)
        pe_ant_slice_d = self.iq_dec_ant_embed(ant_ids).unsqueeze(0)
        pe_dec_full = torch.cat([pe_time_dec, pe_time_slice_d + pe_ant_slice_d], dim=1)  # (1, 1+L, D_dec)

        # 9) Decode trunk
        x_dec = self._decode_trunk('iq', x_latent, ids_restore, pe_dec_full)  # (N, 1+L, D_dec)

        # 10) Predict per-token (I,Q) segments, drop CLS
        pred_tokens = self.iq_decoder_pred(x_dec[:, 1:, :])  # (N, L, 2*M)

        # 11) Targets and loss
        target_tokens = self.iq_targets(x, segments_keep=S, antennas=C)  # (N, L, 2*M)
        loss = self.iq_loss(target_tokens, pred_tokens, mae_mask, token_mask)

        extras = dict(ids_restore=ids_restore, token_mask=token_mask, segments=S, antennas=C)
        return loss, pred_tokens, mae_mask, extras

    # ============================
    # Unified forward
    # ============================
    def forward(
            self,
            modality: str,  # 'vision' | 'iq'
            x: torch.Tensor,
            *,
            mask_ratio: float = 0.75,
            **aux
    ):
        if modality == 'vision':
            return self.forward_vision(x, mask_ratio=mask_ratio)
        elif modality == 'iq':
            time_mask = aux.get('time_mask', None)
            assert time_mask is not None, "forward('iq', ...) requires time_mask (N, T)"
            use_ant_mask = aux.get('use_ant_mask', None)
            return self.forward_iq(x, time_mask, mask_ratio=mask_ratio, use_ant_mask=use_ant_mask)
        else:
            raise ValueError(f"Unknown modality: {modality}")


# --- Presets (registry) -------------------------------------------------------
def mae_multi_micro(**kwargs):
    """
    Tiny config to smoke-test end-to-end (fast).
    """
    return MultimodalMAE(
        embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=8,
        use_conditional_ln=True,
        norm_pix_loss=kwargs.pop("norm_pix_loss", False),
        norm_seg_loss=kwargs.pop("norm_seg_loss", False),
        iq_use_ant_mask=kwargs.pop("iq_use_ant_mask", False),
        separate_decoders=kwargs.pop("separate_decoders", False)
    )


def mae_multi_small(**kwargs):
    """
    ~20–30M params; good default.
    """
    return MultimodalMAE(
        embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=16,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=16,
        use_conditional_ln=True,
        norm_pix_loss=kwargs.pop("norm_pix_loss", False),
        norm_seg_loss=kwargs.pop("norm_seg_loss", False),
        iq_use_ant_mask=kwargs.pop("iq_use_ant_mask", False),
        separate_decoders=kwargs.pop("separate_decoders", False)
    )


def mae_multi_base(**kwargs):
    """
    ~40–60M params.
    """
    return MultimodalMAE(
        embed_dim=512, depth=12, num_heads=8, mlp_ratio=4.0,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=32,
        use_conditional_ln=True,
        norm_pix_loss=kwargs.pop("norm_pix_loss", False),
        norm_seg_loss=kwargs.pop("norm_seg_loss", False),
        iq_use_ant_mask=kwargs.pop("iq_use_ant_mask", False),
        separate_decoders=kwargs.pop("separate_decoders", False)
    )


def mae_multi_large(**kwargs):
    """
    ~80M params+ (heavier).
    """
    return MultimodalMAE(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=32,
        use_conditional_ln=True,
        norm_pix_loss=kwargs.pop("norm_pix_loss", False),
        norm_seg_loss=kwargs.pop("norm_seg_loss", False),
        iq_use_ant_mask=kwargs.pop("iq_use_ant_mask", False),
        separate_decoders=kwargs.pop("separate_decoders", False)
    )


# Optional aliases to match your naming patterns
mae_vit_multi_micro = mae_multi_micro
mae_vit_multi_small = mae_multi_small
mae_vit_multi_base = mae_multi_base
mae_vit_multi_large = mae_multi_large
