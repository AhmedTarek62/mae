# iq_vit.py (no prefix-tuning)
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from timm.models._manipulate import checkpoint_seq

from util.pos_embed import get_1d_sincos_pos_embed


class IQVisionTransformer(nn.Module):
    """
    ViT-style encoder for IQ streams (no prefix-tuning):
      - tokenization: segment time into windows of length segment_len with hop
      - per-antenna tokens via Conv1d over (I,Q) in each segment
      - add 1D sincos time PE + learned antenna embeddings
      - pool: 'token' (cls) or 'avg' (mask-aware)
    """
    def __init__(
        self,
        num_classes: int,
        segment_len: int = 4096,
        hop: int = 0,
        max_tokens: int = 256,
        max_antennas: int = 64,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        global_pool: str = "token",             # 'token' or 'avg'
        head_layers: int = 1,                   # >=1; if >1 builds MLP head
        tanh: bool = False,                     # optional bounded reg head
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        # --- tokenization hyperparams ---
        self.segment_len = int(segment_len)
        self.hop = int(hop) if hop and hop > 0 else int(segment_len)
        self.max_tokens = int(max_tokens)
        self.max_antennas = int(max_antennas)

        # --- embeddings ---
        self.embed_dim = int(embed_dim)
        self.segment_embed = nn.Conv1d(
            in_channels=2, out_channels=self.embed_dim,
            kernel_size=self.segment_len, stride=1, bias=True
        )
        time_pe = get_1d_sincos_pos_embed(self.embed_dim, self.max_tokens, cls_token=True)  # (1+L, D)
        self.register_buffer(
            "time_pos_embed",
            torch.from_numpy(time_pe).float().unsqueeze(0),  # (1, 1+max_tokens, D)
            persistent=False,
        )
        self.ant_embed = nn.Embedding(self.max_antennas, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # --- transformer encoder ---
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm_pre = nn.Identity()
        self.norm = norm_layer(self.embed_dim)
        self.grad_checkpointing = bool(grad_checkpointing)

        # pooling + head
        assert global_pool in ("token", "avg")
        self.global_pool = global_pool
        self.num_classes = int(num_classes)
        if head_layers <= 1:
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            layers = []
            for _ in range(head_layers - 1):
                layers += [nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()]
            layers += [nn.Linear(self.embed_dim, self.num_classes)]
            self.head = nn.Sequential(*layers)
        self.tanh = bool(tanh)

        # init
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.ant_embed.weight, std=0.02)
        self.apply(self._init_weights)

    # ---- utils ----
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _build_seg_mask_from_time(time_mask: torch.Tensor, seg_len: int, hop: int, s: int) -> torch.Tensor:
        seg_mask = time_mask.unfold(dimension=1, size=seg_len, step=hop)  # (n, s?, seg_len)
        if seg_mask.size(1) > s:
            seg_mask = seg_mask[:, :s, :]
        return seg_mask.contiguous()

    def segment(self, x: torch.Tensor, pad_tail: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (n, 2, c, t) -> x_seg: (n, s, 2, c, m), seg_mask_in: (n, s, m)
        """
        assert x.dim() == 4 and x.shape[1] == 2, "expected (n, 2, c, t)"
        n, _, c, t = x.shape
        m, hop = self.segment_len, self.hop
        assert m > 0 and hop > 0 and t >= m

        pad = 0
        if pad_tail:
            rem = (t - m) % hop
            pad = 0 if rem == 0 else (hop - rem)
        xp = F.pad(x, (0, pad)) if pad > 0 else x

        win = xp.unfold(dimension=3, size=m, step=hop)       # (n, 2, c, s, m)
        x_seg = win.permute(0, 3, 1, 2, 4).contiguous()      # (n, s, 2, c, m)
        s = x_seg.size(1)

        seg_mask_in = torch.ones((n, s, m), dtype=torch.bool, device=x.device)
        if pad > 0:
            last_start = (s - 1) * hop
            last_real = max(0, min(m, t - last_start))
            if last_real < m:
                seg_mask_in[:, -1, last_real:] = False

        return x_seg, seg_mask_in

    # ---- core feature extractor ----
    def tokens_from_iq(
        self,
        x: torch.Tensor,                    # (n, 2, c, t)
        time_mask: Optional[torch.Tensor]   # (n, t) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
          tok:        (n, L, d) L = s*c
          token_mask: (n, L) bool
          s: segments per item
          c: antennas
        """
        n, _, c, t = x.shape
        m, hop = self.segment_len, self.hop

        x_seg, _ = self.segment(x, pad_tail=True)  # (n, s, 2, c, m)
        s = x_seg.shape[1]
        L_all = s * c

        if L_all > self.max_tokens:
            s_keep = self.max_tokens // c
            s = max(1, s_keep)
            x_seg = x_seg[:, :s]

        if time_mask is not None:
            t_pad = m + (s - 1) * hop
            tm = F.pad(time_mask, (0, max(0, t_pad - time_mask.size(1))))
            seg_mask = self._build_seg_mask_from_time(tm, m, hop, s)  # (n, s, m)
            token_mask = seg_mask.any(dim=2).repeat_interleave(c, dim=1)  # (n, s*c)
        else:
            token_mask = torch.ones((n, s * c), dtype=torch.bool, device=x.device)

        x_2m = x_seg.permute(0, 1, 3, 2, 4).reshape(n * s * c, 2, m)  # (n*s*c, 2, m)
        tok = self.segment_embed(x_2m).squeeze(-1).view(n, s * c, self.embed_dim)  # (n, L, d)

        pe_time = self.time_pos_embed[:, 1:1 + s, :]          # (1, s, d)
        pe_time = pe_time.repeat_interleave(c, dim=1)         # (1, L, d)
        ant_ids = torch.arange(c, device=x.device).repeat(s)  # (L,)
        pe_ant = self.ant_embed(ant_ids).unsqueeze(0)         # (1, L, d)
        tok = tok + pe_time + pe_ant

        return tok, token_mask, s, c

    def forward_features(
        self,
        x: torch.Tensor,                      # (n, 2, c, t)
        time_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tok, token_mask, _, _ = self.tokens_from_iq(x, time_mask)  # (n, L, d), (n, L)
        cls = (self.cls_token + self.time_pos_embed[:, :1, :]).expand(tok.size(0), -1, -1)
        z = torch.cat([cls, tok], dim=1)  # (n, 1+L, d)

        z = self.norm_pre(z)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            z = checkpoint_seq(self.blocks, z)
        else:
            for blk in self.blocks:
                z = blk(z)
        z = self.norm(z)
        return z, token_mask

    def forward_head(self, z: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "token":
            pooled = z[:, 0]
        else:
            feats = z[:, 1:, :]                               # (n, L, d)
            maskf = token_mask.to(feats.dtype).unsqueeze(-1)  # (n, L, 1)
            denom = maskf.sum(dim=1).clamp_min(1e-6)
            pooled = (feats * maskf).sum(dim=1) / denom       # (n, d)
        out = self.head(pooled)
        return torch.tanh(out) if self.tanh else out

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, token_mask = self.forward_features(x, time_mask)
        return self.forward_head(z, token_mask)

    # ---- finetuning helpers ----
    def unfreeze_segment_embed(self):
        for p in self.segment_embed.parameters():
            p.requires_grad = True

    def freeze_encoder(self, num_blocks: Optional[int] = None):
        if num_blocks is None:
            for p in self.blocks.parameters():
                p.requires_grad = False
        else:
            for p in self.blocks[:num_blocks].parameters():
                p.requires_grad = False
        for p in self.segment_embed.parameters():
            p.requires_grad = False


def vit_iq_debug(**kwargs):
    return IQVisionTransformer(
        num_classes=kwargs.pop("num_classes"),
        segment_len=kwargs.pop("segment_len", 16),
        hop=kwargs.pop("hop", 16),
        max_tokens=kwargs.pop("max_tokens", 1024),
        max_antennas=kwargs.pop("max_antennas", 8),
        embed_dim=32, depth=2, num_heads=4, mlp_ratio=4,
        qkv_bias=True, global_pool=kwargs.pop("global_pool", "token"),
        head_layers=kwargs.pop("head_layers", 1),
        tanh=kwargs.pop("tanh", False),
        grad_checkpointing=kwargs.pop("grad_checkpointing", False),
        **kwargs
    )


def vit_iq_micro(**kwargs):
    return IQVisionTransformer(
        num_classes=kwargs.pop("num_classes"),
        segment_len=kwargs.pop("segment_len", 16),
        hop=kwargs.pop("hop", 16),
        max_tokens=kwargs.pop("max_tokens", 1024),
        max_antennas=kwargs.pop("max_antennas", 8),
        embed_dim=128, depth=4, num_heads=4, mlp_ratio=4,
        qkv_bias=True, global_pool=kwargs.pop("global_pool", "token"),
        head_layers=kwargs.pop("head_layers", 1),
        tanh=kwargs.pop("tanh", False),
        grad_checkpointing=kwargs.pop("grad_checkpointing", False),
        **kwargs
    )


def vit_iq_base(**kwargs):
    return IQVisionTransformer(
        num_classes=kwargs.pop("num_classes"),
        segment_len=kwargs.pop("segment_len", 4096),
        hop=kwargs.pop("hop", 4096),
        max_tokens=kwargs.pop("max_tokens", 256),
        max_antennas=kwargs.pop("max_antennas", 32),
        embed_dim=512, depth=12, num_heads=8, mlp_ratio=4,
        qkv_bias=True, global_pool=kwargs.pop("global_pool", "token"),
        head_layers=kwargs.pop("head_layers", 1),
        tanh=kwargs.pop("tanh", False),
        grad_checkpointing=kwargs.pop("grad_checkpointing", False),
        **kwargs
    )
