from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block, PatchEmbed
from timm.models._manipulate import checkpoint_seq

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed


class ModalityAdapterViT(nn.Module):
    """
    A *simple* ViT fine-tuner that can operate on exactly **one** modality per
    instantiation: either 'vision' or 'iq'. The shared encoder & head behavior
    mirrors your `IQVisionTransformer`, but the input adapter swaps based on
    the chosen modality.

    Use cases
      - Fine-tune on *vision* tasks: pass modality='vision' and provide images.
      - Fine-tune on *iq* tasks: pass modality='iq' and provide (N,2,C,T) IQ.

    Key features
      - Minimal surface area: one adapter picked at init, one head.
      - Mask-aware average pooling for variable-length IQ tokens.
      - Helpers to freeze encoder or tokenizers.
      - (Vision) Optional **channel adapter** to handle input channel count
        mismatches without touching PatchEmbed weights.
    """

    def __init__(
        self,
        modality: str,                     # 'vision' | 'iq'
        num_classes: int,
        # --- shared encoder ---
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        global_pool: str = "token",        # 'token' | 'avg'
        head_layers: int = 1,               # >=1 (if >1 -> MLP head)
        tanh: bool = False,                 # optional bounded reg head
        grad_checkpointing: bool = False,
        use_conditional_ln: bool = False,
        # --- vision adapter ---
        vis_img_size: int = 224,
        vis_patch: int = 16,
        vis_in_chans: int = 1,              # channels PatchEmbed is initialized for (pretrain)
        vis_in_chans_actual: Optional[int] = None,  # task's actual channels; if differs, use 1x1 adapter
        channel_adapter_init: str = "avg",  # 'avg' | 'repeat' | 'zero' | 'rand'
        # --- iq adapter ---
        iq_segment_len: int = 4096,
        iq_hop: int = 0,
        iq_max_tokens: int = 256,
        iq_max_antennas: int = 64,
    ):
        super().__init__()
        assert modality in ("vision", "iq"), "modality must be 'vision' or 'iq'"
        self.modality = modality
        self.global_pool = global_pool
        self.grad_checkpointing = bool(grad_checkpointing)
        self.use_conditional_ln = bool(use_conditional_ln)

        # --- shared encoder ---
        self.embed_dim = int(embed_dim)
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm_pre = nn.Identity()
        self.norm = norm_layer(self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # Optional conditional FiLM (same shapes & names as MAE for easy weight load)
        if self.use_conditional_ln:
            self.mod_ln_scale = nn.ParameterDict({
                'vision': nn.Parameter(torch.ones(self.embed_dim)),
                'iq': nn.Parameter(torch.ones(self.embed_dim)),
            })
            self.mod_ln_bias = nn.ParameterDict({
                'vision': nn.Parameter(torch.zeros(self.embed_dim)),
                'iq': nn.Parameter(torch.zeros(self.embed_dim)),
            })

        # --- head ---
        if head_layers <= 1:
            self.head = nn.Linear(self.embed_dim, int(num_classes))
        else:
            h = []
            for _ in range(head_layers - 1):
                h += [nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()]
            h += [nn.Linear(self.embed_dim, int(num_classes))]
            self.head = nn.Sequential(*h)
        self.tanh = bool(tanh)

        # --- vision adapter ---
        if modality == 'vision':
            self.vis_patch_size = int(vis_patch)
            self.vis_patch_embed = PatchEmbed(vis_img_size, vis_patch, vis_in_chans, self.embed_dim)
            num_patches = self.vis_patch_embed.num_patches
            g = int(num_patches ** 0.5)
            assert g * g == num_patches, "vision expects square grid H==W"
            pe_2d = get_2d_sincos_pos_embed(self.embed_dim, g, cls_token=True)
            self.register_buffer(
                "vis_pos_embed",
                torch.from_numpy(pe_2d).float().unsqueeze(0),  # (1,1+L,D)
                persistent=False,
            )
            # Optional channel adapter to map C_actual -> vis_in_chans (pretrained)
            self.vis_in_chans_pretrained = int(vis_in_chans)
            c_actual = int(vis_in_chans_actual) if vis_in_chans_actual is not None else int(vis_in_chans)
            self.channel_adapter = None
            if c_actual != self.vis_in_chans_pretrained:
                self.channel_adapter = nn.Conv2d(c_actual, self.vis_in_chans_pretrained, kernel_size=1, bias=False)
                self._init_channel_adapter(self.channel_adapter, init=channel_adapter_init)

        # --- iq adapter ---
        if modality == 'iq':
            self.segment_len = int(iq_segment_len)
            self.hop = int(iq_hop) if iq_hop and iq_hop > 0 else int(iq_segment_len)
            self.max_tokens = int(iq_max_tokens)
            self.max_antennas = int(iq_max_antennas)

            self.iq_segment_embed = nn.Conv1d(
                in_channels=2, out_channels=self.embed_dim,
                kernel_size=self.segment_len, stride=1, bias=True
            )
            pe_1d = get_1d_sincos_pos_embed(self.embed_dim, self.max_tokens, cls_token=True)
            self.register_buffer(
                "iq_time_pos_embed",
                torch.from_numpy(pe_1d).float().unsqueeze(0),  # (1,1+Lmax,D)
                persistent=False,
            )
            self.iq_ant_embed = nn.Embedding(self.max_antennas, self.embed_dim)

        # init
        nn.init.normal_(self.cls_token, std=0.02)
        if modality == 'iq':
            nn.init.normal_(self.iq_ant_embed.weight, std=0.02)
        self.apply(self._init_weights)

    # ------------------ init utils ------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _init_channel_adapter(conv1x1: nn.Conv2d, init: str = "avg"):
        """Initialize 1x1 channel adapter mapping C_actual -> C_pretrained.
        Strategies:
          - 'avg': each output channel sees the average of all input channels (good default)
          - 'repeat': if C_actual divides C_pretrained, map in groups; else fallback to tile + trim
          - 'zero': start from zeros (lets model learn linear combo from scratch)
          - 'rand': Kaiming normal
        """
        if init == 'zero':
            nn.init.zeros_(conv1x1.weight)
            return
        if init == 'rand':
            nn.init.kaiming_normal_(conv1x1.weight, nonlinearity='linear')
            return
        # Build a simple deterministic mapping matrix W: (C_out, C_in, 1, 1)
        C_out, C_in = conv1x1.out_channels, conv1x1.in_channels
        W = torch.zeros(C_out, C_in)
        if init == 'repeat' and C_in > 0:
            # round-robin repeat
            for o in range(C_out):
                W[o, o % C_in] = 1.0
        else:  # 'avg' default
            W[:] = 1.0 / max(1, C_in)
        conv1x1.weight.data.copy_(W.view(C_out, C_in, 1, 1))

    # ------------------ IQ helpers ------------------
    @staticmethod
    def _build_seg_mask_from_time(time_mask: torch.Tensor, seg_len: int, hop: int, s: int) -> torch.Tensor:
        seg_mask = time_mask.unfold(dimension=1, size=seg_len, step=hop)  # (N,S?,M)
        if seg_mask.size(1) > s:
            seg_mask = seg_mask[:, :s, :]
        return seg_mask.contiguous()

    def iq_segment(self, x: torch.Tensor, pad_tail: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, 2, C, T) -> (x_seg: (N,S,2,C,M), seg_mask_in: (N,S,M))
        """
        assert x.dim() == 4 and x.shape[1] == 2, "expected (N,2,C,T)"
        N, _, C, T = x.shape
        M, H = self.segment_len, self.hop
        assert M > 0 and H > 0 and T >= M

        pad = 0
        if pad_tail:
            rem = (T - M) % H
            pad = 0 if rem == 0 else (H - rem)
        xp = F.pad(x, (0, pad)) if pad > 0 else x

        win = xp.unfold(dimension=3, size=M, step=H)        # (N,2,C,S,M)
        x_seg = win.permute(0, 3, 1, 2, 4).contiguous()      # (N,S,2,C,M)
        S = x_seg.size(1)

        seg_mask_in = torch.ones((N, S, M), dtype=torch.bool, device=x.device)
        if pad > 0:
            last_start = (S - 1) * H
            last_real = max(0, min(M, T - last_start))
            if last_real < M:
                seg_mask_in[:, -1, last_real:] = False
        return x_seg, seg_mask_in

    # ------------------ tokenizers ------------------
    def _tokens_from_vision(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.channel_adapter is not None:
            imgs = self.channel_adapter(imgs)
        tok = self.vis_patch_embed(imgs)              # (N,L,D)
        tok = tok + self.vis_pos_embed[:, 1:, :]      # add 2D PE (no CLS)
        token_mask = torch.ones(tok.size(0), tok.size(1), dtype=torch.bool, device=tok.device)
        return tok, token_mask

    def _tokens_from_iq(self, x: torch.Tensor, time_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        N, _, C, _ = x.shape
        M, H = self.segment_len, self.hop

        x_seg, _ = self.iq_segment(x, pad_tail=True)        # (N,S_all,2,C,M)
        S_all = x_seg.size(1)

        if time_mask is not None:
            T_ola = M + (S_all - 1) * H
            pad_tail = max(0, T_ola - time_mask.size(1))
            tm = F.pad(time_mask, (0, pad_tail))            # (N, T_ola)
            seg_mask = tm.unfold(1, M, H)                   # (N,S_all,M)
            token_mask = seg_mask.any(dim=2).repeat_interleave(C, dim=1)  # (N,S_all*C)
        else:
            token_mask = torch.ones((N, S_all * C), dtype=torch.bool, device=x.device)

        # cap by max_tokens (truncate segments)
        if S_all * C > self.max_tokens:
            S = self.max_tokens // C
            x_seg = x_seg[:, :S]
            if time_mask is not None:
                seg_mask = seg_mask[:, :S]
                token_mask = seg_mask.any(dim=2).repeat_interleave(C, dim=1)
        else:
            S = S_all

        x_2m = x_seg.permute(0, 1, 3, 2, 4).reshape(N * S * C, 2, M)     # (N*S*C,2,M)
        tok = self.iq_segment_embed(x_2m).squeeze(-1).view(N, S * C, self.embed_dim)

        pe_time = self.iq_time_pos_embed[:, 1:1 + S, :].repeat_interleave(C, dim=1)   # (1,L,D)
        ant_ids = torch.arange(C, device=x.device).repeat(S)
        pe_ant = self.iq_ant_embed(ant_ids).unsqueeze(0)
        tok = tok + pe_time + pe_ant
        return tok, token_mask

    # ------------------ encoder & head ------------------
    def forward_features(self, tok: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls_token.expand(tok.size(0), 1, -1)
        z = torch.cat([cls, tok], dim=1)     # (N,1+L,D)
        z = self.norm_pre(z)
        # Apply conditional FiLM like MultimodalMAE (_encode)
        if getattr(self, 'use_conditional_ln', False):
            g = self.mod_ln_scale[self.modality]
            b = self.mod_ln_bias[self.modality]
            z = z * g + b
        if self.grad_checkpointing and not torch.jit.is_scripting():
            z = checkpoint_seq(self.blocks, z)
        else:
            for blk in self.blocks:
                z = blk(z)
        z = self.norm(z)
        return z, token_mask

    def forward_head(self, z: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if self.global_pool == 'token':
            pooled = z[:, 0]
        else:
            feats = z[:, 1:, :]
            maskf = token_mask.to(feats.dtype).unsqueeze(-1)
            denom = maskf.sum(dim=1).clamp_min(1e-6)
            pooled = (feats * maskf).sum(dim=1) / denom
        out = self.head(pooled)
        return torch.tanh(out) if self.tanh else out

    # ------------------ public forward ------------------
    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.modality == 'vision':
            tok, token_mask = self._tokens_from_vision(x)
        else:  # 'iq'
            tok, token_mask = self._tokens_from_iq(x, time_mask)
        z, token_mask = self.forward_features(tok, token_mask)
        return self.forward_head(z, token_mask)

    # ------------------ finetuning helpers ------------------
    def freeze_encoder(self, num_blocks: Optional[int] = None):
        if num_blocks is None:
            for p in self.blocks.parameters():
                p.requires_grad = False
        else:
            for p in self.blocks[:num_blocks].parameters():
                p.requires_grad = False
        # freeze tokenizers too
        if self.modality == 'vision':
            for p in self.vis_patch_embed.parameters():
                p.requires_grad = False
            if self.channel_adapter is not None:
                for p in self.channel_adapter.parameters():
                    p.requires_grad = False
        else:
            for p in self.iq_segment_embed.parameters():
                p.requires_grad = False

    def unfreeze_tokenizer(self):
        if self.modality == 'vision':
            for p in self.vis_patch_embed.parameters():
                p.requires_grad = True
            if self.channel_adapter is not None:
                for p in self.channel_adapter.parameters():
                    p.requires_grad = True
        else:
            for p in self.iq_segment_embed.parameters():
                p.requires_grad = True

    # -------- conditional LN helpers (useful when encoder is frozen) --------
    def freeze_conditional_ln(self):
        if getattr(self, 'use_conditional_ln', False):
            for p in self.mod_ln_scale.values():
                p.requires_grad = False
            for p in self.mod_ln_bias.values():
                p.requires_grad = False

    def unfreeze_conditional_ln(self):
        if getattr(self, 'use_conditional_ln', False):
            for p in self.mod_ln_scale.values():
                p.requires_grad = True
            for p in self.mod_ln_bias.values():
                p.requires_grad = True


# ----------------------------
# Tiny registry (vision or iq)
# ----------------------------
def vit_multi_micro(**kwargs):
    """Tiny config to smoke-test end-to-end (fast)."""
    return ModalityAdapterViT(
        embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=8, **kwargs
    )


def vit_multi_small(**kwargs):
    """~20–30M params; good default."""
    return ModalityAdapterViT(
        embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=16, **kwargs
    )


def vit_multi_base(**kwargs):
    """~40–60M params."""
    return ModalityAdapterViT(
        embed_dim=512, depth=12, num_heads=8, mlp_ratio=4.0,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=32, **kwargs
    )


def vit_multi_large(**kwargs):
    """~80M params+ (heavier)."""
    return ModalityAdapterViT(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        vis_img_size=224, vis_patch=16, vis_in_chans=1,
        iq_segment_len=16, iq_hop=16, iq_max_tokens=256, iq_max_antennas=32, **kwargs
    )