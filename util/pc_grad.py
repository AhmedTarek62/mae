# Minimal utilities for 2-task (vision/iq) PCGrad with round-robin + accum_iter=2.
# Apply PCGrad ONLY on shared encoder params; sum grads for decoders/adapters.

from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import torch


# -------- param selection --------
def shared_encoder_params(model) -> List[torch.nn.Parameter]:
    """Trainable parameters that belong to the shared encoder."""
    return [p for n, p in model.named_parameters()
            if p.requires_grad and (n.startswith("encoder") or n.startswith("blocks") or n.startswith("norm"))]


def decoder_adapter_params(model) -> List[torch.nn.Parameter]:
    """Trainable parameters that do NOT belong to the shared encoder."""
    return [p for n, p in model.named_parameters()
            if p.requires_grad and not (n.startswith("encoder") or n.startswith("blocks") or n.startswith("norm"))]


# -------- grad math --------
def _dot(g1: List[Optional[torch.Tensor]], g2: List[Optional[torch.Tensor]]) -> torch.Tensor:
    s = None
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        v = (a.detach() * b.detach()).sum()
        s = v if s is None else s + v
    if s is None:
        s = torch.tensor(0.0, device=g1[0].device if g1 and g1[0] is not None else "cpu")
    return s


def _norm2(g: List[Optional[torch.Tensor]]) -> torch.Tensor:
    s = None
    for t in g:
        if t is None:
            continue
        v = (t.detach() ** 2).sum()
        s = v if s is None else s + v
    if s is None:
        s = torch.tensor(0.0, device=g[0].device if g and g[0] is not None else "cpu")
    return s + 1e-12


def project_conflict(g: List[Optional[torch.Tensor]],
                     ref: List[Optional[torch.Tensor]]
                     ) -> Tuple[List[Optional[torch.Tensor]], bool]:
    """
    If g · ref < 0, project g onto the normal plane of ref:
        g <- g - (g·ref / ||ref||^2) ref
    Returns (projected_g, did_project).
    """
    dot = _dot(g, ref)
    if dot.item() >= 0:
        return g, False
    coef = (dot / _norm2(ref)).clamp(min=-1e6, max=1e6)
    out = []
    for gi, ri in zip(g, ref):
        if gi is None or ri is None:
            out.append(gi)
        else:
            out.append(gi - coef * ri)
    return out, True


def assign_grads(params: List[torch.nn.Parameter],
                 grads: List[Optional[torch.Tensor]],
                 accumulate: bool = False) -> None:
    """Write grads into .grad (optionally accumulate)."""
    for p, g in zip(params, grads):
        if g is None:
            continue
        if (p.grad is None) or (not accumulate):
            p.grad = g.detach().clone()
        else:
            p.grad.add_(g.detach())


# -------- buffering for round-robin 2-task training --------
class PCGradPairBuffer:
    """
    Buffer that collects one 'vision' and one 'iq' step (any order),
    then emits combined (projected) encoder grads and summed decoder grads.

    Usage:
        buf = PCGradPairBuffer(enc_params, dec_params)
        buf.add('vision', L_vis)
        if buf.ready(): enc_g, dec_g, projected = buf.flush()
        buf.add('iq', L_iq)
        if buf.ready(): enc_g, dec_g, projected = buf.flush()
    """

    def __init__(self,
                 enc_params: List[torch.nn.Parameter],
                 dec_params: List[torch.nn.Parameter]):
        self.enc_params = enc_params
        self.dec_params = dec_params
        self.params_all = enc_params + dec_params
        self.store = None
        self.have_vis = None
        self.have_iq = None
        self.reset()

    def reset(self):
        self.store: Dict[str, Dict[str, List[Optional[torch.Tensor]] | None]] = {"vision": {"enc": None, "dec": None},
                                                                                 "iq": {"enc": None, "dec": None}}
        self.have_vis = False
        self.have_iq = False

    @torch.no_grad()
    def _flat_cosine(self,
                     ge: list[torch.Tensor | None],
                     gi: list[torch.Tensor | None]) -> float:
        v_list = [t.detach().reshape(-1).float() for t in ge if t is not None]
        u_list = [t.detach().reshape(-1).float() for t in gi if t is not None]
        if not v_list or not u_list:
            return 0.0  # nothing to compare this step
        v = torch.cat(v_list)
        u = torch.cat(u_list)
        return float((v @ u) / (v.norm() * u.norm() + 1e-12))

    def add(self, modality: str, loss: torch.Tensor) -> None:
        """
        Compute grads for one modality and stash them. No optimizer.step here.
        """
        assert modality in ("vision", "iq")
        grads_all = torch.autograd.grad(
            loss, self.params_all,
            retain_graph=False,  # one call only → no need to retain
            create_graph=False,
            allow_unused=True
        )
        G_enc = grads_all[:len(self.enc_params)]
        G_dec = grads_all[len(self.enc_params):]
        self.store[modality]["enc"] = G_enc
        self.store[modality]["dec"] = G_dec
        if modality == "vision":
            self.have_vis = True
        else:
            self.have_iq = True

    def ready(self) -> bool:
        """True once both modalities have been added."""
        return self.have_vis and self.have_iq

    def flush(self) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], bool, float]:
        """
        Returns (encoder_grads_projected_avg, decoder_grads_sum, did_any_projection, preproj_cosine).
        Resets the buffer for the next pair.
        """
        ge = self.store["vision"]["enc"]
        gi = self.store["iq"]["enc"]
        pre_cos = self._flat_cosine(ge, gi)

        ge_proj, p1 = project_conflict(ge, gi)
        gi_proj, p2 = project_conflict(gi, ge)
        did_project = bool(p1 or p2)

        # average encoder grads
        enc_avg = []
        for gv, gi in zip(ge_proj, gi_proj):
            if gv is None and gi is None:
                enc_avg.append(None)
            elif gv is None:
                enc_avg.append(0.5 * gi)
            elif gi is None:
                enc_avg.append(0.5 * gv)
            else:
                enc_avg.append(0.5 * (gv + gi))

        # sum decoder grads
        gd_v = self.store["vision"]["dec"]
        gd_i = self.store["iq"]["dec"]
        dec_sum = []
        for gv, gi in zip(gd_v, gd_i):
            if gv is None and gi is None:
                dec_sum.append(None)
            elif gv is None:
                dec_sum.append(gi)
            elif gi is None:
                dec_sum.append(gv)
            else:
                dec_sum.append(gv + gi)

        # reset for the next pair
        self.reset()
        return enc_avg, dec_sum, did_project, pre_cos
