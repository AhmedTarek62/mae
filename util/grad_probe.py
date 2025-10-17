import torch
import numpy as np


# ----------------- param selection -----------------
def _encoder_param_names(model):
    for n, p in model.named_parameters():
        if p.requires_grad and ('encoder' in n or 'blocks' in n):
            yield n, p


def _pick_slice_params(model):
    # small, stable slice: LN in first encoder block
    for name, module in model.named_modules():
        if ("encoder" in name or "blocks" in name) and (".0." in name):
            if hasattr(module, "weight") and hasattr(module, "bias"):
                sl = [t for t in (module.weight, module.bias) if t is not None and t.requires_grad]
                if sl: return sl
    # fallback: first two trainable tensors
    two = []
    for p in model.parameters():
        if p.requires_grad:
            two.append(p)
            if len(two) == 2:
                break
    return two


# ----------------- low-level math -----------------
def _l2_and_flat(grads, device):
    tot = torch.zeros((), device=device)
    flats = []
    for g in grads:
        if g is None:
            continue
        g = g.float()
        tot = tot + (g * g).sum()
        flats.append(g.reshape(-1))
    norm = torch.sqrt(tot + 1e-12)
    vec = torch.cat(flats) if flats else torch.zeros(1, device=device)
    return norm, vec


def _to_dev_first(x, device):
    if isinstance(x, (list, tuple)): x = x[0]
    return x.to(device, non_blocking=True)


# ----------------- core probes -----------------
def probe_slice(model, batch_vis, batch_iq, device, mr_vis, mr_iq):
    """Fast probe on a tiny, shared slice of params."""
    if batch_vis is None or batch_iq is None:
        return None
    bv = _to_dev_first(batch_vis, device)
    bi = _to_dev_first(batch_iq, device)

    params = _pick_slice_params(model)
    if not params:
        return None

    was_training = model.training
    model.eval()
    with torch.cuda.amp.autocast(enabled=False):
        out_vis = model('vision', bv, mask_ratio=mr_vis)
        L_vis = out_vis[0] if isinstance(out_vis, (list, tuple)) else out_vis
        time_mask = torch.ones((bi.shape[0], bi.shape[-1]), device=bi.device, dtype=torch.bool)
        out_iq = model('iq', bi, time_mask=time_mask, mask_ratio=mr_iq)
        L_iq = out_iq[0] if isinstance(out_iq, (list, tuple)) else out_iq

        Gv = torch.autograd.grad(L_vis, params, retain_graph=True, allow_unused=True)
        gv, vv = _l2_and_flat(Gv, device)
        Gi = torch.autograd.grad(L_iq, params, retain_graph=False, allow_unused=True)
        gi, vi = _l2_and_flat(Gi, device)
        cosine = (vv * vi).sum() / (gv * gi + 1e-12)
        g_ratio = gv / (gi + 1e-12)
    if was_training:
        model.train()
    return {
        "g_vis": float(gv.item()),
        "g_iq": float(gi.item()),
        "g_ratio": float(g_ratio.item()),
        "cosine": float(cosine.item()),
        "cosine_neg": float(cosine.item() < 0.0),
    }


def probe_full_encoder(model, batch_vis, batch_iq, device, mr_vis, mr_iq):
    """Heavier probe across ALL encoder params (streamed norms/cosine)."""
    if batch_vis is None or batch_iq is None:
        return None
    bv = _to_dev_first(batch_vis, device)
    bi = _to_dev_first(batch_iq, device)

    was_training = model.training
    model.eval()
    with torch.amp.autocast("cuda", enabled=False):
        out_vis = model('vision', bv, mask_ratio=mr_vis)
        L_vis = out_vis[0] if isinstance(out_vis, (list, tuple)) else out_vis
        time_mask = torch.ones((bi.shape[0], bi.shape[-1]), device=bi.device, dtype=torch.bool)
        out_iq = model('iq', bi, time_mask=time_mask, mask_ratio=mr_iq)
        L_iq = out_iq[0] if isinstance(out_iq, (list, tuple)) else out_iq

        enc_params = [p for _, p in _encoder_param_names(model)]
        Gv = torch.autograd.grad(L_vis, enc_params, retain_graph=True, allow_unused=True)
        Gi = torch.autograd.grad(L_iq, enc_params, retain_graph=False, allow_unused=True)

        device = bv.device
        gv2 = torch.zeros((), device=device)
        gi2 = torch.zeros((), device=device)
        dot = torch.zeros((), device=device)
        for gv, gi in zip(Gv, Gi):
            if gv is not None:
                gv = gv.float()
                gv2 += (gv * gv).sum()
            if gi is not None:
                gi = gi.float()
                gi2 += (gi * gi).sum()
            if gv is not None and gi is not None:
                dot += (gv * gi).sum()

        gvis = torch.sqrt(gv2 + 1e-12)
        giq = torch.sqrt(gi2 + 1e-12)
        cosine = dot / (gvis * giq + 1e-12)
        g_ratio = gvis / (giq + 1e-12)
    if was_training: model.train()
    return {
        "g_vis": float(gvis.item()),
        "g_iq": float(giq.item()),
        "g_ratio": float(g_ratio.item()),
        "cosine": float(cosine.item()),
        "cosine_neg": float(cosine.item() < 0.0),
    }


# ----------------- batching, summary, logging -----------------
def sample_pairs(dl_vis, dl_iq, K=1):
    """Yield up to K (vis_batch, iq_batch) pairs from loaders."""
    if dl_vis is None or dl_iq is None:
        return []
    out = []
    it_v = iter(dl_vis)
    it_i = iter(dl_iq)
    for _ in range(K):
        try:
            out.append((next(it_v), next(it_i)))
        except StopIteration:
            break
    return out


def summarize(rows):
    if not rows: return None
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0].keys()}
    return {
        "n": int(len(rows)),
        "g_vis_median": float(np.median(arr["g_vis"])),
        "g_iq_median": float(np.median(arr["g_iq"])),
        "g_ratio_median": float(np.median(arr["g_ratio"])),
        "cosine_mean": float(arr["cosine"].mean()),
        "cosine_median": float(np.median(arr["cosine"])),
        "cosine_neg_rate": float((arr["cosine"] < 0.0).mean()),
    }


def log_summary(summary, epoch, log_writer=None, wandb=None, prefix="probe_epoch"):
    if summary is None:
        return
    # TensorBoard
    if log_writer is not None:
        log_writer.add_scalar(f"{prefix}/g_vis_median", summary["g_vis_median"], epoch)
        log_writer.add_scalar(f"{prefix}/g_iq_median", summary["g_iq_median"], epoch)
        log_writer.add_scalar(f"{prefix}/g_ratio_median", summary["g_ratio_median"], epoch)
        log_writer.add_scalar(f"{prefix}/cosine_mean", summary["cosine_mean"], epoch)
        log_writer.add_scalar(f"{prefix}/cosine_median", summary["cosine_median"], epoch)
        log_writer.add_scalar(f"{prefix}/cosine_neg_rate", summary["cosine_neg_rate"], epoch)
    # W&B
    if wandb is not None:
        payload = {"epoch": epoch, f"{prefix}/n": summary["n"]}
        for k, v in summary.items():
            payload[f"{prefix}/{k}"] = v
        wandb.log(payload)
