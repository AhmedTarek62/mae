import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.pc_grad import shared_encoder_params, decoder_adapter_params, PCGradPairBuffer, assign_grads, grad_norm


class RoundRobin:
    """
    Alternates batches 1:1 between two loaders and exposes __len__ for loggers.
    If one loader is shorter, it auto-resets that loader’s iterator as needed.
    One epoch covers roughly len(loader_vis) + len(loader_iq) steps.
    """
    def __init__(self, loader_vis, loader_iq):
        self.loader_vis = loader_vis
        self.loader_iq = loader_iq
        self._len = max(len(loader_vis), 0) + max(len(loader_iq), 0)

    def __len__(self):
        return self._len

    def __iter__(self):
        # handle degenerate cases
        if self._len == 0:
            return
        it_v = iter(self.loader_vis) if len(self.loader_vis) > 0 else None
        it_iq = iter(self.loader_iq) if len(self.loader_iq) > 0 else None

        steps = self._len
        i = 0
        while i < steps:
            # vision step
            if it_v is not None:
                try:
                    batch = next(it_v)
                except StopIteration:
                    it_v = iter(self.loader_vis)
                    batch = next(it_v)
                yield "vision", batch
                i += 1
                if i >= steps:
                    break

            # iq step
            if it_iq is not None:
                try:
                    batch = next(it_iq)
                except StopIteration:
                    it_iq = iter(self.loader_iq)
                    batch = next(it_iq)
                yield "iq", batch
                i += 1
                if i >= steps:
                    break


def train_one_epoch_multi(model: torch.nn.Module,
                          loader_vis: Iterable,
                          loader_iq: Iterable,
                          optimizer: torch.optim.Optimizer,
                          device: torch.device,
                          epoch: int,
                          loss_scaler,
                          log_writer=None,
                          mask_ratio_vis: float = 0.75,
                          mask_ratio_iq: float = 0.10,
                          accum_iter: int = 1,
                          args=None):
    """
    Alternating batches (1:1) between vision and IQ.
    Vision batch: (imgs, label) -> imgs
    IQ batch: (x_pad, time_mask, lengths)
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    wandb = None
    if args.use_wandb:
        import wandb as _wandb
        wandb = _wandb

    header = f'Epoch: [{epoch}]'
    print_freq = 20

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir:', log_writer.log_dir)

    # Combined steps ~ sum of two loader lengths (roughly)
    rr = RoundRobin(loader_vis, loader_iq)
    total_steps = len(rr)

    use_pcgrad = args.use_pcgrad
    if use_pcgrad:
        assert accum_iter == 2, "PCGrad path assumes accum_iter=2 with round-robin."
        enc_params = shared_encoder_params(model)
        dec_params = decoder_adapter_params(model)
        buf = PCGradPairBuffer(enc_params, dec_params)
    else:
        enc_params, dec_params, buf = None, None, None

    step_count = 0
    update_steps = 0
    updates_per_epoch = len(rr) // accum_iter
    for data_iter_step, (modality, batch) in enumerate(
            metric_logger.log_every(rr, print_freq, header)):

        # stop after we covered ~one pass over both loaders
        step_count += 1
        if step_count > total_steps:
            break

        if modality == "vision":
            imgs, _ = batch
            imgs = imgs.to(device, non_blocking=True)
        else:
            if len(batch) == 3:
                x_pad, time_mask, lengths = batch
            else:
                x_pad = batch
                time_mask = torch.ones((x_pad.shape[0], x_pad.shape[-1]), device=x_pad.device, dtype=torch.bool)
            x_pad = x_pad.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)

        # per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / total_steps + epoch, args)

        with torch.amp.autocast("cuda"):
            if modality == "vision":
                loss, _, _, _ = model.forward('vision', imgs, mask_ratio=mask_ratio_vis)
            else:
                loss, _, _, _ = model.forward('iq', x_pad, mask_ratio=mask_ratio_iq, time_mask=time_mask)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if use_pcgrad:
            # collect grads for this step (no backward/step yet)
            buf.add(modality, loss)
            if buf.ready():
                enc_g, dec_g, did_proj, pre_cos = buf.flush()

                # write grads to .grad (encoder gets projected avg, decoders the sum)
                assign_grads(enc_params, enc_g, accumulate=False)
                assign_grads(dec_params, dec_g, accumulate=False)

                total_pre = grad_norm(list(model.parameters()))
                enc_pre = grad_norm(enc_params)
                dec_pre = grad_norm(dec_params)

                global_step = epoch * updates_per_epoch + update_steps
                if wandb is not None and misc.is_main_process():
                    wandb.log({
                        "pretrain/grad/total_preclip": total_pre,
                        "pretrain/grad/enc_preclip": enc_pre,
                        "pretrain/grad/dec_preclip": dec_pre,
                        "pretrain/pcg/pre_cos": pre_cos,
                        "pretrain/pcg/did_proj": int(did_proj)},
                        step=global_step)

                # optional: clip
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.grad is not None],
                        max_norm=args.clip_grad
                    )
                    if wandb is not None and misc.is_main_process():
                        wandb.log({
                            "pretrain/grad/total_postclip": grad_norm(list(model.parameters()))
                        }, step=global_step)

                optimizer.step()
                optimizer.zero_grad()
                update_steps += 1
        else:
            loss = loss / accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=((data_iter_step + 1) % accum_iter == 0))
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / total_steps + epoch) * 1000)
            # track modality-specific losses if you’d like; here we log a single stream
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_vision(data_loader: Iterable, model: torch.nn.Module,
                    device: torch.device, mask_ratio: float = 0.75):
    """
    Validation loop for MAE pre-training on spectrogram images.
    Expects data_loader to yield (imgs, _):
      imgs: (N, C, H, W)
    Returns dict of averaged metrics.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Val(Vis):"

    model.eval()
    total_masked = 0
    total_patches = 0

    for imgs, _ in metric_logger.log_every(data_loader, 10, header):
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            # forward returns: loss, pred_tokens, mae_mask, extras
            loss, pred, mae_mask, _ = model.forward('vision', imgs, mask_ratio=mask_ratio)

        loss_val = float(loss.item())
        metric_logger.update(loss=loss_val)

        # bookkeeping: how many patches masked this step
        total_masked += (mae_mask > 0).sum().item()
        total_patches += mae_mask.numel()

    # sync across processes (if DDP)
    metric_logger.synchronize_between_processes()

    masked_pct = (100.0 * total_masked / total_patches) if total_patches > 0 else 0.0
    print(f"* Val(Vis) loss {metric_logger.loss.global_avg:.4f}  masked% {masked_pct:.1f}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["masked_pct"] = masked_pct
    return stats


@torch.no_grad()
def evaluate_iq(data_loader: Iterable, model: torch.nn.Module,
                device: torch.device, mask_ratio: float = 0.75):
    """
    Optional IQ-only validation (re-uses your IQ engine’s idea).
    Expects data_loader to yield (x_pad, time_mask, lengths).
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Val(IQ):"

    model.eval()
    total_masked_real = 0
    total_real = 0

    for batch in metric_logger.log_every(data_loader, 10, header):
        if len(batch) == 3:
            x_pad, time_mask, lengths = batch
        else:
            x_pad = batch
            time_mask = torch.ones((x_pad.shape[0], x_pad.shape[-1]), device=x_pad.device, dtype=torch.bool)
        x_pad = x_pad.to(device, non_blocking=True)
        time_mask = time_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            loss, pred, mae_mask, extras = model.forward('iq', x_pad, mask_ratio=mask_ratio, time_mask=time_mask)

        token_mask = extras["token_mask"]
        loss_val = float(loss.item())
        metric_logger.update(loss=loss_val)

        masked_real = ((mae_mask > 0) & token_mask).sum().item()
        real_tokens = token_mask.sum().item()
        total_masked_real += masked_real
        total_real += real_tokens

    metric_logger.synchronize_between_processes()
    masked_real_pct = (100.0 * total_masked_real / total_real) if total_real > 0 else 0.0
    print(f"* Val(IQ) loss {metric_logger.loss.global_avg:.4f}  masked_real% {masked_real_pct:.1f}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["masked_real_pct"] = masked_real_pct
    return stats
