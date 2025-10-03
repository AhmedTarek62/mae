import math
import sys
from typing import Iterable, Optional, Tuple

import torch
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def _unpack_batch(batch):
    """Accept (x,y) or (x,time_mask,y). Return (x, time_mask|None, y)."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            return x, None, y
        elif len(batch) == 3:
            x, time_mask, y = batch
            return x, time_mask, y
    raise ValueError("Batch must be (x,y) or (x,time_mask,y).")


def _is_classification(targets: torch.Tensor, outputs: torch.Tensor) -> bool:
    # Heuristic: integer (Long) labels and 2D logits imply classification
    return targets.dtype in (torch.int64, torch.long) and outputs.ndim == 2 and outputs.shape[1] >= 2


def _is_vision_tensor(x: torch.Tensor) -> bool:
    """Heuristic to detect vision tensors vs IQ.
    Vision: (N,C,H,W) with C != 2. IQ: (N,2,C,T).
    """
    return x.dim() == 4 and x.shape[1] != 2


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir:', log_writer.log_dir)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, time_mask, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time_mask = None if time_mask is None else time_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # Our fine-tuner accepts (x, time_mask|None)
            outputs = model(samples, time_mask)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Non-finite loss: {loss_value}. Exiting.")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        # Optional metrics (classification or regression)
        with torch.no_grad():
            if _is_classification(targets, outputs):
                acc1, acc3 = accuracy(outputs, targets, topk=(1, min(3, outputs.shape[1])))
                metric_logger.meters.setdefault('acc1', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
                metric_logger.meters['acc1'].update(acc1.item(), n=samples.size(0))
                if outputs.shape[1] >= 3:
                    metric_logger.meters.setdefault('acc3', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
                    metric_logger.meters['acc3'].update(acc3.item(), n=samples.size(0))
            else:
                # regression: report MAE
                mae = (outputs.squeeze() - targets.squeeze()).abs().mean().item()
                metric_logger.meters.setdefault('mae', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
                metric_logger.meters['mae'].update(mae, n=samples.size(0))

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', misc.all_reduce_mean(loss_value), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader: Iterable, model: torch.nn.Module, criterion: torch.nn.Module, device: torch.device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    # For per-class accuracy if classification
    per_class_correct = None
    per_class_total = None
    is_cls_mode = None

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, time_mask, targets = _unpack_batch(batch)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time_mask = None if time_mask is None else time_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples, time_mask)
            loss = criterion(outputs, targets)

        loss_val = loss.item()
        metric_logger.update(loss=loss_val)

        if is_cls_mode is None:
            is_cls_mode = _is_classification(targets, outputs)

        if is_cls_mode:
            acc1, acc3 = accuracy(outputs, targets, topk=(1, min(3, outputs.shape[1])))
            metric_logger.meters.setdefault('acc1', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
            metric_logger.meters['acc1'].update(acc1.item(), n=samples.size(0))
            if outputs.shape[1] >= 3:
                metric_logger.meters.setdefault('acc3', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))
                metric_logger.meters['acc3'].update(acc3.item(), n=samples.size(0))

            # per-class tracking
            if per_class_correct is None:
                num_classes = outputs.shape[1]
                per_class_correct = torch.zeros(num_classes, device=device)
                per_class_total = torch.zeros(num_classes, device=device)
            _, pred = outputs.max(1)
            for i in range(samples.size(0)):
                lbl = targets[i]
                per_class_total[lbl] += 1
                if pred[i] == lbl:
                    per_class_correct[lbl] += 1
        else:
            # regression metrics
            err = (outputs.squeeze() - targets.squeeze())
            mae = err.abs().mean().item()
            rmse = (err.pow(2).mean().sqrt().item())
            metric_logger.meters.setdefault('mae', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            metric_logger.meters.setdefault('rmse', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            metric_logger.meters['mae'].update(mae, n=samples.size(0))
            metric_logger.meters['rmse'].update(rmse, n=samples.size(0))

    metric_logger.synchronize_between_processes()

    if is_cls_mode:
        per_class_acc = torch.where(
            per_class_total > 0,
            per_class_correct / per_class_total * 100,
            torch.zeros_like(per_class_total)
        ).tolist()
        mean_per_class_acc = sum(per_class_acc) / len(per_class_acc) if len(per_class_acc) > 0 else 0.0
        print('* Mean per-class acc: {m:.3f}  Acc@1 {top1.global_avg:.3f}  '
              'Acc@3 {top3.global_avg:.3f}  Loss {losses.global_avg:.3f}'
              .format(m=mean_per_class_acc,
                      top1=metric_logger.acc1 if 'acc1' in metric_logger.meters else 0.0,
                      top3=metric_logger.acc3 if 'acc3' in metric_logger.meters else 0.0,
                      losses=metric_logger.loss))
        out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        out['pca'] = mean_per_class_acc
        return out
    else:
        print('* MAE {mae.global_avg:.4f}  RMSE {rmse.global_avg:.4f}  Loss {loss.global_avg:.4f}'
              .format(mae=metric_logger.mae, rmse=metric_logger.rmse, loss=metric_logger.loss))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
