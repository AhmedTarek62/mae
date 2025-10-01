# viz_masking_iq_multi.py
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import islice

from torch.utils.data import DataLoader, SequentialSampler
from dataset_classes.iq_dataset import IQDataset, pad_collate, IQResampler
import models_mae_iq  # your model factory fns


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--ckpt", default="", type=str)
    p.add_argument("--model", default="mae_vit_iq_micro", type=str)
    p.add_argument("--segment_len", default=4096, type=int)
    p.add_argument("--hop", default=4096, type=int)
    p.add_argument("--segment_duration_ms", default=1.0, type=float)
    p.add_argument("--mask_ratio", default=0.1, type=float)
    p.add_argument("--antenna", default=0, type=int)  # which antenna to plot
    p.add_argument("--num", default=10, type=int)  # how many samples to plot
    p.add_argument("--save_dir", default="", type=str)
    p.add_argument("--downsample_plot", default=100, type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / loader
    resampler = IQResampler(
        segment_len=args.segment_len,
        segment_duration=args.segment_duration_ms / 1000.0,
    )
    ds = IQDataset(args.data_dir, resampler=resampler, stats="stats.json")
    dl = DataLoader(
        ds, sampler=SequentialSampler(ds), batch_size=1,
        collate_fn=pad_collate, drop_last=False, num_workers=0, pin_memory=False
    )

    # model
    ctor = models_mae_iq.__dict__[args.model]
    model = ctor(segment_len=args.segment_len, hop=args.hop).to(device).eval()

    if args.ckpt and Path(args.ckpt).is_file():
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.ckpt}. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("no checkpoint provided; using randomly initialized weights.")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    fs_int = resampler.internal_rate_hz
    ds_factor = max(1, int(args.downsample_plot))

    for idx, (x_pad, time_mask, lengths) in enumerate(islice(dl, args.num)):
        x_pad = x_pad.to(device)  # (1, 2, C, T)
        time_mask = time_mask.to(device)

        # forward
        loss, pred, mae_mask, token_mask = model(x_pad, time_mask, mask_ratio=args.mask_ratio)
        loss_val = float(loss.item())
        pred = pred.cpu()
        mae_mask = (mae_mask.cpu() > 0)
        token_mask = token_mask.cpu()

        # segments for targets/masks
        x_seg, seg_mask_in = model.segment(x_pad, pad_tail=True)
        x_seg = x_seg.cpu()  # (1, s, 2, C, m)
        seg_mask_in = seg_mask_in.cpu()
        s = x_seg.shape[1]
        m = model.segment_len
        C = x_seg.shape[3]
        L = pred.shape[1]

        # align if encoder truncated to respect max_tokens
        if L < s * C:
            s = L // C
            x_seg = x_seg[:, :s]
            seg_mask_in = seg_mask_in[:, :s]

        # reshape predictions back to (1, s, 2, C, m)
        pred_seg = pred.view(1, s, C, 2, m).permute(0, 1, 3, 2, 4).contiguous()

        # masked input by zeroing masked tokens
        masked_real = (mae_mask & token_mask)  # (1, L)
        masked_real_map = masked_real.view(1, s, C)  # (1, s, C)
        keep_map = (~masked_real).view(1, s, C).to(x_seg.dtype).unsqueeze(2).unsqueeze(-1)
        x_seg_masked = x_seg * keep_map

        # back to time domain (original, masked, reconstruction)
        x_rec = model.unsegment(pred_seg, seg_mask_in).cpu()  # (1, 2, C, T_rec)
        x_orig = model.unsegment(x_seg, seg_mask_in).cpu()
        x_masked = model.unsegment(x_seg_masked, seg_mask_in).cpu()
        T = x_rec.shape[-1]

        t = np.arange(T) / fs_int
        hop = model.hop
        starts = np.arange(s) * hop
        ends = starts + m

        # choose antenna to visualize (clamp)
        ant_idx = min(max(args.antenna, 0), C - 1)
        masked_tokens_idx = masked_real_map[0, :, ant_idx].nonzero(as_tuple=True)[0].cpu().numpy()

        # plot (I and Q)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        titles = ["I (real)", "Q (imag)"]
        for comp in range(2):
            ax = axes[comp]
            # original
            ax.plot(t[::ds_factor], x_orig[0, comp, ant_idx, ::ds_factor].numpy(),
                    linewidth=1.0, label="original")
            # masked input (show gaps)
            xi = x_masked[0, comp, ant_idx, :].numpy().copy()
            for si in masked_tokens_idx:
                t0, t1 = starts[si], min(ends[si], T)
                xi[t0:t1] = np.nan
            ax.plot(t, xi, linewidth=1.0, linestyle="--", label="masked input")
            # reconstruction
            ax.plot(t[::ds_factor], x_rec[0, comp, ant_idx, ::ds_factor].numpy(),
                    linewidth=1.0, linestyle=":", label="reconstruction")

            # shade masked spans
            for si in masked_tokens_idx:
                t0 = starts[si] / fs_int
                t1 = min(ends[si], T) / fs_int
                ax.axvspan(t0, t1, alpha=0.08)

            ax.set_ylabel(titles[comp])
            ax.grid(True)

        axes[-1].set_xlabel("time (s)")
        axes[0].legend(loc="upper right", ncol=3)
        fig.suptitle(f"Sample {idx} — antenna {ant_idx} — loss {loss_val:.4f}")
        plt.tight_layout()

        if save_dir:
            out_path = save_dir / f"viz_sample_{idx:03d}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"saved {out_path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
