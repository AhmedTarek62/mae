import os
import h5py
import glob
from torch import nn
import torch.nn.functional as F
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class IQResampler(nn.Module):
    def __init__(self,
                 segment_len: int,
                 segment_duration: float,
                 antialias: bool = True,
                 lpf_taps: int = 63,
                 lpf_beta: float = 8.0,
                 cutoff_margin: float = 0.95):
        super().__init__()
        self.target_len = int(segment_len)              # K
        self.segment_duration = float(segment_duration)     # seconds
        self.internal_rate_hz = self.target_len / self.segment_duration

        self.antialias = bool(antialias)
        self.lpf_taps = int(lpf_taps)
        self.lpf_beta = float(lpf_beta)
        self.cutoff_margin = float(cutoff_margin)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, fs_hz: float) -> torch.Tensor:
        """
        x: (2, C, T_in)  ->  (2, C, T_out), where T_out ≈ T_in * (f_int / fs_hz)
        """
        assert x.ndim == 3 and x.shape[0] == 2, "expected (2, C, T)"
        num_ant = x.shape[1]
        time_len_in = x.shape[2]
        assert time_len_in >= 1

        rate_ratio = float(self.internal_rate_hz) / float(fs_hz)  # f_out / f_in
        time_len_out = max(1, int(round(time_len_in * rate_ratio)))

        if time_len_out == time_len_in:
            return x.clone()

        # reshape to (1, 2*C, T) so we can use depthwise conv and interpolate
        streams = x.permute(1, 0, 2).reshape(1, 2 * num_ant, time_len_in)  # (1, 2C, T)

        # anti-alias when downsampling
        if self.antialias and rate_ratio < 1.0:
            streams = self._lowpass_depthwise(streams, rate_ratio)

        # linear interpolation along time
        streams = F.interpolate(streams, size=time_len_out, mode="linear", align_corners=False)  # (1, 2C, T_out)

        # back to (2, C, T_out)
        out = streams.view(num_ant, 2, time_len_out).permute(1, 0, 2).contiguous()
        return out

    # ---- helpers ----

    def _lowpass_depthwise(self, streams: torch.Tensor, rate_ratio: float) -> torch.Tensor:
        """
        Depthwise FIR LPF over time, same kernel for every I/Q channel.
        streams: (1, 2C, T_in)
        """
        _, num_ch, time_len_in = streams.shape
        num_taps = self._choose_num_taps(time_len_in)

        cutoff_norm = 0.5 * rate_ratio * self.cutoff_margin  # normalized to input Nyquist
        if cutoff_norm <= 0.0:
            return streams

        device, dtype = streams.device, streams.dtype
        n = torch.arange(num_taps, device=device, dtype=dtype) - (num_taps - 1) / 2
        # torch.sinc is sin(pi x)/(pi x); scale argument by 2 * cutoff_norm
        impulse = 2 * cutoff_norm * torch.sinc(2 * cutoff_norm * n)
        window = torch.kaiser_window(num_taps, beta=self.lpf_beta, periodic=False, dtype=dtype, device=device)
        impulse = (impulse * window)
        impulse = impulse / (impulse.sum() + 1e-12)

        pad_len = num_taps // 2
        weight = impulse.view(1, 1, num_taps).expand(num_ch, 1, num_taps)  # (2C, 1, taps)
        return F.conv1d(F.pad(streams, (pad_len, pad_len)), weight, bias=None, stride=1, padding=0, groups=num_ch)

    def _choose_num_taps(self, time_len_in: int) -> int:
        """Pick an odd tap count not exceeding time_len_in (and at least 3)."""
        num_taps = self.lpf_taps if (self.lpf_taps % 2 == 1) else (self.lpf_taps - 1)
        num_taps = min(num_taps, time_len_in if time_len_in % 2 == 1 else time_len_in - 1)
        return max(3, num_taps)


class IQDataset(Dataset):
    def __init__(self, root_dir, manifest="manifest.jsonl",
                 stats=None,                   # dict path or dict; None to skip
                 resampler=None,               # instance of IQResampler or None
                 channel_first=True):          # we keep (2, C, T)
        self.root = Path(root_dir)
        # read manifest (expects {"file": "...", "fs": ...} per line)
        with open(self.root / manifest, "r") as f:
            self.entries = [json.loads(l) for l in f]
        # optional stats
        if isinstance(stats, (str, Path)):
            self.stats = json.load(open(self.root / stats, "r"))
        else:
            self.stats = stats
        self.resampler = resampler
        self.channel_first = bool(channel_first)
        self.dtype = torch.float32

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _load_iq(path: Path) -> np.ndarray:
        """
        Accepts npy saved as:
          (T, 2)             -> returns (2, 1, T)
          (C, T, 2)          -> returns (2, C, T)
          already (2, C, T)  -> returns (2, C, T)
        """
        a = np.load(path, mmap_mode="r")
        if a.ndim == 2 and a.shape[-1] == 2:                 # (T, 2)
            a = a.astype(np.float32, copy=False).transpose(1, 0)[..., None]  # (2, T) -> (2, T, 1)
            a = np.ascontiguousarray(a[:, None, :, 0]).astype(np.float32)    # (2, 1, T)
        elif a.ndim == 3 and a.shape[-1] == 2:               # (C, T, 2)
            a = a.astype(np.float32, copy=False).transpose(2, 0, 1)          # (2, C, T)
        elif a.ndim == 3 and a.shape[0] == 2:                # (2, C, T)
            a = a.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported IQ shape in {path}: {a.shape}")
        return a

    def __getitem__(self, idx: int):
        ent = self.entries[idx]
        path = self.root / ent["npy"]
        fs_hz = float(ent["fs"])

        # load -> torch (2, C, T)
        x_np = self._load_iq(path)
        x = torch.from_numpy(x_np).to(self.dtype)

        # dataset-side resampling to internal rate (if provided)
        if self.resampler is not None:
            x = self.resampler(x, fs_hz=fs_hz)  # (2, C, T')

        # optional standardization: per I/Q across all antennas & time
        if self.stats is not None:
            mu = torch.tensor(self.stats["mean"], dtype=self.dtype)     # shape (2,)
            sd = torch.tensor(self.stats["std"], dtype=self.dtype).clamp_min(1e-6)  # shape (2,)
            # x: (2, C, T) → broadcast over (C, T)
            x = (x - mu.view(2, 1, 1)) / sd.view(2, 1, 1)

        # return (2, C, T) (channel_first=True is the only layout we use now)
        return x


class IQDatasetH5(Dataset):
    def __init__(self, h5_path, resampler=None, stats=None, mode='pretrain'):
        """
        h5 structure:
          iq_data: (N, T, 2) or (N, C, T, 2) or (N, 2, C, T)
          modulation: (N, ...)   # class id or vector
          angles: (N, ...)       # regression targets
        stats: None or {"mean":[mI, mQ], "std":[sI, sQ]}
        """
        self.h5_path = str(h5_path)
        self.resampler = resampler
        self.stats = stats if stats is not None else {"mean": (0.0, 0.0), "std": (0.3396, 0.3525)}
        assert mode.lower() in ["pretrain", "aoa", "amc"]
        self.mode = mode.lower()
        if self.mode == "amc":
            self.labels = ('bpsk', 'cw', 'pam4', 'qam', 'qam64', 'qpsk', 'sine')

        # probe once for length + quick validation
        with h5py.File(self.h5_path, "r") as f:
            if not {"iq_data", "modulation", "angles"}.issubset(f.keys()):
                raise KeyError("Expected keys: 'iq_data', 'modulation', 'angles'")
            # self.iq_data = f["iq_data"][:]
            # self.modulations = f["modulation"][:]
            # self.angles = f["angles"][:]
            self.N = f["iq_data"].shape[0]
            if f["modulation"].shape[0] != self.N or f["angles"].shape[0] != self.N:
                raise ValueError("Lengths of iq_data/modulation/angles don't match")

    def __len__(self):
        return self.N

    @staticmethod
    def _to_2ct(a: np.ndarray) -> np.ndarray:
        # per-sample shapes -> (2, C, T) float32
        if a.ndim == 2 and a.shape[-1] == 2:         # (T, 2)
            a = a.astype(np.float32, copy=False).T[:, None, :]         # (2,1,T)
        elif a.ndim == 3 and a.shape[-1] == 2:       # (C, T, 2)
            a = a.astype(np.float32, copy=False).transpose(2, 0, 1)    # (2,C,T)
        elif a.ndim == 3 and a.shape[0] == 2:        # (2, C, T)
            a = a.astype(np.float32, copy=False)
        elif a.ndim == 3 and a.shape[1] == 2:
            a = a.astype(np.float32, copy=False).transpose(1, 0, 2)
        else:
            raise ValueError(f"Unsupported IQ item shape {a.shape}")
        return a

    def __getitem__(self, idx):
        # open on demand (simple + lazy)
        with h5py.File(self.h5_path, "r") as f:
            x_np = f["iq_data"][idx]       # one sample
            y_mod = f["modulation"][idx]
            y_ang = f["angles"][idx]

        x_np = self._to_2ct(x_np)
        x = torch.from_numpy(x_np)         # (2, C, T) float32

        if self.resampler is not None:
            x = self.resampler(x, fs_hz=40e6)
        if self.stats is not None:
            mu = torch.tensor(self.stats["mean"], dtype=torch.float32).view(2, 1, 1)
            sd = torch.tensor(self.stats["std"],  dtype=torch.float32).clamp_min(1e-6).view(2, 1, 1)
            x = (x - mu) / sd

        if self.mode == 'pretrain':
            return x
        elif self.mode == 'aoa':
            az_deg, el_deg = y_ang
            el = np.deg2rad(el_deg)
            az = np.deg2rad(az_deg)
            x_ang = np.cos(el) * np.cos(az)
            y_ang = np.cos(el) * np.sin(az)
            z_ang = np.sin(el)
            return x, torch.as_tensor([x_ang, y_ang, z_ang], dtype=torch.float32)
        elif self.mode == 'amc':
            return x, torch.tensor(self.labels.index(y_mod.decode('utf-8')), dtype=torch.long)


class IQDatasetH5Sharded(Dataset):
    def __init__(self, shard_paths, resampler=None, stats=None, mode='pretrain'):
        if isinstance(shard_paths, str):
            if os.path.isdir(shard_paths):
                shard_paths = sorted(glob.glob(os.path.join(shard_paths, "*.h5")))
            else:
                shard_paths = sorted(glob.glob(shard_paths))
        assert isinstance(shard_paths, (list, tuple)) and len(shard_paths) > 0
        self.paths = list(shard_paths)
        self.resampler = resampler
        self.stats = stats if stats is not None else {"mean": (0.0, 0.0), "std": (0.3396, 0.3525)}
        self.mode = mode.lower()
        if self.mode == "amc":
            self.labels = ('bpsk','cw','pam4','qam','qam64','qpsk','sine')

        # build global index
        self.index = []  # (path_idx, local_idx)
        self._sizes = []
        for pi, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                n = f["iq_data"].shape[0]
            self._sizes.append(n)
            self.index.extend([(pi, i) for i in range(n)])

        # lazy-open per worker / per current shard
        self._h5 = None
        self._cur_path = None
        self._iq = self._mod = self._ang = None

    def __len__(self):
        return len(self.index)

    def _open_if_needed(self, path):
        if self._cur_path != path:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception: pass
            # big raw chunk cache, SWMR on (read-only)
            self._h5 = h5py.File(path, "r", libver="latest", swmr=True,
                                 rdcc_nbytes=64*1024*1024, rdcc_nslots=1_000_003, rdcc_w0=0.25)
            self._iq = self._h5["iq_data"]
            self._mod = self._h5["modulation"]
            self._ang = self._h5["angles"]
            self._cur_path = path

    def __getitem__(self, gidx):
        pi, li = self.index[gidx]
        path = self.paths[pi]
        self._open_if_needed(path)

        x_np = self._iq[li]                  # already (2,C,T) float32 from the sharder
        x = torch.from_numpy(x_np)

        if self.resampler is not None:
            x = self.resampler(x, fs_hz=40e6)
        if self.stats is not None:
            mu = torch.tensor(self.stats["mean"], dtype=torch.float32).view(2, 1, 1)
            sd = torch.tensor(self.stats["std"],  dtype=torch.float32).clamp_min(1e-6).view(2, 1, 1)
            x = (x - mu) / sd

        if self.mode == 'pretrain':
            return x
        elif self.mode == 'aoa':
            az_deg, el_deg = self._ang[li]
            el, az = np.deg2rad(el_deg), np.deg2rad(az_deg)
            target = torch.tensor([np.cos(el)*np.cos(az),
                                   np.cos(el)*np.sin(az),
                                   np.sin(el)], dtype=torch.float32)
            return x, target
        elif self.mode == 'amc':
            y_mod = self._mod[li]
            if isinstance(y_mod, (bytes, np.bytes_, np.str_)):
                y_mod = y_mod.decode('utf-8') if isinstance(y_mod, (bytes, np.bytes_)) else str(y_mod)
                cls = self.labels.index(y_mod)
            else:
                cls = int(y_mod)
            return x, torch.tensor(cls, dtype=torch.long)


def pad_collate(batch):
    """
    Batch items shaped (2, C, T) by padding only the time axis.
    Returns:
      x_pad:    (N, 2, C, T_max)
      time_mask:(N, T_max)  True where real samples exist
      lengths:  list[int] original T per item
    """
    if not batch:
        raise ValueError("empty batch")

    b0 = batch[0]
    if b0.ndim != 3 or b0.shape[0] != 2:
        raise ValueError(f"expected items shaped (2, C, T); got {b0.shape}")

    c_ref = b0.shape[1]
    lengths = [x.shape[2] for x in batch]
    # validate same C across batch
    for i, x in enumerate(batch):
        if x.shape[1] != c_ref:
            raise ValueError(f"antenna count differs in batch: item 0 has C={c_ref}, item {i} has C={x.shape[1]}")

    t_max = max(lengths)
    dtype = b0.dtype
    device = b0.device  # usually cpu in collate

    x_pad = torch.zeros((len(batch), 2, c_ref, t_max), dtype=dtype, device=device)
    time_mask = torch.zeros((len(batch), t_max), dtype=torch.bool, device=device)

    for i, x in enumerate(batch):
        t = x.shape[2]
        x_pad[i, :, :, :t] = x
        time_mask[i, :t] = True

    return x_pad, time_mask, lengths


def compute_resampled_stats(root_dir, manifest="manifest.jsonl", resampler=None, dtype=torch.float32):
    root = Path(root_dir)
    with open(root / manifest, "r") as f:
        entries = [json.loads(l) for l in f]

    sum_iq = torch.zeros(2, dtype=torch.float64)
    sumsq_iq = torch.zeros(2, dtype=torch.float64)
    count_iq = torch.zeros(2, dtype=torch.float64)

    for ent in entries:
        x_np = IQDataset._load_iq(root / ent["npy"])           # (2, C, T)
        x = torch.from_numpy(x_np).to(dtype)
        if resampler is not None:
            x = resampler(x, fs_hz=float(ent["fs"]))               # (2, C, T')
        # accumulate over (C, T')
        # sums: (2,)
        sum_iq += x.sum(dim=(1, 2), dtype=torch.float64)
        sumsq_iq += (x * x).sum(dim=(1, 2), dtype=torch.float64)
        count_iq += torch.tensor([x.shape[1] * x.shape[2]] * 2, dtype=torch.float64)

    mean = (sum_iq / count_iq).to(torch.float32)
    var = (sumsq_iq / count_iq - mean.double()**2).clamp_min(0.0).to(torch.float32)
    std = torch.sqrt(var + 1e-12)

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats

