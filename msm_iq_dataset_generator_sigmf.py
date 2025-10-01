import os, json
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from sigmf import sigmffile
import hashlib


# -------- Helpers --------
def sample_uid(src_meta: str, start: int, T: int, fs: float) -> str:
    # 8-byte (16 hex) BLAKE2s; adjust digest_size if you want longer ids
    h = hashlib.blake2s(f"{src_meta}|{start}|{T}|{fs}".encode(),
                        digest_size=8).hexdigest()
    return h  # e.g., "9f2a1c0b5e7d3a10"


def read_sigmf(path: str):
    rec = sigmffile.fromfile(path)
    fs = float(rec.get_global_info()[rec.SAMPLE_RATE_KEY])
    data_path = str(Path(path)) + "-data"
    probe = rec.read_samples(start_index=0, count=1)  # 1 sample
    bytes_per_sample = probe.dtype.itemsize
    total = os.path.getsize(data_path) // bytes_per_sample
    return rec, fs, int(total)


def to_f32_2ch(arr) -> np.ndarray:
    """
    Return (T, 2) float32 array: [:,0]=I, [:,1]=Q, C-contiguous.
    Works for native complex, structured IQ, or interleaved floats.
    """
    a = np.asarray(arr)

    if np.iscomplexobj(a):
        out = np.empty((a.shape[0], 2), dtype=np.float32)
        out[:, 0] = a.real.astype(np.float32, copy=False)
        out[:, 1] = a.imag.astype(np.float32, copy=False)
        return np.ascontiguousarray(out)

    # For structured dtypes like [('f0','<f4'),('f1','<f4')] or interleaved floats
    v = a.view(np.float32).reshape(-1, 2)  # respects memory order (I then Q)
    return np.ascontiguousarray(v)


def main(recordings_dirs, sequence_duration, max_instances_per_file=0, coverage=1.0, seed=42):
    d0, d1 = map(float, sequence_duration)  # seconds
    assert 0 < d0 <= d1

    rng = np.random.default_rng(seed)

    out_root = Path("../datasets")
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_name = (f"iq_dataset_{int(d0*1000)}ms_{int(d1*1000)}ms_{stamp}"
               if d0 != d1 else f"iq_dataset_{int(d0*1000)}ms_{stamp}")
    out_dir = out_root / ds_name
    out_dir.mkdir(exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    mf = open(manifest_path, "w", encoding="utf-8")

    # collect file_name files
    rec_files = []
    for d in recordings_dirs:
        for p in Path(d).iterdir():
            if ".sigmf" in str(p) and not str(p).endswith("-file_name"):
                rec_files.append(str(p)[:-5])

    with tqdm(rec_files, desc="Generating Data", unit="file") as pbar:
        for file_idx, file_name in enumerate(pbar):
            prefix = Path(file_name).stem
            rec, fs, total = read_sigmf(file_name)

            # expected valid starts using mean duration -> used to size the sampling budget
            mean_T = int(round(fs * 0.5 * (d0 + d1)))
            expected_base = max(0, total - mean_T) + 1

            # downweight by name
            frac = 1.0
            name_low = prefix.lower()
            if "analog" in name_low:
                frac = 0.05
            elif "ism" in name_low:
                frac = 0.5

            # instance budget for this file (cap optional)
            num_instances = int(total / mean_T * frac * coverage)
            if max_instances_per_file and max_instances_per_file > 0:
                num_instances = min(num_instances, max_instances_per_file)

            for j in range(num_instances):
                # sample a fresh duration per instance
                t = rng.uniform(d0, d1)               # seconds
                t_samples = int(round(t * fs))                # samples
                if t_samples <= 0 or total < t_samples:
                    continue

                # pick a random valid start for this t_samples
                start = int(rng.integers(0, total - t_samples + 1))

                # read, convert and validate
                raw = rec.read_samples(start_index=start, count=t_samples)
                iq = to_f32_2ch(raw)
                if iq.shape[0] != t_samples:
                    continue

                # save each instance as its own .npy
                uid = sample_uid(Path(file_name).name, start, t_samples, fs)

                fname = f"{uid}.npy"  # opaque, short
                rel_path = f"{fname}"  # store this in manifest
                np.save(out_dir / fname, iq)  # iq_2ch shape (T,2), float32

                # manifest row (add anything else you need)
                mf.write(json.dumps({
                    "npy": rel_path,
                    "src_meta": Path(file_name).name,
                    "start": int(start),
                    "samples": int(t_samples),
                    "t_sec": float(t_samples / fs),
                    "fs": float(fs),
                    "shape": [t_samples, 2],
                    "dtype": "float32",
                    "ver": 1
                }) + "\n")

    mf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate IQ dataset with per-instance variable durations; each crop saved as its own .npy")
    ap.add_argument("--recordings_filepaths_list", type=str, nargs="+", required=True,
                    help="Directories containing .sigmf/.sigmf-data pairs")
    ap.add_argument("--sentence_duration", nargs="+", default=[0.128, 0.256],
                    help="Duration range in seconds, e.g., 0.128 0.256")
    ap.add_argument("--max_instances_per_file", type=int, default=0,
                    help="Cap the number of crops to save per file (0 = no cap)")
    ap.add_argument(
        "--coverage",
        type=float,
        default=1.0,
        help=(
            "Overlap multiplier for sampling. "
            "Use 1.0 for ~non-overlapping, 2–4 for light overlap, 8+ for heavy overlap."
        ),
    )

    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()
    main(args.recordings_filepaths_list, args.sentence_duration,
         args.max_instances_per_file, args.coverage, args.seed)
