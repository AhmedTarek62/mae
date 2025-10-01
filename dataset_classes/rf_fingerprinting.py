from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class RFPrintDataset(Dataset):
    """
    RF fingerprinting dataset for interleaved IQ recordings (.bin) with JSON metadata.

    Each sample is a chunk of 512 complex samples (shape: (2, 512) as float32).
    Label comes from annotations.transmitter['core:location'] in the JSON.

    Args:
        root: directory containing matching *.bin and *.json files (same stem).
        chunk_len: number of complex samples per example (default 512).
        hop_len: hop between chunks (complex samples). If None, equals chunk_len (no overlap).
        allowed_labels: list of transmitter locations to keep (order defines class indices).
                        Default: ['bes','browning','honors','meb']
        normalize: if True, per-chunk per-channel standardization (zero-mean, unit-var).
        return_meta: if True, __getitem__ returns (x, y, meta_dict).
    """
    def __init__(
        self,
        root: str | Path,
        chunk_len: int = 512,
        hop_len: Optional[int] = None,
        allowed_labels: Optional[List[str]] = None,
        normalize: bool = True,
        return_meta: bool = False,
    ):
        self.root = Path(root)
        self.chunk_len = int(chunk_len)
        self.hop_len = int(hop_len) if hop_len is not None else int(chunk_len)
        self.normalize = bool(normalize)
        self.return_meta = bool(return_meta)

        # Allowed transmitter locations -> class indices
        self.labels = allowed_labels or ['bes', 'browning', 'honors', 'meb']
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}

        # Build file list (match .bin/.json by stem), then build a flat index of chunk offsets
        self.records: List[Dict] = []   # per-recording metadata
        self.index: List[Dict] = []     # flat list of chunk entries

        jsons = sorted(self.root.glob("*.json"))
        stems_with_bin = {p.stem for p in self.root.glob("*.bin")}
        for jpath in jsons:
            stem = jpath.stem
            if stem not in stems_with_bin:
                continue
            bpath = self.root / f"{stem}.bin"

            # Read metadata JSON
            meta = self._read_json(jpath)

            # Verify datatype
            dtype_str = meta.get('global', {}).get('core:datatype', '').lower()
            if dtype_str not in ('cf32', 'complex64', 'float32_complex'):
                # Skip unknown/unsupported types; adapt here if your corpus contains others
                continue

            # Transmitter label
            tx = (
                meta.get('annotations', {})
                    .get('transmitter', {})
                    .get('core:location', '')
                    .strip()
                    .lower()
            )
            if tx not in self.label_to_idx:
                # Skip recordings with labels outside our set
                continue
            y = self.label_to_idx[tx]

            # Optional metadata
            sr = self._safe_float(meta.get('global', {}).get('core:sample_rate'))
            cf = self._safe_float(meta.get('captures', {}).get('core:center_frequency'))

            # How many complex samples in this .bin?
            # cf32 => numpy.complex64 => 8 bytes per complex sample
            nbytes = os.path.getsize(bpath)
            itemsize = np.dtype(np.complex64).itemsize  # 8
            n_complex = nbytes // itemsize
            if n_complex < self.chunk_len:
                continue  # too short

            # Chunking
            hop = self.hop_len
            n_chunks = 1 + (n_complex - self.chunk_len) // hop

            rec_id = len(self.records)
            self.records.append(dict(
                stem=stem, bin_path=str(bpath), json_path=str(jpath),
                label_idx=y, label_name=tx, sample_rate=sr, center_freq=cf,
                n_complex=n_complex,
            ))

            for c in range(n_chunks):
                start = c * hop
                self.index.append(dict(
                    rec_id=rec_id,
                    start=start,
                    length=self.chunk_len,
                ))

        if len(self.index) == 0:
            raise RuntimeError(
                f"No chunks indexed. Check your folder '{self.root}' and metadata format."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        rec = self.records[entry['rec_id']]
        start = entry['start']
        length = entry['length']

        # Memory-map just this file and slice needed complex segment
        # Using complex64 directly interprets interleaved float32 IQ as complex numbers.
        mm = np.memmap(rec['bin_path'], mode='r', dtype=np.complex64)
        seg_c = mm[start:start+length]  # shape: (length,)

        # Convert to (2, T) float32 [I; Q]
        x = np.empty((2, length), dtype=np.float32)
        x[0] = seg_c.real.astype(np.float32, copy=False)
        x[1] = seg_c.imag.astype(np.float32, copy=False)

        if self.normalize:
            # Per-channel standardization for stability
            # (zero mean, unit variance, with small epsilon)
            for ch in range(2):
                mu = x[ch].mean()
                sigma = x[ch].std()
                x[ch] = (x[ch] - mu) / (sigma + 1e-7)

        x_t = torch.from_numpy(x).unsqueeze(1)                  # (2, chunk_len), float32
        y_t = torch.tensor(rec['label_idx']).long()  # class index

        if self.return_meta:
            meta = dict(
                label_name=rec['label_name'],
                stem=rec['stem'],
                sample_rate=rec['sample_rate'],
                center_freq=rec['center_freq'],
                start=start,
                chunk_len=length,
            )
            return x_t, y_t, meta
        else:
            return x_t, y_t

    @staticmethod
    def _read_json(path: Path) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _safe_float(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

