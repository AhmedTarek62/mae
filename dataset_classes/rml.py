import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Subset


class RML(Dataset):
    def __init__(self, root, version, get_snr=False):
        self.root = Path(root)
        assert version in ("2016", "2022"), "version must be '2016' or '2022'"
        self.version = version
        self.get_snr = get_snr

        if version == "2022":
            self.data = np.load(self.root / "RML22.01A", allow_pickle=True)
        else:  # 2016
            with open(self.root / "RML2016.10a_dict.pkl", "rb") as f:
                self.data = pickle.load(f, encoding="latin1")

        self.keys = list(self.data.keys())               # keys are (mod, snr)
        self.sizes = [self.data[k].shape[0] for k in self.keys]
        self.cum_sizes = np.cumsum(self.sizes)
        self.starts = np.concatenate(([0], self.cum_sizes[:-1]))  # start idx per key group

        self.labels = (
            "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
            "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
        )
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}

        # --- New: cache a vector of SNRs aligned with global indices (O(N) once) ---
        N = int(self.cum_sizes[-1])
        self._snr_by_index = np.empty(N, dtype=np.int16)
        for gi, ((mod, snr), n, start) in enumerate(zip(self.keys, self.sizes, self.starts)):
            self._snr_by_index[start:start + n] = snr

        # Optional placeholder for per-index weights (filled by helper)
        self._weights = None

    def __len__(self):
        return int(self.cum_sizes[-1])

    def __getitem__(self, idx):
        # which key group?
        batch_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        sample_idx = idx if batch_idx == 0 else idx - self.cum_sizes[batch_idx - 1]

        mod, snr = self.keys[batch_idx]
        x = self.data[(mod, snr)][sample_idx]
        y = self.label_to_idx[mod]

        if self.get_snr:
            return (torch.as_tensor(x, dtype=torch.float32).unsqueeze(1),
                    torch.as_tensor(y, dtype=torch.long),
                    torch.as_tensor(snr, dtype=torch.long))
        else:
            return (torch.as_tensor(x, dtype=torch.float32).unsqueeze(1),
                    torch.as_tensor(y, dtype=torch.long))

    # --- New: expose SNRs (and weights if prepared) ---
    @property
    def snr_by_index(self):
        return self._snr_by_index

    @property
    def weights(self):
        return self._weights


def _unwrap_subset(ds):
    """
    Returns (base_dataset, index_map or None).
    If ds is a Subset (possibly nested), index_map maps base indices -> this view.
    """
    if not isinstance(ds, Subset):
        return ds, None

    # flatten nested Subsets to a single index array into the base dataset
    all_idx = ds.indices
    base = ds.dataset
    while isinstance(base, Subset):
        all_idx = [base.indices[i] for i in all_idx]
        base = base.dataset
    # as numpy array for fancy indexing
    return base, np.asarray(all_idx, dtype=np.int64)


def make_snr_sampler(dataset,
                     policy: str = "custom",
                     *,
                     snr_weights: dict | None = None,
                     mu: float = 12.0, sigma: float = 5.0, floor: float = 0.1,
                     low_cut: int = 0, high_cut: int = 18, tail_weight: float = 0.2,
                     temperature: float = 1.0,
                     normalize: bool = True,
                     num_samples: int | None = None,
                     replacement: bool = True,
                     generator: torch.Generator) -> WeightedRandomSampler:
    """
    Subset-aware: works with RML or any Subset(RML) (even nested).
    """
    base_ds, idx_map = _unwrap_subset(dataset)

    # Require the base dataset to expose SNRs per *base* index
    snrs_full = base_ds.snr_by_index.astype(np.float32)

    # If we're sampling a subset view, restrict to that view
    if idx_map is not None:
        snrs = snrs_full[idx_map]
    else:
        snrs = snrs_full

    # --- same policies as before ---
    if policy == "custom":
        assert snr_weights is not None and len(snr_weights), "Provide snr_weights={snr:int->weight}"
        w = np.vectorize(lambda s: snr_weights.get(int(s), 0.0), otypes=[np.float32])(snrs)

    elif policy == "gaussian":
        w = np.exp(-0.5 * ((snrs - mu) / max(1e-6, sigma))**2).astype(np.float32)
        w = floor + (1.0 - floor) * w

    elif policy == "clip":
        mid = (snrs >= low_cut) & (snrs <= high_cut)
        w = np.where(mid, 1.0, tail_weight).astype(np.float32)

    else:
        raise ValueError("policy must be 'custom', 'gaussian', or 'clip'")

    if temperature != 1.0:
        w = np.power(w, temperature, dtype=np.float32)

    w = np.clip(w, 1e-6, None)
    if normalize:
        w = w / w.sum()

    # sampler expects a 1D tensor aligned to *the dataset you pass to DataLoader*
    weights_tensor = torch.as_tensor(w, dtype=torch.double)

    if num_samples is None:
        num_samples = len(dataset)  # length of the (sub)set you’ll load from

    return WeightedRandomSampler(weights=weights_tensor,
                                 num_samples=num_samples,
                                 replacement=replacement,
                                 generator=generator)
