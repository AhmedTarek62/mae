import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, InterpolationMode
from pathlib import Path


class OfdmChannelEstimation_Dataset(Dataset):
    def __init__(self, data_path, batch_size=64, compute_stats=False, normalize_labels=False, split:str="train"):

        assert split in ["train", "test", "val"], print(f"split parameters must be train, test or val but got {split}")
        if split == "train":
            self.data_path = Path(os.path.join(data_path, 'train'))
        elif split == "val" or split == "test":
            self.data_path = Path(os.path.join(data_path, 'val'))

        self.batch_size = batch_size
        self.file_list = os.listdir(self.data_path)
        if compute_stats:
            self.min_features, self.max_features, self.min_label, self.max_label, self.mu, self.std = self._compute_stats_two()
        else:
            self.min_features = -12.33
            self.max_features = 13.09
            self.min_label = -6.49
            self.max_label = 6.34
            self.mu = [-0.0297, -0.0304]
            self.std = [0.0953, 0.0953]

        self.transform = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                  Lambda(lambda x: 2 * (x - self.min_features) / (self.max_features - self.min_features) - 1),
                                  Normalize(mean=self.mu, std=self.std),
                                  Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),])
        self.transform_label = Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32))
        if normalize_labels:
            self.transform_label = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                            Lambda(lambda x: 2 * (x - self.min_label) / (self.max_label - self.min_label) - 1)])
        else:
            self.transform_label = Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32))

    def _compute_stats(self):
        mu = np.zeros((2,), dtype=np.float32)
        std = np.zeros((2,), dtype=np.float32)
        mu_label = np.zeros((2,), dtype=np.float32)
        std_label = np.zeros((2,), dtype=np.float32)
        for file in self.file_list:
            data = np.load(os.path.join(self.data_path, file))
            x_rg_pilot = np.concatenate((data['x_rg'][:, 2], data['x_rg'][:, 11]), axis=1)
            x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=1)
            y_rg_pilot = np.concatenate((data['y_rg'][:, :, 2], data['y_rg'][:, :, 11]), axis=2)
            y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=1)
            x_model = np.concatenate((x_rg_pilot.reshape((self.batch_size, 2, 1, -1)), y_rg_pilot), axis=2)
            h_freq = np.concatenate([data['h_freq'][:, :, i] for i in range(14)], axis=2)
            h_freq = np.stack((h_freq.real, h_freq.imag), axis=1)
            mu += np.mean(x_model, axis=(0, 2, 3))
            std += np.std(x_model, axis=(0, 2, 3))
            mu_label += np.mean(h_freq, axis=(0, 2, 3))
            std_label += np.std(h_freq, axis=(0, 2, 3))

        mu /= len(self.file_list)
        std /= len(self.file_list)
        mu_label /= len(self.file_list)
        std_label /= len(self.file_list)

        min_label, max_label = np.inf, -np.inf
        for file in self.file_list:
            data = np.load(os.path.join(self.data_path, file))
            h_freq = np.concatenate([data['h_freq'][:, :, i] for i in range(14)], axis=2)
            h_freq = np.stack((h_freq.real, h_freq.imag), axis=1)
            h_freq = (h_freq - mu_label.reshape((1, 2, 1, 1))) / std_label.reshape((1, 2, 1, 1))
            min_label = min(min_label, np.min(h_freq))
            max_label = max(max_label, np.max(h_freq))

        return mu.tolist(), std.tolist(), mu_label.tolist(), std_label.tolist(), min_label, max_label

    def _compute_stats_two(self):
        min_features, max_features = np.inf, -np.inf
        min_label, max_label = np.inf, -np.inf
        for file in self.file_list:
            data = np.load(os.path.join(self.data_path, file))
            x_rg_pilot = np.concatenate((data['x_rg'][:, 2], data['x_rg'][:, 11]), axis=1)
            x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=1)
            y_rg_pilot = np.concatenate((data['y_rg'][:, :, 2], data['y_rg'][:, :, 11]), axis=2)
            y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=1)
            x_model = np.concatenate((x_rg_pilot.reshape((self.batch_size, 2, 1, -1)), y_rg_pilot), axis=2)
            h_freq = np.concatenate([data['h_freq'][:, :, i] for i in range(14)], axis=2)
            h_freq = np.stack((h_freq.real, h_freq.imag), axis=1)
            min_features = min(min_features, np.min(x_model))
            max_features = max(max_features, np.max(x_model))
            min_label = min(min_label, np.min(h_freq))
            max_label = max(max_label, np.max(h_freq))

        mu = np.zeros((2,))
        std = np.zeros((2,))
        for file in self.file_list:
            data = np.load(os.path.join(self.data_path, file))
            x_rg_pilot = np.concatenate((data['x_rg'][:, 2], data['x_rg'][:, 11]), axis=1)
            x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=1)
            y_rg_pilot = np.concatenate((data['y_rg'][:, :, 2], data['y_rg'][:, :, 11]), axis=2)
            y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=1)
            x_model = np.concatenate((x_rg_pilot.reshape((self.batch_size, 2, 1, -1)), y_rg_pilot), axis=2)
            x_model = 2 * (x_model - min_features) / (max_features - min_features) - 1
            mu += np.mean(x_model, axis=(0, 2, 3))
            std += np.std(x_model, axis=(0, 2, 3))
        mu /= len(self.file_list)
        std /= len(self.file_list)
        return min_features, max_features, min_label, max_label, mu.tolist(), std.tolist()

    def __getitem__(self, index):
        file_idx = index // self.batch_size
        sample_idx = index % self.batch_size
        data = np.load(os.path.join(self.data_path, self.file_list[file_idx]))
        x_rg_pilot = np.concatenate((data['x_rg'][sample_idx, 2], data['x_rg'][sample_idx, 11]))
        x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=0)
        y_rg_pilot = np.concatenate((data['y_rg'][sample_idx, :, 2], data['y_rg'][sample_idx, :, 11]), axis=1)
        y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=0)
        x_model = np.concatenate((x_rg_pilot.reshape((2, 1, -1)), y_rg_pilot), axis=1)
        h_freq = np.concatenate([data['h_freq'][sample_idx, :, i] for i in range(14)], axis=1)
        h_freq = np.stack((h_freq.real, h_freq.imag), axis=0)
        snr_db = torch.as_tensor(data['snr_db'][sample_idx], dtype=torch.float32)
        return self.transform(x_model), self.transform_label(h_freq), snr_db

    def create_sample(self, x_rg, y_rg):
        x_rg_pilot = np.concatenate((x_rg[:, 2], x_rg[:, 11]), axis=1)
        x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=1).reshape((x_rg_pilot.shape[0], 2, 1, -1))
        y_rg_pilot = np.concatenate((y_rg[:, :, 2], y_rg[:, :, 11]), axis=2)
        y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=1)
        x_model = np.concatenate((x_rg_pilot, y_rg_pilot), axis=2)
        return self.transform(x_model)

    def reverse_normalize(self, h):
        h = (h + 1) / 2 * (self.max_label - self.min_label) + self.min_label
        # h = (h * np.array(self.std_label, dtype=np.float32).reshape((1, 2, 1, 1)) +
        #      np.array(self.mu_label, dtype=np.float32).reshape((1, 2, 1, 1)))
        return h

    def __len__(self):
        return len(self.file_list) * self.batch_size
