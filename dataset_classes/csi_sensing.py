import os
import re

import torch
from torchvision.transforms import Lambda, Compose, Resize, InterpolationMode, Normalize
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path
import numpy as np


class CSISensingDataset(Dataset):
    def __init__(self, root_dir, img_size=(224, 224), augment_transforms=None, downsampled=False):
        self.root_dir = root_dir
        self.file_list = os.listdir(Path(root_dir))
        self.img_size = img_size
        self.labels = ['run', 'walk', 'fall', 'box', 'circle', 'clean']
        self.downsampled = downsampled
        if self.downsampled:
            self.min_val, self.max_val = -0.44, 56.07
            self.mu, self.std = [0.7352, 0.7653, 0.7686], [0.0899, 0.0717, 0.0831]
        else:
            self.min_val, self.max_val = 2.44, 54.72
            self.mu, self.std = [0.7396, 0.7722, 0.7758], [0.0960, 0.0764, 0.0888]
        self.transforms = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                   Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
                                   Lambda(lambda x: (x - self.min_val) / (self.max_val - self.min_val)),
                                   Normalize(self.mu, self.std)
                                   ])
        self.augment_transforms = augment_transforms

    def compute_stats(self):
        transforms = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                              Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC)])
        min_val, max_val = np.inf, -np.inf
        for sample_name in self.file_list:
            csi = loadmat(os.path.join(self.root_dir, sample_name))['CSIamp'].reshape(3, 114, -1)
            if self.downsampled:
                csi = csi[:, :, ::4]
            csi = transforms(csi)
            min_val = min(min_val, torch.min(csi).item())
            max_val = max(max_val, torch.max(csi).item())

        mu, std = torch.zeros((3,)), torch.zeros((3,))
        for sample_name in self.file_list:
            csi = loadmat(os.path.join(self.root_dir, sample_name))['CSIamp'].reshape(3, 114, -1)
            if self.downsampled:
                csi = csi[:, :, ::4]
            csi = transforms(csi)
            csi = (csi - min_val) / (max_val - min_val)
            mu += torch.mean(csi, dim=(1, 2))
            std += torch.std(csi, dim=(1, 2))

        mu /= len(self.file_list)
        std /= len(self.file_list)
        return min_val, max_val, mu, std

    @staticmethod
    def _split_at_number(s):
        match = re.match(r"([a-zA-Z]+)(\d+)", s)
        if match:
            return match.groups()[0]
        else:
            raise ValueError("String is not in correct format.")

    def __getitem__(self, index):
        sample_name = self.file_list[index]
        csi = loadmat(os.path.join(self.root_dir, sample_name))['CSIamp']
        csi = csi.reshape(3, 114, -1)
        if self.downsampled:
            csi = csi[:, :, ::4]
        label_name = self._split_at_number(sample_name)
        label_index = self.labels.index(label_name)
        if self.augment_transforms:
            csi = self.augment_transforms(self.transforms(csi))
        else:
            csi = self.transforms(csi)
        return csi, torch.as_tensor(label_index, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)

