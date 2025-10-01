import os
import numpy as np
import torch
from scipy.io import loadmat
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, InterpolationMode


class CSI5G(Dataset):
    def __init__(self, root_dir, img_size=(224, 224), augment_transforms=None, downsampled=False):
        self.root_dir = root_dir
        self.file_list = os.listdir(Path(root_dir))
        self.img_size = img_size
        self.downsampled = downsampled
        if self.downsampled:
            self.min_val, self.max_val = -2.432, 2824.16
            self.mu, self.std = [0.1405, 0.1079, 0.1239, 0.0612], [0.0389, 0.0298, 0.0341, 0.0164]
        else:
            self.min_val, self.max_val = 4.306, 2833.35
            self.mu, self.std = [0.1380, 0.1054, 0.1214, 0.0588], [0.0386, 0.0296, 0.0339, 0.0163]
        self.transforms = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                   Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
                                   Lambda(lambda x: (x - self.min_val) / (self.max_val - self.min_val)),
                                   Normalize(self.mu, self.std)
                                   ])
        self.augment_transforms = augment_transforms

    def __getitem__(self, index):
        if index >= len(self.file_list):
            index %= len(self.file_list)
        sample_name = self.file_list[index]
        csi = np.abs(loadmat(os.path.join(self.root_dir, sample_name))['cfr']).transpose(1, 0, 2)
        if self.downsampled:
            csi = csi[:, :, ::4]
        if self.augment_transforms:
            csi = self.augment_transforms(self.transforms(csi))
        else:
            csi = self.transforms(csi)
        return csi, torch.as_tensor([1,], dtype=torch.long)

    def get_raw_item(self, index):
        if index >= len(self.file_list):
            index %= len(self.file_list)
        sample_name = self.file_list[index]
        csi = np.abs(loadmat(os.path.join(self.root_dir, sample_name))['cfr']).transpose(1, 0, 2)
        if self.downsampled:
            csi = csi[:, :, ::4]
        return csi

    def __len__(self):
        if self.augment_transforms:
            return int(3.5 * len(self.file_list))
        else:
            return len(self.file_list)
