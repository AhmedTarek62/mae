import os
import torch
from torchvision.transforms import Lambda, Compose, Resize, InterpolationMode, Normalize
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path


class CSIWiFi(Dataset):
    def __init__(self, root_dir, img_size=(224, 224), augment_transforms=None, downsampled=False, factor=2):
        self.root_dir = root_dir
        self.file_list = os.listdir(Path(root_dir))
        self.img_size = img_size
        self.downsampled = downsampled
        self.factor = factor
        if self.downsampled:
            self.min_val, self.max_val = -1.31, 54.37
            self.mu, self.std = [0.7094, 0.7570, 0.6963], [0.0842, 0.0801, 0.0851]
        else:
            self.min_val, self.max_val = -0.113, 53.83
            self.mu, self.std = [0.7098, 0.7590, 0.6966], [0.0824, 0.0789, 0.0824]

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
        csi = loadmat(os.path.join(self.root_dir, sample_name))['CSIamp']
        csi = csi.reshape(3, 114, -1)
        if self.downsampled:
            csi = csi[:, :, ::4]
        if self.augment_transforms:
            csi = self.augment_transforms(self.transforms(csi))
        else:
            csi = self.transforms(csi)
        return csi, torch.as_tensor([1,], dtype=torch.long)

    def get_raw_item(self, index):
        sample_name = self.file_list[index]
        csi = loadmat(os.path.join(self.root_dir, sample_name))['CSIamp']
        csi = csi.reshape(3, 114, -1)
        if self.downsampled:
            csi = csi[:, :, ::4]
        return csi

    def __len__(self):
        if self.augment_transforms:
            return self.factor * len(self.file_list)
        else:
            return len(self.file_list)

