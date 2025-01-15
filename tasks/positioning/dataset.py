import numpy as np
from torch.utils.data import Dataset
import h5py
import os
from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
import torch
from torch.utils.data import random_split
from pathlib import Path


class TaskDataset(Dataset):
    def __init__(self, datapath, img_size=(224, 224), scene='outdoor', split:str="train"):

        assert split in ["train", "test", "val"], print(f"split parameters must be train, test or val but got {split}")
        if split == "train":
            self.datapath = Path(os.path.join(datapath, 'train'))
        elif split == "val" or split == "test":
            self.datapath = Path(os.path.join(datapath, 'test'))

        self.img_size = img_size
        self.scene = scene
        self.file_paths = [os.path.join(self.datapath, filename)
                           for filename in os.listdir(self.datapath) if scene in filename]

        self.offsets = []

        # Create a global index for all samples across all files
        offset = 0
        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as data_file:
                num_samples = len(data_file['features'])
                self.offsets.append(num_samples + offset)
                offset += num_samples

        # Precompute stats (based on a subset if necessary)
        # self.min_val, self.max_val, self.mu, self.std, self.coord_min, self.coord_max = self.compute_stats()
        if scene == 'outdoor':
            self.min_val, self.max_val = -0.096, 1.136
            self.mu = torch.as_tensor([0.4638, 0.4631, 0.4703, 0.4620])
            self.std = torch.as_tensor([0.1154, 0.1176, 0.0979, 0.1281])
            self.coord_nominal_min = torch.as_tensor([0, 0, 0], dtype=torch.float32)
            self.coord_nominal_max = torch.as_tensor([80, 60, 40], dtype=torch.float32)
        elif scene == 'indoor':
            self.min_val, self.max_val = -0.123, 1.415
            self.mu = torch.as_tensor([0.3824, 0.3853, 0.3841, 0.3931, 0.3909])
            self.std = torch.as_tensor([0.1168, 0.1112, 0.1182, 0.0988, 0.0972])
            self.coord_nominal_min = torch.as_tensor([0, 0, 0], dtype=torch.float32)
            self.coord_nominal_max = torch.as_tensor([60, 20, 4], dtype=torch.float32)
        else:
            raise ValueError('Scene not recognized')
        # Define transformations
        self.transforms = Compose([
            Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
            Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
            Lambda(lambda x: (x - self.min_val) / (self.max_val - self.min_val)),
            Normalize(self.mu, self.std)
        ])


    @staticmethod
    def _nan_fill_with_mean(array):
        # Compute the mean of non-NaN values
        mean_value = np.nanmean(array)  # Handles NaNs automatically

        # Replace NaNs with the mean value
        array[np.isnan(array)] = mean_value

        return array

    def compute_stats(self):
        data_files = [h5py.File(file_path) for file_path in self.file_paths]
        min_val, max_val = float('inf'), -float('inf')
        coord_min = [float('inf'),] * 3
        coord_max = [-float('inf'),] * 3

        transforms = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                              Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC)])
        for data_file in data_files:
            features = np.array(data_file['features'])
            features = transforms(features)
            labels = np.array(data_file['labels']['position'])
            min_val = min(features.min().item(), min_val)
            max_val = max(features.max().item(), max_val)
            coord_min = [min(i, j) for (i, j) in zip(coord_min, np.min(labels, axis=0))]
            coord_max = [max(i, j) for (i, j) in zip(coord_max, np.max(labels, axis=0))]

        coord_min = torch.tensor(coord_min, dtype=torch.float32)
        coord_max = torch.tensor(coord_max, dtype=torch.float32)
        mu = torch.zeros((4,), dtype=torch.float32) if self.scene == 'outdoor' else torch.zeros((5,), dtype=torch.float32)
        std = torch.zeros((4,), dtype=torch.float32) if self.scene == 'outdoor' else torch.zeros((5,), dtype=torch.float32)

        transforms = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                              Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
                              Lambda(lambda x: (x - min_val) / (max_val - min_val))
                              ])

        chunk_size = 500
        num_samples = 0
        for data_file in data_files:
            for i in range(0, len(data_file['features']), chunk_size):
                if i + chunk_size > len(data_file['features']):
                    features = data_file['features'][i:]
                else:
                    features = data_file['features'][i: i + chunk_size]
                features = transforms(features)
                mu += torch.mean(features, dim=(0, 2, 3))
                std += torch.std(features, dim=(0, 2, 3))
                num_samples += 1
        mu /= num_samples
        std /= num_samples
        return min_val, max_val, mu, std, coord_min.view((1, 3)), coord_max.view((1, 3))

    def __getitem__(self, index):
        # Map global index to specific file and local index
        if index >= self.offsets[-1]:
            raise IndexError

        for i, offset in enumerate(self.offsets):
            if index < offset:
                file_idx = i
                break

        if file_idx != 0:
            relative_index = index - self.offsets[file_idx - 1]
        else:
            relative_index = index

        # Lazy load features and labels
        with h5py.File(self.file_paths[file_idx], 'r') as data_file:
            features = data_file['features'][relative_index]
            labels = torch.as_tensor(data_file['labels']['position'][relative_index], dtype=torch.float)

        # Normalize labels
        labels = 2 * (labels - self.coord_nominal_min) / (self.coord_nominal_max - self.coord_nominal_min) - 1

        # Apply transformations to features
        features = self.transforms(features)

        return features, labels

    def __len__(self):
        return self.offsets[-1]

