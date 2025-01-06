import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose, Lambda, Resize
import torchvision.transforms.v2.functional as F
from torchvision import transforms
import torch
from scipy.io import loadmat
import os
from pathlib import Path


input_name_format = 'LTE_NR_spect_frame_{}.mat'
target_name_format = 'LTE_NR_labels_frame_{}.mat'
metadata_name_format = 'LTE_NR_metadata_frame_{}.mat'


class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir, img_shape=(224, 224), split:str="train"):

        assert split in ["train", "test", "val"], print(f"split parameters must be train, test or val but got {split}")
        if split == "train":
            self.dataset_dir = Path(os.path.join(dataset_dir, 'Train/LTE_NR'))
        elif split == "val" or split == "test":
            self.dataset_dir = Path(os.path.join(dataset_dir, 'Test/LTE_NR'))


        self.transform = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                  Lambda(lambda x: x.reshape((1, x.shape[0], x.shape[1]))),
                                  Resize(img_shape, antialias=True)])

        # calculate class weights
        class_frequency = np.zeros((3,))
        for idx in range(len(os.listdir(self.dataset_dir)) // 3):
            labels = loadmat(os.path.join(self.dataset_dir, target_name_format.format(idx)))['data']
            class_frequency[0] += np.sum(labels == 0)
            class_frequency[1] += np.sum(labels == 127)
            class_frequency[2] += np.sum(labels == 255)

        weights = 1 / class_frequency
        weights /= np.sum(weights)
        self.weights = weights

    def __getitem__(self, index):
        features = loadmat(os.path.join(self.dataset_dir, input_name_format.format(index)))['rxSpectrogram']
        labels = loadmat(os.path.join(self.dataset_dir, target_name_format.format(index)))['data']

        labels[labels == 127] = 1
        labels[labels == 255] = 2

        # tokenize features
        features = self.transform(features)
        _, h, w = features.shape
        labels = F.resize(torch.as_tensor(labels, dtype=torch.long).reshape((1, 256, 256)),(h, w), antialias=True,
                          interpolation=transforms.InterpolationMode.BICUBIC)
        return features, labels

    def __len__(self):
        return len(os.listdir(self.dataset_dir)) // 3 - 1
