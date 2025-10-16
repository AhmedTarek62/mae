import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, InterpolationMode


class Positioning5G(Dataset):
    """
    A PyTorch Dataset for 5G NR-based positioning.

    This dataset:
    - Loads CSI (Channel State Information) features from `.h5` files.
    - Normalizes and transforms features for model training.
    - Converts position labels to a normalized range [-1, 1].
    - Supports both 'outdoor' and 'indoor' positioning scenarios.

    Parameters:
    ----------
    datapath : str or Path
        Path to the dataset directory containing `.h5` files.
    img_size : tuple, default=(224, 224)
        Target image size for resizing.
    scene : str, default='outdoor'
        Specifies whether to load 'outdoor' or 'indoor' scene data.

    Raises:
    -------
    ValueError
        If the specified scene is not recognized.
    """

    def __init__(self, datapath, img_size=(224, 224), scene='outdoor', pretrain=False, augment_transforms=None):
        self.img_size = img_size
        self.scene = scene
        self.pretrain = pretrain

        # Get list of all `.h5` files for the specified scene
        self.data_files = tuple(sorted([os.path.join(datapath, filename) for filename in os.listdir(datapath)]))
        self.num_samples = len(self.data_files)
        self.augment_transforms = augment_transforms

        # Define scene-specific normalization parameters
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
            raise ValueError("Scene not recognized. Choose either 'outdoor' or 'indoor'.")

        # Define transformations for feature preprocessing
        self.transforms = Compose([
            Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),  # Convert to PyTorch tensor
            Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC),  # Resize feature maps
            Lambda(lambda x: (x - self.min_val) / (self.max_val - self.min_val)),  # Min-max normalization
            Normalize(self.mu, self.std)  # Standardization
        ])

    def __getitem__(self, index):
        """
        Loads and preprocesses a sample.

        Parameters:
        ----------
        index : int
            Index of the sample to retrieve.

        Returns:
        -------
        tuple : (torch.Tensor, torch.Tensor)
            Processed CSI feature tensor and normalized position label.
        """
        # Load features and labels from an `.h5` file
        with h5py.File(self.data_files[index], 'r') as data_file:
            features = np.array(data_file['features'])  # CSI feature map
            if self.pretrain:
                labels = torch.as_tensor([1,], dtype=torch.float32)
            else:
                labels = torch.as_tensor(np.array(data_file['position']), dtype=torch.float)  # Position label
                labels = 2 * (labels - self.coord_nominal_min) / (self.coord_nominal_max - self.coord_nominal_min) - 1

        # Apply transformations to features
        features = self.transforms(features)
        if self.augment_transforms is not None:
            features = self.augment_transforms(features)

        return features, labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        -------
        int
            Total number of samples.
        """
        return self.num_samples
