import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import Normalize, Compose, Resize, InterpolationMode, Grayscale, ToTensor, Lambda
from pathlib import Path

class SignalIdentificatio_Dataset(Dataset):
    def __init__(self, data_path, split:str="train"):
         
        assert split in ["train", "test", "val"], print(f"split parameters must be train, test or val but got {split}")
        if split == "train":
            self.data_path = Path(os.path.join(data_path, 'train'))
        elif split == "val" or split == "test":
            self.data_path = Path(os.path.join(data_path, 'test'))

        self.labels = ['ads-b', 'airband', 'ais', 'automatic-picture-transmission', 'bluetooth', 'cellular',
                       'digital-audio-broadcasting', 'digital-speech-decoder', 'fm', 'lora', 'morse', 'on-off-keying',
                       'packet', 'pocsag', 'Radioteletype', 'remote-keyless-entry', 'RS41-Radiosonde', 'sstv', 'vor',
                       'wifi']
        self.class_freqs = {label: sum(1 if sample.startswith(label) else 0 for sample in os.listdir(self.data_path))
                            for label in self.labels}
        total_samples = sum(self.class_freqs.values())
        class_weights = [freq / total_samples for freq in self.class_freqs.values()]
        class_weights = [1 / weight for weight in class_weights]
        sum_class_weights = sum(class_weights)
        class_weights = [weight / sum_class_weights for weight in class_weights]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.transform = Compose([ToTensor(),
                                  Grayscale(),
                                  Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                  Normalize(mean=[0.5,], std=[0.5,])])

    def __getitem__(self, index):
        sample_name = os.listdir(self.data_path)[index]
        sample_path = os.path.join(self.data_path, sample_name)
        sample = Image.open(sample_path)
        sample = sample.transpose(Image.ROTATE_90)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = sample_name.split('_')[0]
        return self.transform(sample), torch.as_tensor(self.labels.index(label), dtype=torch.long)

    def __len__(self):
        return len(os.listdir(self.data_path))

