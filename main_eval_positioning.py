from tqdm import tqdm

import mae.models.sensing as sensing
from dataset_classes.positioning_nr import PositioningNR
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.utils.data import random_split, DataLoader


def reverse_normalize(x, coord_min, coord_max):
    return (x + 1) / 2 * (coord_max - coord_min) + coord_min


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
scene = 'indoor'
dataset = PositioningNR(Path('../datasets/5G_NR_Positioning'), scene=scene)
dataset_train, dataset_test = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(seed))
coord_min, coord_max = dataset.coord_nominal_min.view((1, -1)), dataset.coord_nominal_max.view((1, -1))

dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=False, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = sensing.__dict__['vit_medium_patch16'](global_pool='token', num_classes=3, drop_path_rate=0.1, in_chans=5)
checkpoint = torch.load(Path('checkpoints/checkpoint-70.pth'), map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=True)
print(msg)

model = model.to(device)
distances_train = torch.zeros((len(dataset_train),))

with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader_train), desc='Train Batch', total=len(dataloader_train)):
        image, target = batch
        image = image.to(device)
        pred_position = reverse_normalize(model(image).cpu(), coord_min, coord_max)
        position = reverse_normalize(target.cpu(), coord_min, coord_max)
        num_samples = target.shape[0]
        distances_train[i * num_samples: (i + 1) * num_samples] = torch.sqrt(torch.sum((pred_position - position) ** 2, dim=1))


distances_test = torch.zeros((len(dataset_test),))
with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader_test), desc='Test Batch', total=len(dataloader_test)):
        image, target = batch
        image = image.to(device)
        pred_position = reverse_normalize(model(image).cpu(), coord_min, coord_max)
        position = reverse_normalize(target.cpu(), coord_min, coord_max)
        num_samples = target.shape[0]
        distances_test[i * num_samples: (i + 1) * num_samples] = torch.sqrt(torch.sum((pred_position - position) ** 2, dim=1))

distances_train = distances_train.numpy()
distances_test = distances_test.numpy()
# distances_train.sort()
# distances_test.sort()

# cdf_train = np.linspace(0, 1, len(dataset_train))
# cdf_test = np.linspace(0, 1, len(dataset_test))
# idx_90_train = np.argmin(np.abs(cdf_train - 0.1))
# idx_90_test = np.argmin(np.abs(cdf_test - 0.1))
#
# plt.rcParams['font.family'] = 'serif'
# fig, axs = plt.subplots(1, 1)
# axs.plot(distances_train, cdf_train, label='train', linewidth=2, color='r')
# axs.plot(distances_test, cdf_test, label='test', linewidth=2, color='b')
# axs.axhline(y=0.1, linewidth=1, linestyle='--', label='90% likely', color='k')
# axs.axvline(x=distances_train[idx_90_train], linewidth=1, linestyle='--', color='r', alpha=0.8)
# axs.axvline(x=distances_test[idx_90_test], linewidth=1, linestyle='--', color='b', alpha=0.8)
# axs.set_xlabel('Positioning Error (m)')
# axs.set_ylabel('CDF')
# axs.legend(loc='lower right')
# plt.tight_layout()
# # plt.savefig('cdf_positioning.png', dpi=300)
# plt.show()

plt.rcParams['font.family'] = 'serif'
mean_train = np.mean(distances_train)
mean_test = np.mean(distances_test)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
model = 'Finetuning ViT-M'
other = '(2 out of 12 blocks + linear layer)'
fig.suptitle(f'{model} {other}\n{scene} scenario')
bins = 25
axs[0].hist(distances_train, bins=bins, color='red', edgecolor='w', alpha=0.7, density=True)
axs[0].axvline(mean_train, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_train:.2f} (m)')
axs[0].set_title('Training')
axs[0].set_xlabel('Positioning Error (m)')
axs[0].set_ylabel('Probability Density')
axs[0].legend()

axs[1].hist(distances_test, bins=bins, color='blue', edgecolor='w', alpha=0.7, density=True)
axs[1].axvline(mean_test, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_test:.2f} (m)')
axs[1].set_title('Test')
axs[1].set_xlabel('Positioning Error (m)')
axs[1].set_ylabel('Probability Density')
axs[1].legend()

plt.tight_layout()
plt.savefig('hist_positioning.png', dpi=300)
plt.show()
