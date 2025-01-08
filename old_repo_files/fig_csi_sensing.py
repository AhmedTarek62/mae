from pathlib import Path
from mae.dataset_classes.csi_sensing_dataset import CSISensingDataset
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
csi_sensing = CSISensingDataset(Path('../datasets/NTU-Fi_HAR/train'))

indices = [10, 200, 320, 570, 670, 830]
data = [csi_sensing[i] for i in indices]
class_names = csi_sensing.labels
images_labels = [(np.transpose(image.numpy(), (1, 2, 0)), class_names[int(label)]) for image, label in data]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axs = axs.ravel()

for ax, (image, label) in zip(axs, images_labels):
    ax.imshow(image[:, :, 0], cmap='viridis')
    ax.set_title(label, fontsize=28)
    ax.axis('off')
plt.tight_layout()
plt.savefig(Path('Figures/fig_csi_sensing.png'), dpi=800)
plt.show()
