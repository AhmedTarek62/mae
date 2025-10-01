import torch
from pathlib import Path
from models_mae_hetero import mae_vit_small_patch16
from dataset_classes.csi_sensing import CSISensingDataset
from dataset_classes.radio_sig import RadioSignal
from dataset_classes.ofdm_channel_estimation import OfdmChannelEstimation
from dataset_classes.positioning import Positioning5G
from dataset_classes.pretrain_csi_5g import CSI5G
from dataset_classes.pretrain_csi_wifi import CSIWiFi
from dataset_classes.spectrogram_images import SpectrogramImages
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def sample_batch(dataset, n, num_workers=0, pin_memory=True):
    """
    Return a single tensor of `n` randomly drawn examples (no labels).
    Assumes each Dataset item is (x, y) or just x.
    """
    loader = DataLoader(
        dataset,
        batch_size=n,            # ask for all n at once
        shuffle=True,            # random draw
        drop_last=True,          # guarantees exactly n
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    batch = next(iter(loader))   # one shot, no explicit loop

    # if dataset returns (data, label) tuples, strip the labels
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    return batch


def generate_embeddings(model, x):
    # embed patches
    c = x.shape[1]
    idx = model.chan_mapping[c]
    x = model.patch_embed[idx](x)

    # add pos embed w/o cls token
    x = x + model.pos_embed[:, 1:, :]

    # append cls token
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # apply Transformer blocks
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)

    return x


# pretraining datasets
dataset_wifi = CSIWiFi(Path('../datasets/NTU-Fi-HumanID/'))
dataset_5g = CSI5G(Path('../datasets/5G_CFR'))

transform_train = transforms.Compose([
    transforms.functional.pil_to_tensor,
    transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
    transforms.Lambda(lambda x: (x + 120) / (-0.5 + 120)),
    transforms.Resize((224, 224), antialias=True,
                      interpolation=InterpolationMode.BICUBIC),  # Resize
    transforms.Normalize(mean=[0.451], std=[0.043])  # Normalize
])
dataset_spectro = SpectrogramImages(['..\\datasets\\spectrogram_dataset', '..\\datasets\\spectrogram_iqengine_dataset'],
                                    transform=transform_train)

# fine-tuning datasets
dataset_sensing = CSISensingDataset(Path('../datasets/NTU-Fi_HAR/train'))
dataset_positioning = Positioning5G(Path('../datasets/5G_NR_Positioning/outdoor/train'))
dataset_ce = OfdmChannelEstimation(Path('../datasets/channel_estimation_dataset_(5,10)/train_preprocessed'))
dataset_rf = RadioSignal(Path('../datasets/radio_sig_identification/train'))

# all data
datasets = {'RFS': dataset_spectro, 'WiFi-CSI': dataset_wifi, '5G-CSI': dataset_5g, 'sensing': dataset_sensing,
            'rf_classification': dataset_rf, 'positioning': dataset_positioning}

# models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(Path('checkpoints/pretrained_all_data.pth'))['model']
model = mae_vit_small_patch16(in_chans=[1, 3, 4])
model.load_state_dict(ckpt, strict=True)
model = model.to(device)
model.eval()

# compute embeddings
num_samples = 400
embed_dim = 512
embeddings = dict()
for name, dataset in datasets.items():
    samples = sample_batch(dataset, n=num_samples)
    samples = samples.to(device)
    with torch.no_grad():
        embed = generate_embeddings(model, samples)
        embed = embed[:, 1:].mean(axis=1)
    embeddings[name] = embed


all_embeds = []
all_labels = []
for name, e in embeddings.items():              # e: (400, 512) on GPU
    all_embeds.append(e.cpu().numpy())         # move → CPU, ndarray
    all_labels.extend([name] * e.shape[0])

all_embeds = np.vstack(all_embeds)             # (N_total, 512)

# ---- 2. fit UMAP (2‑D) ----
n_neighbours = 30
min_dist = 0.1
reducer_2d = umap.UMAP(n_neighbors=n_neighbours, min_dist=min_dist, metric="cosine", random_state=0)
emb_2d = reducer_2d.fit_transform(all_embeds)     # (N_total, 2)

pca2d = PCA(n_components=2, random_state=0).fit_transform(all_embeds)   # (N, 2)
perplexity = min(30, (len(all_labels)-1)//3)   # empirical safe choice
tsne2d = TSNE(n_components=2,
              perplexity=perplexity,
              metric="cosine",
              init="pca",
              learning_rate="auto",
              random_state=0).fit_transform(all_embeds)

# ---- 3. plot ----
unique_labels = sorted(set(all_labels))
palette = plt.cm.get_cmap("tab10", len(unique_labels))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["UMAP", "PCA", "t‑SNE"]

for ax, proj, title in zip(axes,
                           [emb_2d, pca2d, tsne2d],
                           titles):
    for i, lbl in enumerate(unique_labels):
        idx = [j for j, x in enumerate(all_labels) if x == lbl]
        ax.scatter(proj[idx, 0],
                   proj[idx, 1],
                   s=6,
                   alpha=0.7,
                   color=palette(i),
                   label=lbl if title == "UMAP" else None)  # legend only once
    ax.set_title(title, fontsize=12)
    ax.axis("off")

axes[0].legend(markerscale=2, fontsize=9, frameon=False, loc="upper right")
plt.tight_layout()
plt.show()
test = []
# pairs = list(product(['RFS', 'WiFi-CSI', '5G-CSI'], ['sensing', 'positioning', 'rf_classification']))
#
# fig, axes = plt.subplots(3, 3, figsize=(12, 12))   # adjust grid
# for ax, (src, tgt) in zip(axes.flat, pairs):
#     feats = np.vstack([embeddings[src].cpu(), embeddings[tgt].cpu()])
#     labels = [src] * 400 + [tgt] * 400
#     emb2d = umap.UMAP(n_neighbors=15, min_dist=0.1,
#                       metric='cosine', random_state=0).fit_transform(feats)
#     for lbl, c in zip([src, tgt], ['tab:blue', 'tab:orange']):
#         idx = [i for i, l in enumerate(labels) if l == lbl]
#         ax.scatter(emb2d[idx, 0], emb2d[idx, 1], s=6, alpha=0.7, label=lbl, c=c)
#     ax.set_title(f'{src} ↔ {tgt}')
#     ax.axis('off')
#     ax.legend(frameon=False, fontsize=8)
# plt.tight_layout()
# plt.show()

