import argparse
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from dataset_classes.csi_sensing import CSISensingDataset
from torch.utils.data import DataLoader
import mae.models.models_vit as models_vit
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def main(args):
    device = args.device
    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    mask_ratios = args.mask_ratios
    models = [("vit_small_patch16", "finetuned_sensing_small_%d.pth"),]
              # ("vit_medium_patch16", "finetuned_sensing_medium_%d.pth"),
              # ("vit_large_patch16", "finetuned_sensing_large_%d.pth")]
    batch_size = args.batch_size
    num_workers = args.num_workers

    test_set = CSISensingDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    accuracies = torch.zeros((len(models), len(mask_ratios)), dtype=torch.float).to(device)

    with torch.no_grad():
        for i, (model_key, model_name) in enumerate(tqdm(models, desc="Models")):
            for j, mask_ratio in enumerate(tqdm(mask_ratios, desc=f"Mask Ratios for {model_key}", leave=False)):
                ckpt_path = os.path.join(ckpt_dir, model_name % mask_ratio)
                ckpt = torch.load(ckpt_path, map_location=device)['model']
                model = getattr(models_vit, model_key)(global_pool='token', num_classes=6)
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias', 'pos_embed']:
                    if k in ckpt and ckpt[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del ckpt[k]
                ckpt['patch_embed.proj.weight'] = ckpt['patch_embed.proj.weight'].expand(-1, 3, -1, -1)

                model.load_state_dict(ckpt, strict=False)

                model = model.to(device)

                for k, (images, targets) in enumerate(tqdm(test_loader, desc="Batches", leave=False)):
                    images = images.to(device)
                    targets = targets.to(device)
                    pred = model(images)
                    pred = torch.argmax(pred, dim=-1)
                    accuracies[i, j] += torch.sum(targets == pred) / len(targets)

    accuracies /= batch_size

    # Save the accuracy array
    np.save(args.output_accuracy, accuracies.cpu().numpy())

    # Plot accuracy vs mask_ratio for each model
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']  # Different markers for each model
    colors = ['r', 'b', 'm']
    labels = ['ViT-S', 'ViT-M', 'ViT-L']
    for i in range(len(models)):
        plt.plot(mask_ratios, accuracies[i].cpu().numpy(), color=colors[i], marker=markers[i], label=labels[i],
                 linewidth=2)

    plt.xlabel('Mask Ratio (%)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Accuracy vs Mask Ratio for Different Models')
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(args.output_plot, dpi=800, bbox_inches='tight')
    print("Accuracies:", accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT models on CSI sensing dataset with varying mask ratios.")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--data_dir', type=str, default='../datasets/NTU-Fi_HAR/test',
                        help='Path to CSI Sensing dataset directory')
    parser.add_argument('--mask_ratios', type=int, nargs='+', default=[20, 30, 40, 50, 60, 65, 70, 75, 80, 85],
                        help='List of mask ratios to test')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--plot', action='store_true', help='Flag to enable plotting of images')
    parser.add_argument('--output_plot', type=str, default='accuracy_vs_mask_ratio_sensing.png',
                        help='Path to save the accuracy plot')
    parser.add_argument('--output_accuracy', type=str, default='accuracies_sensing.npy',
                        help='Path to save the accuracy array as a .npy file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)
