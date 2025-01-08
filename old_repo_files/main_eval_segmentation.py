import argparse
from pathlib import Path
import os

import torch
from dataset_classes.segmentation_dataset import SegmentationDataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import mae.models.models_segmentation as models_segmentation
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np


def main(args):
    device = args.device
    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    models = [("seg_vit_small_patch16", "segm_small_70.pth"), ("seg_vit_small_patch16", "segm_small_75.pth"), ("seg_vit_small_patch16", "segm_small_80.pth"),
              ("seg_vit_medium_patch16", "segm_medium_70.pth"), ("seg_vit_medium_patch16", "segm_medium_75.pth"), ("seg_vit_medium_patch16", "segm_medium_80.pth"),
              ("seg_vit_large_patch16", "segm_large_70.pth"), ("seg_vit_large_patch16", "segm_large_75.pth"), ("seg_vit_large_patch16", "segm_large_80.pth")]

    batch_size = args.batch_size
    num_workers = args.num_workers

    test_set = SegmentationDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    accuracies = np.zeros((len(models),))
    conf_matrices = np.zeros((len(models), 3, 3))
    with torch.no_grad():
        for i, (model_key, model_name) in enumerate(tqdm(models, desc="Models")):
            ckpt_path = os.path.join(ckpt_dir, model_name)
            ckpt = torch.load(ckpt_path, map_location=device)['model']
            model = getattr(models_segmentation, model_key)()
            model.load_state_dict(ckpt, strict=False)

            model = model.to(device)

            all_targets = []
            all_preds = []
            for k, (images, targets) in enumerate(tqdm(test_loader, desc="Batches", leave=False)):
                images = images.to(device)
                targets = targets.to(device)
                pred = model(images)
                pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1]).argmax(
                    dim=-1).detach().cpu().numpy()
                targets = targets.view(-1).detach().cpu().numpy()
                all_targets.extend(targets.tolist())
                all_preds.extend(pred.tolist())
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            conf_matrices[i] = confusion_matrix(all_targets, all_preds)
            accuracies[i] = np.sum(all_targets == all_preds) / len(all_targets)

    # Save the accuracy array
    np.save(args.output_conf_mats, conf_matrices)
    np.save(args.output_accuracy, accuracies)
    for i in range(9):
        row_sums = np.sum(conf_matrices[i], axis=1)
        row_sums[row_sums == 0] = 1
        conf_matrices[i] = conf_matrices[i] / row_sums.astype(float)

    class_labels = ['Noise', 'NR', 'LTE']
    titles = ['ViT-S70', 'ViT-S75', 'ViT-S80',
              'ViT-M70', 'ViT-M75', 'ViT-M80',
              'ViT-L70', 'ViT-L75', 'ViT-L80']

    plt.rcParams['font.family'] = 'serif'

    # Define a custom gridspec for the subplots
    fig = plt.figure(figsize=(14, 12))  # Increase figure width
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.2])  # Adjust the width ratio of the last column

    # Create subplots based on the gridspec
    axs = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for i, ax in enumerate(axs):
        if (i + 1) % 3 == 0:
            sns.heatmap(conf_matrices[i], annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                        annot_kws={'size': 10})  # Add colorbar here
        else:
            sns.heatmap(conf_matrices[i], annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                        annot_kws={'size': 10}, cbar=False)  # No colorbar for other subplots
        ax.set_title(titles[i], fontsize=16)
        ax.tick_params(axis='both', labelsize=10)

    # Adjust axis labels for the second row, first column (index 3) and third row, second column (index 7)
    axs[3].set_ylabel('True label', fontsize=16)
    axs[7].set_xlabel('Predicted label', fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output_plot, dpi=400)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT models on Segmentation dataset with varying mask ratios.")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--data_dir', type=str, default='../datasets/SegmentationData/test/LTE_NR',
                        help='Path to Segmentation dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--output_plot', type=str, default='fig_conf_matrices_segm.png',
                        help='Path to save the accuracy plot')
    parser.add_argument('--output_accuracy', type=str, default='accuracies_segm.npy',
                        help='Path to save the accuracy array as a .npy file')
    parser.add_argument('--output_conf_mats', type=str, default='conf_mats_segm.npy',
                        help='Path to save the accuracy array as a .npy file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)
