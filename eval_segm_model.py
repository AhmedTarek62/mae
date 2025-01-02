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
    os.makedirs('Figures', exist_ok=True)
    device = args.device
    data_dir = Path(args.data_dir)
    model_name = args.model
    ckpt_path = Path(args.ckpt_path)
    batch_size = args.batch_size
    num_workers = args.num_workers

    test_set = SegmentationDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    with torch.no_grad():
        ckpt = torch.load(ckpt_path, map_location=device)['model']
        model = getattr(models_segmentation, model_name)()
        model.load_state_dict(ckpt, strict=False)

        model = model.to(device)

        all_targets = []
        all_preds = []
        for k, (images, targets) in enumerate(tqdm(test_loader, desc="Batches", leave=False)):
            images = images.to(device)
            targets = targets.to(device)
            pred = model(images)
            pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1]).argmax(dim=-1).detach().cpu().numpy()
            targets = targets.view(-1).detach().cpu().numpy()
            all_targets.extend(targets.tolist())
            all_preds.extend(pred.tolist())
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        conf_mat = confusion_matrix(all_targets, all_preds)
        accuracy = np.sum(all_targets == all_preds) / len(all_targets)


    row_sums = np.sum(conf_mat, axis=1)
    row_sums[row_sums == 0] = 1
    conf_mat = conf_mat / row_sums.astype(float)

    print(f"Confusion Matrix: {conf_mat}")
    print(f"Accuracy: {accuracy}")

    class_labels = ['Noise', 'NR', 'LTE']

    plt.rcParams['font.family'] = 'serif'

    # Define a custom gridspec for the subplots
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))  # Increase figure width

    sns.heatmap(conf_mat, annot=True, fmt='.2f', cmap='Reds', xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={'size': 10}, vmin=0, vmax=1)
    ax.tick_params(axis='both', labelsize=10)

    # Adjust axis labels for the second row, first column (index 3) and third row, second column (index 7)
    ax.set_ylabel('True label', fontsize=16)
    ax.set_xlabel('Predicted label', fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    masking = str(ckpt_path).split('_')[-1].split('.')[0]
    plt.savefig(Path(f'Figures/conf_mat_{model_name}_{masking}_segm.png'), dpi=400)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT models on segmentation dataset.")
    parser.add_argument('--ckpt_path', type=str, help='Path of model to evaluate')
    parser.add_argument('--model', type=str, help='Model to evaluate')
    parser.add_argument('--data_dir', type=str, default='../datasets/SegmentationData/test/LTE_NR',
                        help='Path to Segmentation dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)
