import argparse
from pathlib import Path

import torch
from dataset_classes.csi_sensing import CSISensingDataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import models_vit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from advanced_finetuning.lora import create_lora_model
from advanced_finetuning.prefix import create_prefix_tuning_model


def main(args):

    device = args.device
    ckpt_path = Path(args.ckpt_path)
    model_name = args.model_name
    data_dir = Path(args.data_dir)

    batch_size = args.batch_size
    num_workers = args.num_workers

    test_set = CSISensingDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    with torch.no_grad():
        ckpt = torch.load(ckpt_path, map_location=device)['model']
        model = getattr(models_vit, model_name)(global_pool='token', num_classes=6, num_prefix_tokens=1, prefix_tuning=True)
        # model = create_lora_model(model, lora_rank=40, lora_alpha=1)
        model = create_prefix_tuning_model(model, 'token')
        model.load_state_dict(ckpt, strict=True)

        model = model.to(device)

        all_targets = []
        all_preds = []
        for k, (images, targets) in enumerate(tqdm(test_loader, desc="Batches", leave=False)):
            images = images.to(device)
            targets = targets.to(device)
            pred = model(images)
            pred = torch.argmax(pred, dim=-1)
            all_targets.extend(targets.tolist())
            all_preds.extend(pred.tolist())
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        accuracy = np.sum(all_targets == all_preds) / len(all_targets)

    row_sums = np.sum(conf_matrix, axis=1)
    row_sums[row_sums == 0] = 1
    conf_matrix = conf_matrix / row_sums.astype(float)

    print(f"Confusion Matrix: {conf_matrix}")
    print(f"Accuracy: {accuracy}")
    class_labels = test_set.labels

    plt.rcParams['font.family'] = 'serif'

    # Define a custom gridspec for the subplots
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                annot_kws={'size': 10})
    # Adjust axis labels for the second row, first column (index 3) and third row, second column (index 7)
    ax.set_ylabel('True label', fontsize=16)
    ax.set_xlabel('Predicted label', fontsize=16)
    plt.title("Pretrained on Spect + Prefix Tuning (1 Token)", fontsize=16)
    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    masking = str(ckpt_path).split('_')[-1].split('.')[0]
    #plt.savefig(Path(f'Figures/conf_mat_{model_name}_{masking}_sensing.png'), dpi=400)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT model on CSI sensing dataset.")
    parser.add_argument('--ckpt_path', type=str, help='Path of model to evaluate')
    parser.add_argument('--model_name', type=str, help='Model to evaluate')
    parser.add_argument('--data_dir', type=str, default='../datasets/NTU-Fi_HAR/test',
                        help='Path to CSI Sensing dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)
