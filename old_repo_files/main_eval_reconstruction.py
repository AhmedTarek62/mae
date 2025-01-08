import argparse
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from dataset_classes.spectrogram_images import SpectrogramImages
from torch.utils.data import DataLoader
import models_mae
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def main(args):
    device = args.device
    ckpt_dir = Path(args.ckpt_dir)
    data_dir = Path(args.data_dir)
    mask_ratios = args.mask_ratios
    models = [("mae_vit_small_patch16", "pretrained_small_%d.pth"),
              ("mae_vit_medium_patch16", "pretrained_medium_%d.pth"),
              ("mae_vit_large_patch16", "pretrained_large_%d.pth")]
    labels = ['ViT-S', 'ViT-M', 'ViT-L']
    kernel_size = args.kernel_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    transform_train = transforms.Compose([
        transforms.functional.pil_to_tensor,
        transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
        transforms.Lambda(lambda x: (x + 155.8) / (-8.41 + 155.8)),
        transforms.Resize((224, 224), antialias=True,
                          interpolation=transforms.InterpolationMode.BICUBIC),  # Resize
        transforms.Normalize(mean=[0.634], std=[0.0664])  # Normalize
    ])

    test_set = SpectrogramImages(data_dir, transform=transform_train)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    accuracies = torch.zeros((len(models), len(mask_ratios)), dtype=torch.float)

    with torch.no_grad():
        for i, (model_key, model_name) in enumerate(tqdm(models, desc="Models")):
            for j, mask_ratio in enumerate(tqdm(mask_ratios, desc=f"Mask Ratios for {model_key}", leave=False)):
                ckpt_path = os.path.join(ckpt_dir, model_name % mask_ratio)
                pretrained = torch.load(ckpt_path, map_location=device)['model']
                model = getattr(models_mae, model_key)()
                model.load_state_dict(pretrained, strict=False)
                model = model.to(device)

                for k, (images, _) in enumerate(tqdm(test_loader, desc="Batches", leave=False)):
                    images = images.to(device)
                    loss, reconstructed, mask = model(images, mask_ratio=mask_ratio / 100)
                    images = torch.einsum('nchw->nhwc', images)
                    reconstructed = torch.einsum('nchw->nhwc', model.unpatchify(reconstructed))
                    mask = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, 16 ** 2 * 1))
                    mask = torch.einsum('nchw->nhwc', mask)
                    reconstructed = (1 - mask) * images + mask * reconstructed

                    pooled_images = F.avg_pool2d(images.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=kernel_size)
                    pooled_reconstructed = F.avg_pool2d(reconstructed.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=kernel_size)
                    mu, std = torch.mean(pooled_images, dim=(1, 2, 3)), torch.std(pooled_images, dim=(1, 2, 3))
                    threshold = mu + 0.5 * std
                    threshold = threshold.view(-1, 1, 1, 1).repeat((1, 1, pooled_images.shape[2], pooled_images.shape[3]))
                    pooled_images = pooled_images > threshold
                    pooled_reconstructed = pooled_reconstructed > threshold

                    if args.plot and k % 10 == 0:
                        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                        fig.suptitle(labels[i], fontsize=16)
                        axs[0].imshow(images[0, :, :, 0].detach().cpu().numpy())
                        axs[0].axis('off')
                        axs[0].set_title('Original')
                        axs[1].imshow((images * (1 - mask))[0, :, :, 0].detach().cpu().numpy())
                        axs[1].axis('off')
                        axs[1].set_title('Masked')
                        axs[2].imshow(reconstructed[0, :, :, 0].detach().cpu().numpy())
                        axs[2].axis('off')
                        axs[2].set_title('Reconstructed + Visible')
                        plt.tight_layout()
                        plt.show()
                        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                        fig.suptitle(labels[i], fontsize=16)
                        axs[0].imshow(pooled_images[0, 0].detach().cpu().numpy())
                        axs[0].axis('off')
                        axs[0].set_title('Original Grid')
                        axs[1].imshow(pooled_reconstructed[0, 0].detach().cpu().numpy())
                        axs[1].axis('off')
                        axs[1].set_title('Reconstructed Grid')
                        plt.tight_layout()
                        plt.show()

                    accuracies[i, j] += (pooled_images == pooled_reconstructed).sum().item()

    accuracies /= (len(test_set) * (224 // kernel_size) ** 2)

    # Save the accuracy array
    np.save(args.output_accuracy, accuracies.cpu().numpy())

    # Plot accuracy vs mask_ratio for each model
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']  # Different markers for each model
    colors = ['r', 'b', 'm']
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
    parser = argparse.ArgumentParser(description="Evaluate MAE models on spectrogram dataset with varying mask ratios.")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--data_dir', type=str, default='../datasets/spectrogram_dataset/images',
                        help='Path to spectrogram dataset directory')
    parser.add_argument('--mask_ratios', type=int, nargs='+', default=[20, 30, 40, 50, 60, 65, 70, 75, 80, 85],
                        help='List of mask ratios to test')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for pooling')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--plot', action='store_true', help='Flag to enable plotting of images')
    parser.add_argument('--output_plot', type=str, default='accuracy_vs_mask_ratio.png',
                        help='Path to save the accuracy plot')
    parser.add_argument('--output_accuracy', type=str, default='accuracies.npy',
                        help='Path to save the accuracy array as a .npy file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)
