import yaml
import argparse
import os 
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import util.misc as misc
import json
import datetime
from util.misc import report_score
import importlib

from advanced_finetuning.lora import create_lora_model
import util.lr_decay as lrd
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.loss import LabelSmoothingCrossEntropy

# Constants
CURRENT_TASKS = ['segmentation', 'csi_sensing', 'channel_estimation', 'signal_identification', 'positioning']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str)
    parser.add_argument('--model', default='vit_medium_patch16', type=str, metavar='MODEL',
                        help='Name of model to finetune')
    parser.add_argument('--checkpoint', default='/home/elsayedmohammed/vit-models/pretrained_medium_75.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Number of epochs after which checkpoint is saved')
    parser.add_argument('--label', default='exp', help='Experiment Label')

    parser.add_argument('--prefix', action='store_true', help='Whether to use Prefix Tuning (default: False)')
    parser.add_argument('--ePrefix', action='store_true', help='Whether to use E-Prefix Tuning (default: False)')
    
    # LoRA Arguments    
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRa (default: False)')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank of LoRa (default: 8)')
    parser.add_argument('--lora_alpha', type=float, default=1, help='Alpha for LoRa (default: 0.5)')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--outpath', default='/home/elsayedmohammed/elsayed/finetuning_results2',
                        help='Path of the output directory')

    return parser.parse_args()

if __name__ == "__main__":
    # 1. read args
    args = parse_arguments()
    task = args.task
    assert task in CURRENT_TASKS, print(f'Invalid task input. Unrecognizable task ({task})!')
    
    prefix_type = ''
    if args.prefix:
        prefix_type = '_prefix'
    elif args.ePrefix:
        prefix_type = '_elsayedPrefix'
    model_type = 'model' + prefix_type
    lora_label = f'_lora_({args.lora_rank}-{args.lora_alpha})' if args.lora else ''

    root_dir = os.getcwd()

    ## Load config
    config_path = os.path.join(root_dir, 'tasks', task, 'config.yaml')
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)    
    model_params = config['model_params']
    optimizer_params = config['optimizer_params']

    ## Load dataset
    data_path = config["data_path"]
    assert os.path.isdir(data_path), print(f"Incorrect data_path! ({data_path})")
    print(f"The dataset path provided: {data_path}")
    TaskDataset = getattr(importlib.import_module(f"tasks.{task}.dataset"), "TaskDataset")
    dataset_train = TaskDataset(data_path, split="train")
    dataset_val = TaskDataset(data_path, split="val")
    # For Training dataset
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_mem'],
            drop_last=True,
        )
    # For Valdiation dataset
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
             batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_mem'],
            drop_last=False
        )
    
    coords = None
    if task == 'positioning':
        coords = {"min": dataset_val.coord_nominal_min.view((1, -1)),
                  "max": dataset_val.coord_nominal_max.view((1, -1))}
    ## Set output directory
    output_dir = os.path.join(args.outpath, args.task)
    print(f"The outputs path: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, args.label + prefix_type +lora_label +".txt")
    with open(log_file, "w") as f:  
        f.write("Epoch\tTrainLoss\tLoss\tAccuracy\n")  # Write header

    # Set Seed and Device
    print(f"==== Setting the device and random seed..")
    device = torch.device(args.device)
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True # DEVELOPERS:check
    
    ## Load Model
    TaskModel = importlib.import_module(f"tasks.{task}.{model_type}")
    assert args.model in list(TaskModel.__dict__.keys()),\
        print(f"This model architecture ({args.model}) is not available!")
    print(f"model_params: {model_params}")
    model = TaskModel.__dict__[args.model](**model_params)
    if args.lora:
        model = create_lora_model(model, args.lora_rank, args.lora_alpha)
    # Load the model checkpoint
    print(f"Loading pre-trained checkpoint from: {args.checkpoint} ...")
    msg = model.load_model_checkpoint(checkpoint_path=args.checkpoint)
    if not args.lora:
        model.freeze_encoder()
    model.to(device) 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    print("The unfrozen layers are:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    # Fine-tuning
    train_one_epoch = getattr(importlib.import_module(f"tasks.{task}.finetuning_engine"), "train_one_epoch")
    evaluate = getattr(importlib.import_module(f"tasks.{task}.finetuning_engine"), "evaluate")

    param_groups = lrd.param_groups_lrd(model, optimizer_params['weight_decay'], 
                                        layer_decay=optimizer_params['layer_decay'])
    if optimizer_params['lr'] is None:  # only base_lr is specified
        lr = optimizer_params['blr'] * config['batch_size'] / 256
        optimizer_params['lr'] = lr
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    loss_scaler = NativeScaler()
    
    if args.task in ["positioning", "channel_estimation"]:
        from torch.nn import MSELoss
        criterion = MSELoss()
    else:
        if optimizer_params['smoothing'] > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=optimizer_params['smoothing'])
        else:
            criterion = torch.nn.CrossEntropyLoss()

    print(f"Training for {args.epochs} epochs..")
    start_time = time.time()
    least_val_loss, best_stats_epoch = np.inf, 0
    n_epochs = args.epochs

    for epoch in range(n_epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, n_epochs, loss_scaler, lr, optimizer_params
        )            

        if coords:
            val_stats = evaluate(data_loader_val, model, criterion, device, coords)
        else:
            val_stats = evaluate(data_loader_val, model, criterion, device)
        if val_stats["avg_loss"] < least_val_loss:
            least_val_loss = val_stats["avg_loss"]
            best_stats_epoch = epoch
            # misc.save_model(args=config, output_dir=output_dir, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            misc.save_best_chechpoint(args=config, output_dir=output_dir, model=model, epoch=epoch)

        if task == 'positioning':
            avg_acc = np.mean(val_stats['distances'])
        else:
            avg_acc = val_stats['avg_acc']

        if not avg_acc:
            avg_acc = 0
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{train_stats['avg_loss']:.4f}\t{val_stats['avg_loss']:.4f}\t{avg_acc:.2f}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))