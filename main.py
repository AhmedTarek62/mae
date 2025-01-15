import yaml
import argparse
import os 
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from FineTuningArgs import FineTuningArgs
import time
import util.misc as misc
import json
import datetime
from util.misc import report_score
import tasks
import importlib

# Constants
CURRENT_TASKS = ['segmentation', 'csi_sensing', 'channel_estimation', 'signal_identification', 'positioning']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    # 1. read args
    args = parse_arguments()
    task = args.task
    assert task in CURRENT_TASKS, print(f'Invalid task input. Unrecognizable task ({task})!')

    root_dir = os.getcwd()

    config_path = os.path.join(root_dir, 'tasks', task, 'config.yaml')
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    exp_name = config["experiment_name"]
    del config["experiment_name"]

    data_path = config["data_path"]
    assert os.path.isdir(data_path), print(f"Incorrect data_path! ({data_path})")
    print(f"The dataset path provided: {data_path}")

    output_dir = os.path.join(config["output_dir"], exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"The outputs path provided: {output_dir}")

    print(f"==== Loading all the configs..")
    config = FineTuningArgs(**config)

    print(f"==== Setting the device and random seed..")
    device = torch.device(config.device)
    print(f"Device: {device}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    cudnn.benchmark = True # DEVELOPERS:check
    # Import the task files
    # if task == 'segmentation':
    #     from tasks.segmentation.dataset import SegmentationDataset as TaskDataset
    #     import tasks.segmentation.model as TaskModel
    #     from tasks.segmentation.finetuning_engine import train_one_epoch, evaluate
    # elif task == 'csi_sensing':
    #     from tasks.csi_sensing.dataset import CSISensingDataset as TaskDataset
    #     import tasks.csi_sensing.model as TaskModel
    #     from tasks.csi_sensing.finetuning_engine import train_one_epoch, evaluate
    # elif task == 'channel_estimation':
    #     from tasks.channel_estimation.dataset import OfdmChannelEstimation_Dataset as TaskDataset
    #     import tasks.channel_estimation.model as TaskModel
    #     from tasks.channel_estimation.finetuning_engine import train_one_epoch, evaluate
    # elif task == 'positioning':
    #     from tasks.positioning.dataset import PositioningNR_Dataset as TaskDataset
    #     import tasks.positioning.model as TaskModel
    #     from tasks.positioning.finetuning_engine import train_one_epoch, evaluate
    # elif task == 'signal_identification':
    #     from tasks.signal_identification.dataset import SignalIdentificatio_Dataset as TaskDataset
    #     import tasks.signal_identification.model as TaskModel
    #     from tasks.signal_identification.finetuning_engine import train_one_epoch, evaluate
    # else:
    #     # TODO
    #     assert False, print("Replace this line with import statment \
    #                         for your dataset class as TasDataset")
    #     # You can also build your dataset class here in this cell and then change the two following lines accordingly

    TaskDataset = getattr(importlib.import_module(f"tasks.{task}.dataset"), "SegmentationDataset") # TODO
    TaskModel = importlib.import_module(f"tasks.{task}.model")
    train_one_epoch = getattr(importlib.import_module(f"tasks.{task}.finetuning_engine"), "train_one_epoch")
    evaluate = getattr(importlib.import_module(f"tasks.{task}.finetuning_engine"), "evaluate")

    dataset_train = TaskDataset(data_path, split="train")
    dataset_val = TaskDataset(data_path, split="val")

    # For Training dataset
    ## 1. Create the sampling object (Training)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    ## 2. Create the dataloader (Training)
    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_mem,
            drop_last=True,
        )

    # For Valdiation dataset
    ## 1. Create the sampling object (Validation)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    ## 2. Create the dataloader (Validation)
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_mem,
            drop_last=False
        )
    
    # Model
    assert config.base_arch in list(TaskModel.__dict__.keys()),\
        print(f"This model architecture ({config.base_arch}) is not available!")

    if config.task == 'segmentation':
        model = TaskModel.__dict__[config.base_arch]() 

    elif config.task == 'sensing':
        model = TaskModel.__dict__[config.base_arch](global_pool=config.global_pool,
                                                    num_classes=config.num_classes,
                                                    drop_path_rate=config.drop_path)    
    elif config.task == 'signal_identification':
        model = TaskModel.__dict__[config.base_arch](global_pool=config.global_pool,
                                                    num_classes=config.num_classes,
                                                    drop_path_rate=config.drop_path,
                                                    in_chans=1)
    elif config.task == 'positioning':
        scene = "outdoor" # TODO: (DEVELOPERS)
        tanh = False # TODO: (DEVELOPERS)
        model = TaskModel.__dict__[config.base_arch](global_pool=config.global_pool, num_classes=config.num_classes,
                                                drop_path_rate=config.drop_path, tanh=tanh,
                                                in_chans=4 if scene == 'outdoor' else 5)
    elif config.task == 'channel_estimation':
        model = TaskModel.__dict__[config.base_arch]() 

    else:
        # TODO
        assert False, print("Replace this line with import statment \
                            for your model class as task_model")
        # You can also build your model class here in this cell and then change the two following lines accordingly
        #  
    
    # Load the model checkpoint
    print(f"Loading pre-trained checkpoint from: {config.base_model_path} ...")
    msg = model.load_model_checkpoint(checkpoint_path=config.base_model_path)
    print(msg) # TODO- (DEVELOPERS): why In_IncompatibleKeys?

    # Freeze the encoder weights (the base)
    model.freeze_encoder()
    model.to(device) 

    # Check the model's number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))


    # TODO: Feel free to set your own loss function or LR scheduler

    import util.lr_decay as lrd
    from util.misc import NativeScalerWithGradNormCount as NativeScaler
    from timm.loss import LabelSmoothingCrossEntropy

    param_groups = lrd.param_groups_lrd(model, config.weight_decay, layer_decay=config.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr)
    loss_scaler = NativeScaler()

    if config.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.task in ["positioning", "channel_estimation"]:
        from torch.nn import MSELoss
        criterion = MSELoss()

    print(f"criterion selected: {str(criterion)}")

    # TODO
    # assert False, print("Replace this line with import statment \
    #                     for your finetuing engine script with train_one_epoch and evaluate functions")
    # You can also build your functions here in this cell.
    
    print(f"Training for {config.epochs} epochs..")
    start_time = time.time()
    least_val_loss, best_stats_epoch = np.inf, 0

    for epoch in range(config.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            config.clip_grad, None,
            args=config
        )
        if config.output_dir and (epoch % config.save_every == 0):
            misc.save_model(args=config, output_dir=output_dir, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        val_stats = evaluate(data_loader_val, model, criterion, device)
        if val_stats["avg_loss"] < least_val_loss:
            least_val_loss = val_stats["avg_loss"]
            best_stats_epoch = epoch

        if config.output_dir:
            log_stats = {"epoch": epoch,
                        "train_loss": train_stats["avg_loss"],
                        "val_loss": val_stats["avg_loss"],
                        "val_acc": val_stats["avg_acc"]}
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    report = report_score(config, model, dataset_val, least_val_loss, None)
    with open(os.path.join(output_dir, "report.json"), "w") as json_file:
        json.dump(report, json_file)