import os


class FineTuningArgs:
    """
    A configuration class for setting up fine-tuning experiments.

    This class initializes and manages various parameters required for fine-tuning a model,
    including model paths, training hyperparameters, optimizer settings, and dataset specifications.

    Attributes:
        experiment_
        base_model_path (str): Path to the pre-trained model weights.
        data_path (str): Path to the dataset to be used for fine-tuning.
        task (str): Task title to finetune for. Options: ['segmenation', 'sensing', 'custom']
        num_classes (int): Number of target classes for the model. Defaults to 3.
        output_dir (str): Directory to store output files, such as logs and model checkpoints. Defaults to './'.
        batch_size (int): Batch size for training. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 50.
        input_size (int): Size of input images (height/width). Defaults to 224.
        drop_path (float): Drop path rate for regularization. Defaults to 0.1.
        base_arch (str): Model architecture to be used, must be a valid key in `models_segmentation`. Defaults to 'seg_vit_medium_patch16'.
        smoothing (float): Cross entropy smoothing parameter. Defaults to 0 (no smoothing).
        weight_decay (float): Weight decay for the optimizer. Defaults to 0.05.
        lr (float): Learning rate. Either `lr` or `blr` (base learning rate) must be provided.
        blr (float): Base learning rate for scaling with batch size. Defaults to 1e-3.
        layer_decay (float): Layer-wise learning rate decay factor. Defaults to 0.75.
        min_lr (float): Minimum learning rate during training. Defaults to 1e-6.
        warmup_epoch (int): Number of warmup epochs for learning rate scheduling. Defaults to 5.
        device (str): Device to use for training (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
        seed (int): Random seed for reproducibility. Defaults to 0.
        num_workers (int): Number of data loader workers. Defaults to 10.
        pin_mem (bool): Whether to use pinned memory for data loading. Defaults to True.
    """
    def __init__(self, base_model_path:str, data_path:str, task:str, num_classes:int=3, output_dir:str='./', batch_size:int=64, 
                 epochs:int=50, save_every:int=10, input_size:int=224, drop_path:float=0.1, base_arch:str='seg_vit_medium_patch16', 
                 smoothing:float=0, weight_decay:float=0.05, lr:float=None, blr:float=1e-3, layer_decay:float=0.75, 
                 min_lr:float=1e-6, warmup_epoch:int=5, device:str='cuda', seed:int=0, num_workers:int=10, pin_mem:bool=True):
        
        # Task parameters
        self.task = task
        assert task in ["segmentation", "sensing", "signal_identification", "positioning", "channel_estimation", "custom"], print(f"Incorrect task provided! ({task})")
        print(f"Finetuning on the ({self.task}) task..")

        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_every = save_every

        # Model parameters
        self.base_model_path = base_model_path
        self.base_arch = base_arch
        self.input_size = input_size
        self.drop_path = drop_path

        # Optimizer parameters
        self.clip_grad = None
        self.weight_decay = weight_decay
        assert lr or blr, print('Either lr or blr must have a value!')
        self.lr = lr
        self.blr = blr #base learning rate
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * self.batch_size / 256

        self.layer_decay = layer_decay
        self.min_lr = float(min_lr)
        self.warmup_epochs = warmup_epoch
        self.smoothing = smoothing

        # Dataset parameters
        self.data_path = data_path
        self.num_classes = num_classes
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = device
        self.seed = seed
        self.num_workers = num_workers
        self.pin_mem = pin_mem

        self.accum_iter = 1 # TODO: remove this parameter!
        self.global_pool = 'token' # TODO: Obly in sensing


