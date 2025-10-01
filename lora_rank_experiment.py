import os

# List of LoRa ranks to try
lora_ranks = [1, 5, 10, 25, 50, 100, 200, 300]

# Loop through each rank and run the training script
for r in lora_ranks:
    output_dir = f"./output_dir/lora_r{r}"

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the training command; update --data_path and other args as needed.
    cmd = (
        f"python main_finetune_sig_identification.py "
        f"--lora --lora_rank {r} "
        f"--output_dir {output_dir} --log_dir {output_dir} "
        f"--data_path ../datasets/radio_sig_identification "  # update this path if necessary
        f"--finetune checkpoints/pretrained_small_75.pth "
        f"--num_workers 0"
    )
    print(f"Running command: {cmd}")
    os.system(cmd)  # This will run the command in the shell
    print(f"Completed experiment with LoRa rank {r}\n")
