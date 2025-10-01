import os
import numpy as np
from tqdm import tqdm


# Set your data path
min_snr = 10
max_snr = 15
data_path = f'../datasets/channel_estimation_dataset_({min_snr},{max_snr})'
batch_size = 64

# List files in the train and val folders
train_file_list = os.listdir(os.path.join(data_path, 'train'))
val_file_list = os.listdir(os.path.join(data_path, 'val'))

# Create directories for preprocessed data if they don't exist
train_output_dir = os.path.join(data_path, 'train_preprocessed')
val_output_dir = os.path.join(data_path, 'val_preprocessed')
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Process both train and val splits
for split, file_list, output_dir in [('train', train_file_list, train_output_dir),
                                     ('val', val_file_list, val_output_dir)]:
    sample_idx = 0
    for file_name in tqdm(file_list, desc=f'{split} files', total=len(file_list)):
        # Load the file from the corresponding folder
        file_path = os.path.join(data_path, split, file_name)
        data = np.load(file_path)
        # Loop over the batch dimension (each file is assumed to contain 'batch_size' samples)
        for idx in range(batch_size):
            # Create x_model
            x_rg_pilot = np.concatenate((data['x_rg'][idx, 2], data['x_rg'][idx, 11]))
            x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=0)
            y_rg_pilot = np.concatenate((data['y_rg'][idx, :, 2], data['y_rg'][idx, :, 11]), axis=1)
            y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=0)
            x_model = np.concatenate((x_rg_pilot.reshape((2, 1, -1)), y_rg_pilot), axis=1)

            # Create h_freq
            h_freq = np.concatenate([data['h_freq'][idx, :, i] for i in range(14)], axis=1)
            h_freq = np.stack((h_freq.real, h_freq.imag), axis=0)

            # Get snr_db
            snr_db = data['snr_db'][idx]
            if split == 'train':
                assert min_snr <= snr_db <= max_snr
            elif split == 'val':
                assert -10 <= snr_db <= 20
            # Save the arrays to an .npz file.
            # The file name combines the original file name (without extension) and the sample index.
            base_name = os.path.splitext(file_name)[0]
            output_file = os.path.join(output_dir, f"sample_{sample_idx}.npz")
            np.savez(output_file, x=x_model, h=h_freq, snr_db=snr_db)
            sample_idx += 1
