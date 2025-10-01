import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Set TensorFlow logger level to ignore messages
tf.get_logger().setLevel('ERROR')
try:
    import sionna as sn
except AttributeError:
    import sionna as sn

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray, UMa, RMa
from sionna.utils import QAMSource

from dataset_classes.ofdm_channel_estimation import OfdmChannelEstimation
import models_ofdm_ce


# Helper functions for error metrics
def calculate_mse(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# ----------------------------
# Load deep learning model(s) from checkpoints
# ----------------------------

# Using plain Python lists instead of pandas DataFrame
base_dir = 'checkpoints/ofdm_ce/'
# ckpt_runs = {"baseline": "Run_5", "pretrained_5g": "Run_4", "pretrained_all_data": "Run_4",
#              "pretrained_spect": "Run_5", "pretrained_wifi": "Run_5", "pretrained_wifi_spect": "Run_5"}
ckpt_runs = {"baseline": "Run_5", "pretrained_all_data": "Run_4"}

# Construct checkpoint paths dictionary for channel_estimation.
# The folder structure is assumed to be:
# {base_dir}/{best_run}/channel_estimation_{ckpt}/best_model.pth
checkpoint_paths = {}
for ckpt_name, run in ckpt_runs.items():
    folder_name = f"channel_estimation_{ckpt_name}"
    checkpoint_path = f"{base_dir}/{run}/{folder_name}/best_model.pth"
    checkpoint_paths[ckpt_name] = checkpoint_path

print("Checkpoint paths for channel_estimation:")
for ckpt, path in checkpoint_paths.items():
    print(f"{ckpt}: {path}")

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models from the checkpoints.
# We assume the model type is 'ce_small_patch16' from models_ofdm_ce.
models = {}
for ckpt_name, ckpt_path in checkpoint_paths.items():
    # Instantiate the model
    model = models_ofdm_ce.__dict__['ce_small_patch16']()
    checkpoint_file = Path(ckpt_path)
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device)
    model.eval()
    models[ckpt_name] = model
    print(f"Loaded model for {ckpt_name} from {ckpt_path}")

# ----------------------------
# System parameters
# ----------------------------
subcarrier_spacing = 30e3  # Hz
carrier_frequency = 3.5e9  # Hz
speed = 3.  # m/s
fft_size = 12 * 4  # 4 PRBs
num_ofdm_symbols = 14
num_rx_ant = 16

# Antenna and array definitions for Sionna channel models:
ut_antenna = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni',
                     carrier_frequency=carrier_frequency)

bs_array = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901',
                      carrier_frequency=carrier_frequency)

qam_source = QAMSource(num_bits_per_symbol=2)
rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=subcarrier_spacing,
                  num_tx=1,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2, 11])
rg_mapper = ResourceGridMapper(rg)

# Create channel model and estimators:
channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model='low',
                    ut_array=ut_antenna,
                    bs_array=bs_array,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)
# channel_model = RMa(carrier_frequency=carrier_frequency,
#                     #o2i_model='low',
#                     ut_array=ut_antenna,
#                     bs_array=bs_array,
#                     direction='uplink',
#                     enable_shadow_fading=False,
#                     enable_pathloss=False)
channel = OFDMChannel(channel_model, rg, return_channel=True)
channel_sampler = GenerateOFDMChannel(channel_model, rg)

# Load pre-computed covariance matrices (saved as .npy files)
freq_cov_mat = tf.constant(np.load('sionna_use_case/freq_cov_mat.npy'), tf.complex64)
time_cov_mat = tf.constant(np.load('sionna_use_case/time_cov_mat.npy'), tf.complex64)
space_cov_mat = tf.constant(np.load('sionna_use_case/space_cov_mat.npy'), tf.complex64)

ls_estimator = LSChannelEstimator(rg, interpolation_type='nn')
lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
lmmse_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)

# Load dataset for channel estimation evaluation
dataset = OfdmChannelEstimation(Path('../datasets/channel_estimation_dataset_(5,10)/val_preprocessed'))
all_snr_db = np.arange(-10, 22, 2)
num_snr = len(all_snr_db)

# Initialize arrays to accumulate MSE over several iterations
mse_ls = np.zeros((num_snr,))
mse_lmmse = np.zeros((num_snr,))
# Get model keys from the dictionary for iteration.
model_keys = list(models.keys())
mse_models = np.zeros((len(model_keys), num_snr,))

batch_size = 64
num_it = 10

for i, snr_db in enumerate(all_snr_db):
    tqdm.write(f"SNR = {snr_db}")
    noise_power = tf.pow(10.0, -snr_db / 10.0)
    for _ in tqdm(range(num_it), total=num_it, desc='Iteration'):
        # Generate QAM symbols and map them to the resource grid.
        x = qam_source([batch_size, 1, 1, rg.num_data_symbols])
        x_rg = rg_mapper(x)
        # Generate channel topology (using a single-sector topology)
        topology = gen_single_sector_topology(batch_size, 1, 'umi',
                                              min_ut_velocity=speed, max_ut_velocity=speed)
        channel_model.set_topology(*topology)
        y_rg, h_freq = channel((x_rg, noise_power))
        # LS estimation
        h_ls = np.squeeze(ls_estimator((y_rg, noise_power))[0].numpy())
        # LMMSE estimation
        h_lmmse = np.squeeze(lmmse_estimator((y_rg, noise_power))[0].numpy())
        h_freq = np.squeeze(h_freq.numpy())
        mse_ls[i] += calculate_mse(h_freq, h_ls)
        mse_lmmse[i] += calculate_mse(h_freq, h_lmmse)
        # Prepare sample for deep-learning model (convert resource grid to sample format)
        x_rg_np = np.squeeze(x_rg.numpy())
        y_rg_np = np.squeeze(y_rg.numpy())
        x_model = dataset.create_sample(x_rg_np, y_rg_np).to(device)
        # Concatenate the channel frequency response across OFDM symbols
        h_freq_concat = np.concatenate([h_freq[:, :, idx] for idx in range(num_ofdm_symbols)], axis=-1)
        # Loop over each deep-learning model
        for j, key in enumerate(model_keys):
            model = models[key]
            h_model = model(x_model).detach().cpu().numpy()
            # Assume the model output has shape [batch, 2] representing real and imaginary parts.
            h_model = h_model[:, 0] + 1j * h_model[:, 1]
            mse_models[j, i] += calculate_mse(h_freq_concat, h_model)

# Average the MSE over iterations
mse_ls /= num_it
mse_lmmse /= num_it
for j in range(len(model_keys)):
    mse_models[j, :] /= num_it

# Define legend mapping for deep-learning models
model_names_map = {
    "baseline": "ViT-SL",
    "pretrained_wifi": "ViT-WiFi",
    "pretrained_5g": "ViT-5G",
    "pretrained_wifi_spect": "ViT-RFS/WiFi",
    "pretrained_spect": "ViT-RFS",
    "pretrained_all_data": "ViT-All"
}
# Define colors for each model key (make sure you have enough colors)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
markers = ['o', 's', '*', '^']

# Plotting: Create a figure with two subplots sharing the same axis ranges.
plt.rcParams['font.family'] = 'serif'
plt.style.use('seaborn-v0_8-white')

fig, ax = plt.subplots(figsize=(8, 6))

# Plot conventional estimators (LS and LMMSE)
ax.semilogy(all_snr_db, mse_ls, label='LS', color='#2ca02c',
            linewidth=3, linestyle='--')
ax.semilogy(all_snr_db, mse_lmmse, label='LMMSE', color='#d62728',
            linewidth=3, linestyle='--')

# Plot deep-learning model(s)
for j, key in enumerate(model_keys):
    label = model_names_map.get(key, key)
    ax.semilogy(all_snr_db, mse_models[j], label=label, color=colors[j],
                linewidth=3, marker=markers[j % 4])

# Set axis labels with black font color and appropriate fontsize
ax.set_xlabel('SNR (dB)', fontsize=22, color='black')
ax.set_ylabel('MSE', fontsize=22, color='black')
# Optionally, you can set a title:
# ax.set_title('OFDM Channel Estimation: MSE vs SNR', fontsize=16, color='black')

# Configure tick parameters for both axes with black labels
ax.tick_params(axis='both', labelsize=18, colors='black')

# Update spines to use black color for a consistent look
for spine in ax.spines.values():
    spine.set_color('black')

# Set grid lines to be black, dashed, with a light line width
ax.grid(True, color='black', linestyle='--', linewidth=0.5)

# Create and customize the legend: white background, black border, and black text
legend = ax.legend(loc='lower left', fontsize=18, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
for text in legend.get_texts():
    text.set_color('black')

plt.tight_layout()
# plt.savefig(Path('Figures/fig_best_ofdm_ce.png'), dpi=400)
plt.show()
test = []

