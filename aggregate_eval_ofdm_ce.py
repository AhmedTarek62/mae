import os
import numpy as np
from pathlib import Path
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom modules
import models_ofdm_ce
from dataset_classes.ofdm_channel_estimation import OfdmChannelEstimation

# Set TensorFlow logger level to ignore messages
tf.get_logger().setLevel('ERROR')
try:
    import sionna as sn
except AttributeError:
    import sionna as sn

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.utils import QAMSource

# ----------------------------
# Configuration
# ----------------------------

# Valid DL checkpoints and their display alias.
VALID_CKPTS = [
    "pretrained_all_data", "pretrained_wifi", "pretrained_wifi_spect",
    "pretrained_spect", "pretrained_5g", "baseline"
]
alias = {
    "pretrained_all_data": "ViT-All",
    "pretrained_wifi": "ViT-WiFi",
    "pretrained_wifi_spect": "ViT-RFS/WiFi",
    "pretrained_spect": "ViT-RFS",
    "pretrained_5g": "ViT-5G",
    "baseline": "ViT-SL",
    "LS": "LS Estimator",
    "LMMSE": "LMMSE Estimator"
}

# List of all methods to evaluate: DL checkpoints plus the conventional estimators.
methods_list = VALID_CKPTS + ["LS", "LMMSE"]

# Directory that contains subdirectories of the form Run_{i}
experiment_dir = 'checkpoints/ofdm_ce/'  # Update this path accordingly

# Evaluation parameters
subcarrier_spacing = 30e3  # Hz
carrier_frequency = 3.5e9  # Hz
speed = 3.0              # m/s
fft_size = 12 * 4        # 4 PRBs
num_ofdm_symbols = 14
batch_size = 16
num_it = 5
all_snr_db = np.arange(-10, 22, 2)  # e.g., -10 to 20 dB

# ----------------------------
# Sionna Setup: Antenna, array, channel, and estimators
# ----------------------------

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

channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model='low',
                    ut_array=ut_antenna,
                    bs_array=bs_array,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)
channel = OFDMChannel(channel_model, rg, return_channel=True)
channel_sampler = GenerateOFDMChannel(channel_model, rg)

freq_cov_mat = tf.constant(np.load('sionna_use_case/freq_cov_mat.npy'), tf.complex64)
time_cov_mat = tf.constant(np.load('sionna_use_case/time_cov_mat.npy'), tf.complex64)
space_cov_mat = tf.constant(np.load('sionna_use_case/space_cov_mat.npy'), tf.complex64)

ls_estimator = LSChannelEstimator(rg, interpolation_type='nn')
lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
lmmse_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)

# Dataset for channel estimation evaluation
dataset = OfdmChannelEstimation(Path('../datasets/channel_estimation_dataset_(5,10)/val_preprocessed'))

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Helper function for error metric
# ----------------------------
def calculate_mse(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) ** 2)

# ----------------------------
# Sample Generation Function
# ----------------------------
def generate_samples():
    """
    Generates and returns a list of samples.
    Each sample is a dictionary containing:
        - x_rg: TF tensor from resource grid mapping.
        - y_rg: TF tensor output from channel.
        - x_rg_np, y_rg_np: numpy arrays (for the DL model).
        - h_freq_concat: ground truth channel (concatenated across OFDM symbols).
        - noise_power: the noise power used.
    """
    samples = []
    for snr_db in all_snr_db:
        noise_power = tf.pow(10.0, -snr_db / 10.0)
        for _ in range(num_it):
            # Generate QAM symbols and map them.
            x = qam_source([batch_size, 1, 1, rg.num_data_symbols])
            x_rg = rg_mapper(x)
            # Generate channel topology (using a single-sector topology)
            topology = gen_single_sector_topology(batch_size, 1, 'umi',
                                                  min_ut_velocity=speed, max_ut_velocity=speed)
            channel_model.set_topology(*topology)
            y_rg, h_freq = channel((x_rg, noise_power))
            h_freq = np.squeeze(h_freq.numpy())
            # Concatenate channel frequency responses across OFDM symbols.
            h_freq_concat = np.concatenate([h_freq[:, :, idx] for idx in range(num_ofdm_symbols)], axis=-1)
            sample = {
                "x_rg": x_rg,
                "y_rg": y_rg,
                "x_rg_np": np.squeeze(x_rg.numpy()),
                "y_rg_np": np.squeeze(y_rg.numpy()),
                "h_freq_concat": h_freq_concat,
                "noise_power": noise_power
            }
            samples.append(sample)
    return samples

# ----------------------------
# Evaluation on Pre-generated Samples
# ----------------------------
def evaluate_on_samples(samples, mode, model=None):
    """
    Evaluates a method on a list of pre-generated samples.
    mode: "LS", "LMMSE", or "DL".
    For DL, a loaded model must be provided.
    Returns the mean MSE error.
    """
    mse_total = 0.0
    count = 0
    for sample in samples:
        noise_power = sample["noise_power"]
        h_freq_concat = sample["h_freq_concat"]
        if mode == "LS":
            h_est = np.squeeze(ls_estimator((sample["y_rg"], noise_power))[0].numpy())
            h_est = h_est.reshape((16, 16, -1))  # adjust shape as needed
        elif mode == "LMMSE":
            h_est = np.squeeze(lmmse_estimator((sample["y_rg"], noise_power))[0].numpy())
            h_est = h_est.reshape((16, 16, -1))
        elif mode == "DL":
            x_model = dataset.create_sample(sample["x_rg_np"], sample["y_rg_np"]).to(device)
            h_model = model(x_model).detach().cpu().numpy()
            h_est = h_model[:, 0] + 1j * h_model[:, 1]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        mse = calculate_mse(h_freq_concat, h_est)
        mse_total += mse
        count += 1
    return mse_total / count if count > 0 else np.nan

# ----------------------------
# Aggregate results across runs and methods using pre-generated samples
# ----------------------------

# We will store DL results per run.
results = {method: [] for method in VALID_CKPTS}

# Loop over run directories for DL checkpoints only.
for run_dir in sorted(os.listdir(experiment_dir)):
    if run_dir.startswith("Run_"):
        run_path = os.path.join(experiment_dir, run_dir)
        print(f"\nProcessing {run_dir} ...")
        # Generate samples once for the current run.
        samples = generate_samples()
        # Evaluate each DL checkpoint using the same samples.
        for ckpt in VALID_CKPTS:
            ckpt_dir = os.path.join(run_path, f"channel_estimation_{ckpt}")
            ckpt_file = os.path.join(ckpt_dir, "best_model.pth")
            if os.path.exists(ckpt_file):
                print(f"Evaluating DL checkpoint: {ckpt_file}")
                model = models_ofdm_ce.__dict__['ce_small_patch16']()
                checkpoint = torch.load(Path(ckpt_file), map_location='cpu')
                model.load_state_dict(checkpoint['model'], strict=True)
                model = model.to(device)
                model.eval()
                error = evaluate_on_samples(samples, "DL", model=model)
                results[ckpt].append((run_dir, error))
            else:
                print(f"Checkpoint not found: {ckpt_file}")

# Evaluate LS and LMMSE only once using a common set of samples.
print("\nEvaluating LS estimator (single evaluation)")
samples_conv = generate_samples()
error_ls = evaluate_on_samples(samples_conv, "LS")
results["LS"] = [("SingleEvaluation", error_ls)]
print("Evaluating LMMSE estimator (single evaluation)")
error_lmmse = evaluate_on_samples(samples_conv, "LMMSE")
results["LMMSE"] = [("SingleEvaluation", error_lmmse)]

# ----------------------------
# Compute aggregated results: mean error, standard deviation, and best run per method
# ----------------------------
print("\nAggregated Evaluation Results:")
aggregated = {}
for method, run_results in results.items():
    if run_results:
        errors = np.array([err for (_, err) in run_results])
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        best_run, best_error = min(run_results, key=lambda x: x[1])
        aggregated[method] = (mean_error, std_error)
        print(f"{alias[method]}: Mean Error = {mean_error:.4f} ± {std_error:.4f}, Best Run = {best_run} with Error = {best_error:.4f}")
    else:
        print(f"No results for method: {method}")

# ----------------------------
# (Optional) Plotting the aggregated error per method with error bars
# ----------------------------
method_names = [alias[m] for m in methods_list if m in aggregated]
mean_errors = [aggregated[m][0] for m in methods_list if m in aggregated]
std_errors = [aggregated[m][1] for m in methods_list if m in aggregated]

plt.figure(figsize=(10, 6))
x_pos = np.arange(len(method_names))
plt.bar(x_pos, mean_errors, yerr=std_errors, capsize=5)
plt.xticks(x_pos, method_names)
plt.xlabel('Estimator / Model Checkpoint')
plt.ylabel('MSE')
plt.title('Aggregated OFDM Channel Estimation Error')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(Path('Figures/aggregated_error.png'), dpi=300)
plt.show()
test = []