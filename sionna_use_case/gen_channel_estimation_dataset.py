import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import sys

tf.get_logger().setLevel('ERROR')
try:
    import sionna as sn
except AttributeError:
    import sionna as sn

from sionna.ofdm import ResourceGrid, ResourceGridMapper
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from channel_estimation_util import estimate_covariance_matrices
from sionna.utils import QAMSource
import os


estimate_cov = False
# system parameters
subcarrier_spacing = 30e3  # Hz
carrier_frequency = 3.5e9  # Hz
speed = 3.  # m/s
fft_size = 12*4   # 4 PRBs
num_ofdm_symbols = 14
num_rx_ant = 16

# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
ut_antenna = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni',  # Omnidirectional antenna pattern
                     carrier_frequency=carrier_frequency)


bs_array = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901',  # 3GPP 38.901 antenna pattern
                      carrier_frequency=carrier_frequency)

qam_source = QAMSource(num_bits_per_symbol=2)
rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=subcarrier_spacing,
                  num_tx=1,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2, 11]
                  )
rg_mapper = ResourceGridMapper(rg)
# 3GPP UMi channel model is considered
channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model='low',
                    ut_array=ut_antenna,
                    bs_array=bs_array,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)
channel = OFDMChannel(channel_model, rg, return_channel=True)
channel_sampler = GenerateOFDMChannel(channel_model, rg)

if estimate_cov:
    sn.config.xla_compat = True  # Enable Sionna's support of XLA
    freq_cov_mat, time_cov_mat, space_cov_mat = (
        estimate_covariance_matrices(
            100, 1000, fft_size, num_ofdm_symbols, num_rx_ant, channel_model, channel_sampler, speed))
    sn.config.xla_compat = False  # Disable Sionna's support of XLA
    np.save('freq_cov_mat.npy', freq_cov_mat)
    np.save('time_cov_mat.npy', time_cov_mat)
    np.save('space_cov_mat.npy', space_cov_mat)

# freq_cov_mat : [fft_size, fft_size]
# time_cov_mat : [num_ofdm_symbols, num_ofdm_symbols]
# space_cov_mat : [num_rx_ant, num_rx_ant]

if not estimate_cov:
    freq_cov_mat = np.load('freq_cov_mat.npy')
    time_cov_mat = np.load('time_cov_mat.npy')
    space_cov_mat = np.load('space_cov_mat.npy')

mode = 'val'
data_dir_path = Path(f'../../datasets/test/{mode}')
os.makedirs(data_dir_path, exist_ok=True)
all_snr_db = np.linspace(-10.0, 20.0, 30)
batch_size = 64
num_it = 25
for i in tqdm(range(num_it), total=num_it, desc='Iteration'):
    x = qam_source([batch_size, 1, 1, rg.num_data_symbols])
    x_rg = rg_mapper(x)
    if mode == 'train':
        snr_db = 5 * np.random.random((batch_size,)) + 10
    else:
        snr_db = np.random.choice(all_snr_db, (batch_size,))
    no = tf.pow(10.0, -snr_db / 10.0)
    topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=speed, max_ut_velocity=speed)
    channel_model.set_topology(*topology)
    y_rg, h_freq = channel((x_rg, no))
    x_rg = np.squeeze(x_rg.numpy())
    y_rg = np.squeeze(y_rg.numpy())
    h_freq = np.squeeze(h_freq.numpy())
    file_path = os.path.join(data_dir_path, f'batch_{i}.npz')
    # Save all variables into a single .npz file
    np.savez(file_path, x_rg=x_rg, y_rg=y_rg, h_freq=h_freq, snr_db=snr_db)
