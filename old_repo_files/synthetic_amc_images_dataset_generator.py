import h5py
from pathlib import Path
from scipy.signal import ShortTimeFFT
from scipy.signal import windows
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


recordings_path = Path('../datasets/synthetic_recordings_amc/')
data_path = Path('../datasets/amc_synthetic')
filenames = os.listdir(recordings_path)
os.makedirs(data_path, exist_ok=True)
mfft = 1024
hop = 256
win_size = mfft
win = windows.hann(win_size)
SFT = ShortTimeFFT(win=win, hop=hop, fs=1, scale_to='magnitude', fft_mode='centered', mfft=mfft)

for i, filename in tqdm(enumerate(filenames), total=len(filenames), desc="Processing files"):
    with h5py.File(os.path.join(recordings_path, filename), 'r') as f:
        _, modulation, snr = filename.split('_')
        snr = int(snr.split('.')[0][3:])
        frames = np.array(f['frames'])
        frames = np.transpose(frames['real'] + 1j * frames['imag'])
        for j in range(frames.shape[0]):
            sig = frames[j]
            spectrogram = 10 * np.log10(SFT.spectrogram(sig))
            spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
            spectrogram_img = Image.fromarray(spectrogram).resize((224, 224), resample=Image.BICUBIC)
            spectrogram_img.save(os.path.join(data_path, f'{modulation}_{snr}_{j}.tiff'))
            # if np.random.random() < 1e-3 and snr > 5:
            #     fig, ax = plt.subplots(1, 2)
            #     fig.suptitle(f'mod: {modulation} - snr: {snr}')
            #     ax[0].plot(np.real(sig), linewidth=2, color='blue', label='I')
            #     ax[0].plot(np.imag(sig), linewidth=2, color='red', label='Q')
            #     ax[0].set_title('Signal')
            #     ax[0].set_ylabel('Amplitude')
            #     ax[0].set_xlabel('Time')
            #     ax[1].imshow(np.array(spectrogram_img), vmin=0, vmax=1)
            #     ax[1].set_title('Spectrogram')
            #     ax[1].set_xlabel('Time')
            #     ax[1].set_ylabel('Frequency')
            #     plt.tight_layout()
            #     plt.show()
