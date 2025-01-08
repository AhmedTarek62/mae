from multiprocessing import Pool
import os
import h5py
from pathlib import Path
from scipy.signal import ShortTimeFFT
from scipy.signal import windows
import numpy as np
from PIL import Image
import os


recordings_path = Path('../datasets/synthetic_recordings_amc/')
data_path = Path('../datasets/amc_synthetic')
filenames = os.listdir(recordings_path)
os.makedirs(data_path, exist_ok=True)
mfft = 1024
hop = 256
win_size = mfft
win = windows.hann(win_size)
SFT = ShortTimeFFT(win=win, hop=hop, fs=1, scale_to='magnitude', fft_mode='centered', mfft=mfft)


def process_file(filename):
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


if __name__ == '__main__':
    num_cpus = os.cpu_count()
    print(f'Using {num_cpus} CPUs')
    with Pool(num_cpus) as p:
        p.map(process_file, filenames)
