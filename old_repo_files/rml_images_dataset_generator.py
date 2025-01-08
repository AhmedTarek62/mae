import pickle
from pathlib import Path
from scipy.signal import ShortTimeFFT
from scipy.signal import windows
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

data_path = Path('../datasets/amc_rml16')
os.makedirs(data_path, exist_ok=True)
mfft = 1024
hop = 128
win_size = 256
win = windows.hann(win_size)
SFT = ShortTimeFFT(win=win, hop=hop, fs=1, scale_to='magnitude', fft_mode='centered', mfft=mfft)
num_slices = 50
num_samples = 2000 // num_slices

file_path = Path('../datasets/RML/RML2016.10a_dict.pkl')
with open(file_path, "rb") as file:
    data = pickle.load(file, encoding="latin1")

for i, (label, sig) in tqdm(enumerate(data.items()), total=len(data), desc="Processing signals"):
    modulation, snr_db = label
    sig = sig[:, 0, :] + 1j * sig[:, 1, :]
    sig = sig.reshape((-1,))
    sig /= np.sqrt(np.mean(np.abs(sig)**2))
    spectrogram = 10 * np.log10(SFT.spectrogram(sig))[:, :-1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    for j in range(num_slices):
        spectrogram_img = Image.fromarray(spectrogram[:, j * num_samples:(j + 1) * num_samples]).resize((224, 224))
        spectrogram_img.save(os.path.join(data_path, f'{modulation}_{snr_db}_{j}.tiff'))

