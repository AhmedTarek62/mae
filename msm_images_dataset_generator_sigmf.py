import os
from datetime import datetime
from scipy.signal import ShortTimeFFT
from scipy.signal import windows
from numpy.random import randint
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import numpy as np
from sigmf import sigmffile
import matplotlib.pyplot as plt


def read_sig_mf(recording_name: str):
    recording = sigmffile.fromfile(recording_name)
    freq_s = recording.get_global_info()[recording.SAMPLE_RATE_KEY]
    if 'ism' in recording_name:
        signal = recording.read_samples(start_index=0, count=1_500_000)
    else:
        signal = recording.read_samples()
    return signal, freq_s


def main(recordings_filepath_list, sentence_duration, mfft, overlap=None, hop=None):
    # dsp params
    if overlap is None:
        overlap = mfft // 2
    if hop is None:
        hop = mfft - overlap

    t_min = int(sentence_duration[0] // 1e-3)
    t_max = int(sentence_duration[1] // 1e-3)

    win = windows.hann(mfft)
    SFT = ShortTimeFFT(win=win, hop=hop, fs=1, scale_to='magnitude', fft_mode='centered', mfft=mfft)
    # dataset params
    # start generation
    os.makedirs('../datasets', exist_ok=True)

    dataset_dir_name = (f'dataset_{int(sentence_duration[0] * 1000)}ms_{int(sentence_duration[1] * 1000)}_ms'
                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(os.path.join('../datasets', dataset_dir_name), exist_ok=True)

    recordings_filenames = list()
    for recordings_path in recordings_filepath_list:
        recordings_filenames.extend([os.path.join(Path(recordings_path), filename)
                                     for filename in os.listdir(recordings_path)])

    recordings_filenames = [filename.split('-data')[0] for filename in recordings_filenames if not filename.endswith('-meta')]
    with (tqdm(recordings_filenames, desc='Generating Data', unit='File') as pbar):
        for file_idx, filename in enumerate(pbar):
            # read recording data
            prefix = filename.split('\\')[-1].split('.')[0]
            print(prefix)
            recording, freq_s = read_sig_mf(filename)

            t = randint(t_min, t_max + 1) * 1e-3
            samples_per_spect = int(t * freq_s)
            num_spects = int(recording.shape[0] // samples_per_spect)
            if 'analog' in prefix:
                num_samples = int(0.05 * num_spects)
            elif 'ism' in prefix:
                num_samples = int(0.5 * num_spects)
            else:
                num_samples = num_spects
            sample_indices = np.random.choice(range(num_spects), num_samples, replace=False)

            for idx, i in enumerate(sample_indices):
                spectrogram = SFT.spectrogram(recording[i * samples_per_spect:(i + 1) * samples_per_spect])
                spectrogram_img = Image.fromarray(spectrogram)
                if np.random.random() < 0.01:
                    plt.imshow(np.array(spectrogram_img.resize((224, 224))))
                    plt.show()
                    plt.close()
                img_path = Path('../datasets/' + dataset_dir_name +
                                f'/{prefix}_fs_{int(freq_s // 1e6)}MHz_{file_idx}_{i}.tiff')
                spectrogram_img.save(img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MSM dataset')
    parser.add_argument('--recordings_filepaths_list', type=str, nargs='+',
                        help='Path to the directory containing raw recordings')
    parser.add_argument('--sentence_duration', nargs='+', default=[16e-3, 16e-3],
                        help='Range of sentence duration')
    parser.add_argument('--mfft', type=int, default=1024,
                        help='Length of the FFT window')
    parser.add_argument('--overlap', type=int, default=None,
                        help='Overlap between consecutive frames')
    parser.add_argument('--hop', type=int, default=None,
                        help='Hop size between consecutive frames')
    args = parser.parse_args()

    main(args.recordings_filepaths_list, args.sentence_duration, args.mfft, args.overlap, args.hop)
