import os
from datetime import datetime
from util.data_loading import read_iqdata
from scipy.signal import ShortTimeFFT
from scipy.signal import windows
from numpy.random import randint
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path


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
    os.makedirs('datasets', exist_ok=True)

    dataset_dir_name = (f'dataset_{int(sentence_duration[0] * 1000)}ms_{int(sentence_duration[1] * 1000)}_ms'
                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(os.path.join('datasets', dataset_dir_name), exist_ok=True)

    recordings_filenames = list()
    for recordings_path in recordings_filepath_list:
        recordings_filenames.extend([os.path.join(Path(recordings_path), filename)
                                     for filename in os.listdir(recordings_path)])

    with (tqdm(recordings_filenames, desc='Generating Data', unit='File') as pbar):
        for file_idx, filename in enumerate(pbar):
            # read recording data
            try:
                complex_sig, metadata = read_iqdata(filename)
            except ValueError:
                continue

            freq_c = metadata['center_freq']
            freq_s = metadata['effective_sample_rate']

            t = randint(t_min, t_max + 1) * 1e-3
            samples_per_spect = int(t * freq_s)
            num_spects = int(complex_sig.shape[0] // samples_per_spect)
            if num_spects == 0:
                continue

            for i in range(num_spects):
                spectrogram = SFT.spectrogram(complex_sig[i * samples_per_spect:(i + 1) * samples_per_spect])
                spectrogram_img = Image.fromarray(spectrogram)
                img_path = Path('datasets/' + dataset_dir_name +
                                f'/fc_{int(freq_c // 1e6)}MHz_fs_{int(freq_s // 1e6)}MHz_{file_idx}_{i}.tiff')
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
