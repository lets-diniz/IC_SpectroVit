"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import os

import h5py
import numpy as np
import torch
import yaml
from scipy import signal


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Using {}".format(device))

    return device


def clean_directory(dir_path):
    for file_name in os.listdir(dir_path):
        file_absolute_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_absolute_path):
            os.remove(file_absolute_path)
        elif os.path.isdir(file_absolute_path):
            clean_directory(file_absolute_path)
            os.rmdir(file_absolute_path)


def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


def zero_padding(matrix, output_shape=(224, 224)):
    pad_width = (
        (0, output_shape[0] - matrix.shape[0]),
        (0, output_shape[1] - matrix.shape[1]),
    )
    padded_matrix = np.pad(matrix, pad_width, mode="constant")
    return padded_matrix


def normalized_stft(
    fid, fs, larmorfreq, window_size, hop_size, window="hann", nfft=None
):
    noverlap = window_size - hop_size

    if not signal.check_NOLA(window, window_size, noverlap):
        raise ValueError(
            "signal windowing fails Non-zero Overlap Add (NOLA) criterion; "
            "STFT not invertible"
        )

    f, t, stft_coefficients = signal.stft(
        fid,
        fs=fs,
        nperseg=window_size,
        window=window,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
    )

    f = np.concatenate([np.split(f, 2)[1], np.split(f, 2)[0]])
    ppm = 4.65 + f / larmorfreq

    stft_coefficients_ordered = np.concatenate(
        [np.split(stft_coefficients, 2)[1], np.split(stft_coefficients, 2)[0]]
    )
    stft_coefficients_ordered = np.flip(stft_coefficients_ordered, axis=0)
    stft_coefficients_onesided = stft_coefficients_ordered[(ppm >= 0), :]
    stft_coefficients_onesided_norm = stft_coefficients_onesided / (
        np.max(np.abs(stft_coefficients_onesided))
    )

    return stft_coefficients_onesided_norm, f, ppm, t


def get_normalized_spgram(
    fid, fs, larmorfreq, linear_shift, window_size, hop_size, window, nfft=None,SFT=None
):

    noverlap = window_size - hop_size
    if not signal.check_NOLA(window, window_size, noverlap):
        raise ValueError(
            "signal windowing fails Non-zero Overlap Add (NOLA) criterion; "
            "STFT not invertible"
        )
    if nfft == None:
        nfft = window_size

    if SFT == None:
        SFT = signal.ShortTimeFFT(
            win=window,
            hop=hop_size,
            fs=fs,
            mfft=nfft,
            scale_to="magnitude",
            fft_mode="centered",
        )
    t_lo, t_hi, f_lo, f_hi = SFT.extent(fid.shape[0])

    stft_coefficients = SFT.stft(fid)
    f = np.linspace(f_lo, f_hi, nfft)
    t = np.linspace(t_lo, t_hi, stft_coefficients.shape[1])
    ppm = linear_shift + f / larmorfreq

    zero_idx = np.abs(t - 0.0).argmin()
    one_idx = np.abs(t - 1.0).argmin()
    t = t[zero_idx:one_idx]
    stft_coefficients = stft_coefficients[:, zero_idx:one_idx]

    # opt 1: Gab
    # stft_coefficients_ordered = np.flip(stft_coefficients, axis=0)
    # stft_coefficients_onesided = stft_coefficients_ordered[(ppm >= 0), :]
    # opt 2:
    stft_coefficients_onesided = stft_coefficients[(ppm >= 0), :]
    stft_coefficients_onesided = np.flip(stft_coefficients_onesided, axis=0)
    # opt 3:
    # stft_coefficients_onesided = np.flip(stft_coefficients, axis=0)
    stft_coefficients_onesided_norm = stft_coefficients_onesided / (
        np.max(np.abs(stft_coefficients_onesided))
    )

    return stft_coefficients_onesided_norm


class ReadDatasets:
    @staticmethod
    def read_h5_complete(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float64, np.float64, np.float64]:
        with h5py.File(filename) as hf:
            transients = hf["transient_specs"][()]
            target_spectrum = hf["target_spectra"][()]
            ppm = hf["ppm"][()]
            fs = hf["fs"][()]
            tacq = hf["tacq"][()]
            larmorfreq = hf["larmorfreq"][()]

        return transients, target_spectrum, ppm, fs, tacq, larmorfreq

    @staticmethod
    def read_h5_sample_track_1(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            transients = hf["ground_truth_fids"][()]

        return transients, ppm, t

    @staticmethod
    def read_h5_sample_track_2(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            transients = hf["transient_fids"][()]

        return transients, ppm, t

    def read_h5_sample_track_3(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            input_ppm_down = hf["data_2048"]["ppm"][()]
            input_t_down = hf["data_2048"]["t"][()]
            input_transients_down = hf["data_2048"]["transient_fids"][()]

            input_ppm_up = hf["data_4096"]["ppm"][()]
            input_t_up = hf["data_4096"]["t"][()]
            input_transients_up = hf["data_4096"]["transient_fids"][()]

        return (
            input_transients_down,
            input_ppm_down,
            input_t_down,
            input_transients_up,
            input_ppm_up,
            input_t_up,
        )

    @staticmethod
    def write_h5_track1_predict_submission(
        filename: str, spectra_predict: np.ndarray, ppm: np.ndarray
    ):
        with h5py.File(filename, "w") as hf:
            hf.create_dataset(
                "result_spectra",
                spectra_predict.shape,
                dtype=float,
                data=spectra_predict,
            )
            hf.create_dataset("ppm", ppm.shape, dtype=float, data=ppm)

    @staticmethod
    def write_h5_track2_predict_submission(
        filename: str, spectra_predict: np.ndarray, ppm: np.ndarray
    ):
        with h5py.File(filename, "w") as hf:
            hf.create_dataset(
                "result_spectra",
                spectra_predict.shape,
                dtype=float,
                data=spectra_predict,
            )
            hf.create_dataset("ppm", ppm.shape, dtype=float, data=ppm)

    @staticmethod
    def write_h5_track3_predict_submission(
        filename: str,
        spectra_predict_down: np.ndarray,
        ppm_down: np.ndarray,
        spectra_predict_up: np.ndarray,
        ppm_up: np.ndarray,
    ):
        with h5py.File(filename, "w") as hf:
            hf.create_dataset(
                "result_spectra_2048",
                spectra_predict_down.shape,
                dtype=float,
                data=spectra_predict_down,
            )
            hf.create_dataset("ppm_2048", ppm_down.shape, dtype=float, data=ppm_down)

            hf.create_dataset(
                "result_spectra_4096",
                spectra_predict_up.shape,
                dtype=float,
                data=spectra_predict_up,
            )
            hf.create_dataset("ppm_4096", ppm_up.shape, dtype=float, data=ppm_up)


def get_Hz_ppm_conversion(gt_fids, dwelltime, ppm):

    # note: if we use np.fft.fftshift(np.fft.ifft()) to get the spectrum, if we use np.flip(np.fft.fftshift(np.fft.fftfreq())) the frequency values
    # will start from the most positive to the most negative, as in the ppm axis. Then freq[i] will have its correspondent ppm value in ppm[i]
    # and the spectrum in the frequency freq[i] will be obtainted by spectrum[0,i]
    # however, this is by using the inverse fourier transform to get the spectrum, and we want to use the Fourier transform
    # for this we do np.fft.fftshift(np.fft.fft()) to get the spectrum, but still we flip the frequency axis as np.flip(np.fft.fftshift(np.fft.fftfreq())),
    # so the freq[i] will have its correspondent ppm value in ppm[i], but the spectrum at a given frequency f will occur at
    # spectrum[0,np.abs(np.flip(freq)-f).argmin()], i.e. the frequency axis is flipped with respect to the spectrum, and the ppm axis as well
    # this also happens for the spectrogram...
    # gts
    spectra_gt_fids = np.fft.fftshift(
        np.fft.fft(gt_fids, n=gt_fids.shape[1], axis=1), axes=1
    )
    spectra_gt_diff = spectra_gt_fids[:, :, 1] - spectra_gt_fids[:, :, 0]
    freq = np.flip(np.fft.fftshift(np.fft.fftfreq(gt_fids.shape[1], d=dwelltime)))
    # to get ppm axis
    idx_min = np.real(spectra_gt_diff[0, :]).argmin()
    idx_max = np.real(spectra_gt_diff[0, :]).argmax()
    # p = a*f + b
    # not really necessary to flip to find a and b since freq and ppm agree, but we use it to enphasize the flip
    # and also so np.flip(ppm)[0,idx_max] = 4.7 ppm, np.flip(ppm)[0,idx_min] = 2 ppm
    # and np.flip(freq)[idx_max] = 216 Hz, and np.flip(freq)[idx_min] = -127 Hz, which agree with visual observation
    a = (np.flip(ppm)[0, idx_max] - np.flip(ppm)[0, idx_min]) / (
        np.flip(freq)[idx_max] - np.flip(freq)[idx_min]
    )
    b = np.flip(ppm)[0, idx_max] - a * np.flip(freq)[idx_max]
    # ppm_aux = b + freq*a

    if a == 0:
        raise ValueError("Larmor frequency can't be infinit.")
    else:
        return (1 / a), b
