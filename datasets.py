"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_augmentation import TransientMaker
from pre_processing import PreProcessing
from torch_snippets import Dataset
from utils import ReadDatasets, get_Hz_ppm_conversion, zero_padding


class DatasetThreeChannelSpectrogram(Dataset):
    def __init__(self, **kargs: dict) -> None:
        self.path_data = kargs["path_data"]
        self.file_list = os.listdir(self.path_data)
        self.evaluation = kargs["evaluation"]
        if not self.evaluation:
            self.random_augment = kargs["random_augment"]

    def __len__(self) -> int:
        return len(self.file_list)

    def _get_interval_method_augment(self, key):
        max = self.random_augment[key]["noise_level_base"]["max"] + 1
        min = self.random_augment[key]["noise_level_base"]["min"]

        noise_level_base = np.random.randint(min, max)

        max = self.random_augment[key]["noise_level_scan_var"]["max"] + 1
        min = self.random_augment[key]["noise_level_scan_var"]["min"]

        noise_level_scan_var = np.random.randint(min, max)

        return noise_level_base, noise_level_scan_var

    def create_FID_noise(self, transients, t):

        tm = TransientMaker(
            np.expand_dims(transients, axis=0), np.expand_dims(t, axis=0)
        )

        noise_level_base, noise_level_scan_var = self._get_interval_method_augment(
            "amplitude"
        )

        tm.add_random_amplitude_noise(
            noise_level_base=noise_level_base, noise_level_scan_var=noise_level_scan_var
        )

        noise_level_base, noise_level_scan_var = self._get_interval_method_augment(
            "frequency"
        )

        tm.add_random_frequency_noise(
            noise_level_base=noise_level_base, noise_level_scan_var=noise_level_scan_var
        )

        noise_level_base, noise_level_scan_var = self._get_interval_method_augment(
            "phase"
        )

        tm.add_random_phase_noise(
            noise_level_base=noise_level_base, noise_level_scan_var=noise_level_scan_var
        )
        fids = tm.fids
        return fids

    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        path_sample = os.path.join(self.path_data, self.file_list[idx])
        filename = os.path.basename(path_sample)

        transients, target_spectrum, ppm, fs, tacq, larmorfreq = (
            ReadDatasets.read_h5_complete(path_sample)
        )
        constant_factor = np.max(np.abs(target_spectrum))
        target_spectrum /= constant_factor

        t = np.arange(0, tacq, 1 / fs)

        if not self.evaluation:
            transients_augment = self.create_FID_noise(transients, t)
            fid_off, fid_on = (
                transients_augment[0, :, 0, :],
                transients_augment[0, :, 1, :],
            )
        else:
            fid_off, fid_on = transients[:, 0, :], transients[:, 1, :]

        spectrogram1 = PreProcessing.spectrogram_channel(
            fid_off=fid_off[:, 0:14],
            fid_on=fid_on[:, 0:14],
            fs=fs,
            larmorfreq=larmorfreq,
        )
        spectrogram2 = PreProcessing.spectrogram_channel(
            fid_off=fid_off[:, 14:27],
            fid_on=fid_on[:, 14:27],
            fs=fs,
            larmorfreq=larmorfreq,
        )
        spectrogram3 = PreProcessing.spectrogram_channel(
            fid_off=fid_off[:, 27:40],
            fid_on=fid_on[:, 27:40],
            fs=fs,
            larmorfreq=larmorfreq,
        )

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        spectrogram1 = torch.from_numpy(spectrogram1.real)

        spectrogram2 = zero_padding(spectrogram2)
        spectrogram2 = spectrogram2[np.newaxis, ...]
        spectrogram2 = torch.from_numpy(spectrogram2.real)

        spectrogram3 = zero_padding(spectrogram3)
        spectrogram3 = spectrogram3[np.newaxis, ...]
        spectrogram3 = torch.from_numpy(spectrogram3.real)

        target_spectrum = torch.from_numpy(target_spectrum)
        ppm = torch.from_numpy(ppm)

        three_channels_spectrogram = torch.concat(
            [spectrogram1, spectrogram2, spectrogram3]
        )

        return (
            three_channels_spectrogram.type(torch.FloatTensor),
            target_spectrum.type(torch.FloatTensor),
            ppm,
            constant_factor,
            filename,
        )


class DatasetSpgramSyntheticData(Dataset):
    def __init__(
        self,
        path_data,
        start,
        end,
        augment,
        fs=None,
        larmorfreq=None,
        linear_shift=None,
    ):

        self.path_data = path_data
        self.start_pos = start
        self.end_pos = end
        self.augment = augment

        with h5py.File(self.path_data) as hf:
            fids = hf["ground_truth_fids"][()][:1]
            ppm = hf["ppm"][()][:1]
            t = hf["t"][()][:1]

        if fs == None:
            dwelltime = t[0, 1] - t[0, 0]
            self.fs = 1 / dwelltime
        else:
            self.fs = fs

        if larmorfreq == None or linear_shift == None:
            a_inv, b = get_Hz_ppm_conversion(
                gt_fids=fids, dwelltime=1 / self.fs, ppm=ppm
            )
        if larmorfreq == None and linear_shift == None:
            self.larmorfreq = a_inv
            self.linear_shift = b
        elif larmorfreq == None and linear_shift != None:
            self.larmorfreq = a_inv
            self.linear_shift = linear_shift
        elif larmorfreq != None and linear_shift == None:
            self.larmorfreq = larmorfreq
            self.linear_shift = b
        else:
            self.larmorfreq = larmorfreq
            self.linear_shift = linear_shift

        if self.augment == True:
            self.idx_data = np.empty(200 * (self.end_pos - self.start_pos), dtype="int")
            for i in range(self.start_pos, self.end_pos):
                for j in range(200):
                    self.idx_data[200 * i + j] = i

        else:
            self.idx_data = np.arange(self.start_pos, self.end_pos)

    def __len__(self) -> int:
        return self.idx_data.shape[0]

    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        filename = self.path_data
        idx_in_file = self.idx_data[idx]
        with h5py.File(self.path_data) as hf:
            fid = hf["ground_truth_fids"][()][idx_in_file : idx_in_file + 1]
            ppm = hf["ppm"][()][idx_in_file : idx_in_file + 1]
            t = hf["t"][()][idx_in_file : idx_in_file + 1]

        transientmkr = TransientMaker(fids=fid, t=t, n_transients=40)
        transientmkr.add_random_amplitude_noise(
            noise_level_base=6, noise_level_scan_var=2
        )
        aug_fids = transientmkr.fids

        spectrogram1 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 0:14],
            fid_on=aug_fids[0, :, 1, 0:14],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
        )

        spectrogram2 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 14:27],
            fid_on=aug_fids[0, :, 1, 14:27],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
        )
        spectrogram3 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 27:40],
            fid_on=aug_fids[0, :, 1, 27:40],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
        )

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        spectrogram1 = torch.from_numpy(np.real(spectrogram1))

        spectrogram2 = zero_padding(spectrogram2)
        spectrogram2 = spectrogram2[np.newaxis, ...]
        spectrogram2 = torch.from_numpy(np.real(spectrogram2))

        spectrogram3 = zero_padding(spectrogram3)
        spectrogram3 = spectrogram3[np.newaxis, ...]
        spectrogram3 = torch.from_numpy(np.real(spectrogram3))
        three_channels_spectrogram = torch.concat(
            [spectrogram1, spectrogram2, spectrogram3]
        )

        spectra_gt_fid = np.fft.fftshift(
            np.fft.fft(fid[0, :, :], n=fid.shape[1], axis=0), axes=0
        )
        spectra_gt_diff = spectra_gt_fid[:, 1] - spectra_gt_fid[:, 0]
        constant_factor = np.max(np.abs(spectra_gt_diff))
        spectra_norm = spectra_gt_diff / constant_factor
        spectra_reordered = np.flip(np.real(spectra_norm))
        target_spectrum = torch.from_numpy(spectra_reordered.copy())

        freq = np.flip(np.fft.fftshift(np.fft.fftfreq(fid.shape[1], d=1 / self.fs)))
        ppm_np = self.linear_shift + freq / self.larmorfreq
        ppm = torch.from_numpy(ppm_np)

        return (
            three_channels_spectrogram.type(torch.FloatTensor),
            target_spectrum.type(torch.FloatTensor),
            ppm,
            constant_factor,
            filename,
        )
