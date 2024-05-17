"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

from torch_snippets import Dataset
import numpy as np
import torch
import os
from data_augmentation import TransientMaker
from utils import ReadDatasets, zero_padding
from pre_processing import PreProcessing


class DatasetThreeChannelSpectrogram(Dataset):
    def __init__(self, **kargs: dict) -> None:
        self.path_data = kargs['path_data']
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

        tm = TransientMaker(np.expand_dims(transients, axis=0),
                            np.expand_dims(t, axis=0))

        noise_level_base, noise_level_scan_var = self._get_interval_method_augment("amplitude")

        tm.add_random_amplitude_noise(noise_level_base=noise_level_base,
                                      noise_level_scan_var=noise_level_scan_var)

        noise_level_base, noise_level_scan_var = self._get_interval_method_augment("frequency")

        tm.add_random_frequency_noise(noise_level_base=noise_level_base,
                                      noise_level_scan_var=noise_level_scan_var)

        noise_level_base, noise_level_scan_var = self._get_interval_method_augment("phase")

        tm.add_random_phase_noise(noise_level_base=noise_level_base,
                                  noise_level_scan_var=noise_level_scan_var)
        fids = tm.fids
        return fids

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, str):

        path_sample = os.path.join(self.path_data, self.file_list[idx])
        filename = os.path.basename(path_sample)

        transients, target_spectrum, ppm, fs, tacq, larmorfreq = ReadDatasets.read_h5_complete(path_sample)
        target_spectrum /= np.max(np.abs(target_spectrum))

        t = np.arange(0, tacq, 1 / fs)

        if not self.evaluation:
            transients_augment = self.create_FID_noise(transients, t)
            fid_off, fid_on = transients_augment[0, :, 0, :], transients_augment[0, :, 1, :]
        else:
            fid_off, fid_on = transients[:, 0, :], transients[:, 1, :]

        spectrogram1 = PreProcessing.spectrogram_channel(fid_off=fid_off[:, 0:14],
                                                         fid_on=fid_on[:, 0:14],
                                                         fs=fs,
                                                         larmorfreq=larmorfreq)
        spectrogram2 = PreProcessing.spectrogram_channel(fid_off=fid_off[:, 14:27],
                                                         fid_on=fid_on[:, 14:27],
                                                         fs=fs,
                                                         larmorfreq=larmorfreq)
        spectrogram3 = PreProcessing.spectrogram_channel(fid_off=fid_off[:, 27:40],
                                                         fid_on=fid_on[:, 27:40],
                                                         fs=fs,
                                                         larmorfreq=larmorfreq)

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

        three_channels_spectrogram = torch.concat([spectrogram1, spectrogram2, spectrogram3])

        return three_channels_spectrogram.type(torch.FloatTensor), target_spectrum.type(
            torch.FloatTensor), ppm, filename
