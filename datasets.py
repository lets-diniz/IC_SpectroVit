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
from scipy.signal import ShortTimeFFT
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
        augment_with_noise,
        augment_with_idx_repetition,
        start=None,
        end=None,
        fs=None,
        larmorfreq=None,
        linear_shift=None,
        hop_size=None,
        window_size=None,
        window=None,
        qntty_to_augment_by_idx=None,
        **kwargs_augment_by_noise
    ):

        self.path_data = path_data
        self.augment_with_noise = augment_with_noise
        self.augment_with_idx_repetition = augment_with_idx_repetition
        self.get_item_first_time = True
        self.random_augment = kwargs_augment_by_noise

        with h5py.File(self.path_data) as hf:
            ppm = hf['ppm'][()][:1]
            t = np.empty((1,ppm.shape[-1]))
            hf["t"].read_direct(t,source_sel=np.s_[0:1])
            if self.augment_with_noise is True:
                fids = np.empty((1,ppm.shape[-1],2),dtype='complex128')
                hf["ground_truth_fids"].read_direct(fids,source_sel=np.s_[0:1])
                total_qntty = len(hf["ground_truth_fids"])
            else:
                fids = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
                hf["corrupted_fids"].read_direct(fids,source_sel=np.s_[0:1])
                total_qntty = len(hf["corrupted_fids"])

        if start is not None and end is not None:
            self.start_pos = start
            self.end_pos = end
        elif start is not None and end is None:
            self.start_pos = start
            self.end_pos = total_qntty
        elif start is None and end is not None:
            self.start_pos = 0
            self.end_pos = end
        else:
            self.start_pos = 0
            self.end_pos = total_qntty

        if self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is not None:
            self.qntty_to_augment_by_idx = qntty_to_augment_by_idx
        elif self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is None:
            self.qntty_to_augment_by_idx = 100
        else:
            self.qntty_to_augment_by_idx = None

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

        if hop_size is not None and window_size is not None and window is not None:
            if window.shape[0] == window_size:
                self.SFT = ShortTimeFFT(
                    win=window,
                    hop=hop_size,
                    fs=self.fs,
                    mfft=window_size,
                    scale_to="magnitude",
                    fft_mode="centered",
                )
                self.hop_size = hop_size
                self.window_size = window_size
                self.window = window
            else:
                self.SFT = None
                self.hop_size = None
                self.window_size = None
                self.window = None
        else:
            self.SFT = None
            self.hop_size = None
            self.window_size = None
            self.window = None
        
        if self.augment_with_idx_repetition is True:
            self.idx_data = np.empty(self.qntty_to_augment_by_idx * (self.end_pos - self.start_pos), dtype="int")
            idx_counter = 0
            for i in range(self.start_pos, self.end_pos):
                for j in range(self.qntty_to_augment_by_idx):
                    self.idx_data[self.qntty_to_augment_by_idx * idx_counter + j] = i
                idx_counter=idx_counter+1
        else:
            self.idx_data = np.arange(self.start_pos, self.end_pos)

    def __len__(self) -> int:
        return self.idx_data.shape[0]
    
    def _get_interval_method_augment(self, key):
        if key in self.random_augment.keys():
            max = self.random_augment[key]["noise_level_base"]["max"] + 1
            min = self.random_augment[key]["noise_level_base"]["min"]

            noise_level_base = np.random.randint(min, max)

            max = self.random_augment[key]["noise_level_scan_var"]["max"] + 1
            min = self.random_augment[key]["noise_level_scan_var"]["min"]

            noise_level_scan_var = np.random.randint(min, max)

            return noise_level_base, noise_level_scan_var
        else:
            return None,None

    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        filename = self.path_data
        idx_in_file = self.idx_data[idx]
        noise_amplitude_base, noise_amplitude_var = self._get_interval_method_augment("amplitude")
        noise_frequency_base, noise_frequency_var = self._get_interval_method_augment("frequency")
        noise_phase_base, noise_phase_var = self._get_interval_method_augment("phase")

        with h5py.File(self.path_data) as hf:
            ppm = hf["ppm"][()][idx_in_file : idx_in_file + 1]
            t = np.empty((1,ppm.shape[-1]))
            hf['t'].read_direct(t,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            if self.augment_with_noise is True:
                fid = np.empty((1,ppm.shape[-1],2),dtype='complex128')
                hf["ground_truth_fids"].read_direct(fid,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            else:
                fid = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
                spectrum = np.empty((ppm.shape[-1]),dtype='complex128')
                hf['corrupted_fids'].read_direct(fid,source_sel=np.s_[idx_in_file : idx_in_file + 1])
                hf["spectrum"].read_direct(spectrum,source_sel=np.s_[idx_in_file])

        if self.augment_with_noise is True:
            transientmkr = TransientMaker(fids=fid, t=t, n_transients=40)
            if noise_amplitude_base is not None and noise_amplitude_var is not None:
                transientmkr.add_random_amplitude_noise(noise_level_base=noise_amplitude_base, 
                                                        noise_level_scan_var=noise_amplitude_var)
            if noise_frequency_base is not None and noise_frequency_var is not None:
                transientmkr.add_random_frequency_noise(noise_level_base=noise_frequency_base, 
                                                        noise_level_scan_var=noise_frequency_var)
            if noise_phase_base is not None and noise_phase_var is not None:
                transientmkr.add_random_phase_noise(noise_level_base=noise_phase_base, 
                                                        noise_level_scan_var=noise_phase_var)
            aug_fids = transientmkr.fids
        else:
            aug_fids = fid

        spectrogram1 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 0:14],
            fid_on=aug_fids[0, :, 1, 0:14],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        spectrogram2 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 14:27],
            fid_on=aug_fids[0, :, 1, 14:27],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        spectrogram3 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 27:40],
            fid_on=aug_fids[0, :, 1, 27:40],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        if self.get_item_first_time == True:
            print("Generating Spectrograms of size: ", spectrogram1.shape)

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        if self.get_item_first_time == True:
            print("Zero padded to shape: ", spectrogram1.shape)
            self.get_item_first_time = False
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

        if self.augment_with_noise is True:
            spectra_gt_fid = np.fft.fftshift(
                np.fft.fft(fid[0, :, :], n=fid.shape[1], axis=0), axes=0
            )
            spectrum = spectra_gt_fid[:, 1] - spectra_gt_fid[:, 0]

        constant_factor = np.max(np.abs(spectrum))
        spectra_norm = spectrum / constant_factor
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


class DatasetSpgramSyntheticDataOldSTFT(Dataset):
    def __init__(
        self,
        path_data,
        augment_with_noise,
        augment_with_idx_repetition,
        start=None,
        end=None,
        fs=None,
        larmorfreq=None,
        linear_shift=None,
        hop_size=None,
        window_size=None,
        window=None,
        qntty_to_augment_by_idx=None,
        **kwargs_augment_by_noise
    ):
        

        self.path_data = path_data
        self.augment_with_noise = augment_with_noise
        self.augment_with_idx_repetition = augment_with_idx_repetition
        self.get_item_first_time = True
        self.random_augment = kwargs_augment_by_noise

        with h5py.File(self.path_data) as hf:
            ppm = hf['ppm'][()][:1]
            t = np.empty((1,ppm.shape[-1]))
            hf["t"].read_direct(t,source_sel=np.s_[0:1])
            if self.augment_with_noise is True:
                fids = np.empty((1,ppm.shape[-1],2),dtype='complex128')
                hf["ground_truth_fids"].read_direct(fids,source_sel=np.s_[0:1])
                total_qntty = len(hf["ground_truth_fids"])
            else:
                fids = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
                hf["corrupted_fids"].read_direct(fids,source_sel=np.s_[0:1])
                total_qntty = len(hf["corrupted_fids"])

        if start is not None and end is not None:
            self.start_pos = start
            self.end_pos = end
        elif start is not None and end is None:
            self.start_pos = start
            self.end_pos = total_qntty
        elif start is None and end is not None:
            self.start_pos = 0
            self.end_pos = end
        else:
            self.start_pos = 0
            self.end_pos = total_qntty

        if self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is not None:
            self.qntty_to_augment_by_idx = qntty_to_augment_by_idx
        elif self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is None:
            self.qntty_to_augment_by_idx = 100
        else:
            self.qntty_to_augment_by_idx = None

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

        if hop_size is not None and window_size is not None and window is not None:
            if window.shape[0] == window_size:
                self.hop_size = hop_size
                self.window_size = window_size
                self.window = window
            else:
                self.hop_size = None
                self.window_size = None
                self.window = None
        else:
            self.hop_size = None
            self.window_size = None
            self.window = None

        if self.augment_with_idx_repetition is True:
            self.idx_data = np.empty(self.qntty_to_augment_by_idx * (self.end_pos - self.start_pos), dtype="int")
            idx_counter = 0
            for i in range(self.start_pos, self.end_pos):
                for j in range(self.qntty_to_augment_by_idx):
                    self.idx_data[self.qntty_to_augment_by_idx * idx_counter + j] = i
                idx_counter=idx_counter+1
        else:
            self.idx_data = np.arange(self.start_pos, self.end_pos)


    def __len__(self) -> int:
        return self.idx_data.shape[0]
    
    def _get_interval_method_augment(self, key):
        if key in self.random_augment.keys():
            max = self.random_augment[key]["noise_level_base"]["max"] + 1
            min = self.random_augment[key]["noise_level_base"]["min"]

            noise_level_base = np.random.randint(min, max)

            max = self.random_augment[key]["noise_level_scan_var"]["max"] + 1
            min = self.random_augment[key]["noise_level_scan_var"]["min"]

            noise_level_scan_var = np.random.randint(min, max)

            return noise_level_base, noise_level_scan_var
        else:
            return None,None


    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        filename = self.path_data
        idx_in_file = self.idx_data[idx]
        noise_amplitude_base, noise_amplitude_var = self._get_interval_method_augment("amplitude")
        noise_frequency_base, noise_frequency_var = self._get_interval_method_augment("frequency")
        noise_phase_base, noise_phase_var = self._get_interval_method_augment("phase")

        with h5py.File(self.path_data) as hf:
            ppm = hf["ppm"][()][idx_in_file : idx_in_file + 1]
            t = np.empty((1,ppm.shape[-1]))
            hf['t'].read_direct(t,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            if self.augment_with_noise is True:
                fid = np.empty((1,ppm.shape[-1],2),dtype='complex128')
                hf["ground_truth_fids"].read_direct(fid,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            else:
                fid = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
                spectrum = np.empty((ppm.shape[-1]),dtype='complex128')
                hf['corrupted_fids'].read_direct(fid,source_sel=np.s_[idx_in_file : idx_in_file + 1])
                hf["spectrum"].read_direct(spectrum,source_sel=np.s_[idx_in_file])

        if self.augment_with_noise is True:
            transientmkr = TransientMaker(fids=fid, t=t, n_transients=40)
            if noise_amplitude_base is not None and noise_amplitude_var is not None:
                transientmkr.add_random_amplitude_noise(noise_level_base=noise_amplitude_base, 
                                                        noise_level_scan_var=noise_amplitude_var)
            if noise_frequency_base is not None and noise_frequency_var is not None:
                transientmkr.add_random_frequency_noise(noise_level_base=noise_frequency_base, 
                                                        noise_level_scan_var=noise_frequency_var)
            if noise_phase_base is not None and noise_phase_var is not None:
                transientmkr.add_random_phase_noise(noise_level_base=noise_phase_base, 
                                                        noise_level_scan_var=noise_phase_var)
            aug_fids = transientmkr.fids
        else:
            aug_fids=fid


        spectrogram1 = PreProcessing.spgram_channel_old_STFT(
            fid_off=aug_fids[0, :, 0, 0:14],
            fid_on=aug_fids[0, :, 1, 0:14],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
        )

        spectrogram2 = PreProcessing.spgram_channel_old_STFT(
            fid_off=aug_fids[0, :, 0, 14:27],
            fid_on=aug_fids[0, :, 1, 14:27],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
        )

        spectrogram3 = PreProcessing.spgram_channel_old_STFT(
            fid_off=aug_fids[0, :, 0, 27:40],
            fid_on=aug_fids[0, :, 1, 27:40],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
        )

        if self.get_item_first_time == True:
            print("Generating Spectrograms of size: ", spectrogram1.shape)

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        if self.get_item_first_time == True:
            print("Zero padded to shape: ", spectrogram1.shape)
            self.get_item_first_time = False
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


        if self.augment_with_noise is True:
            spectra_gt_fid = np.fft.fftshift(
                np.fft.fft(fid[0, :, :], n=fid.shape[1], axis=0), axes=0
            )
            spectrum = spectra_gt_fid[:, 1] - spectra_gt_fid[:, 0]

        constant_factor = np.max(np.abs(spectrum))
        spectra_norm = spectrum / constant_factor
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
    

class DatasetSpgramRealData(Dataset):
    def __init__(
        self,
        path_data,
        augment_with_noise,
        augment_with_idx_repetition,
        start=None,
        end=None,
        fs=None,
        larmorfreq=None,
        linear_shift=None,
        hop_size=None,
        window_size=None,
        window=None,
        qntty_to_augment_by_idx=None,
        **kwargs_augment_by_noise
    ):

        self.path_data = path_data
        self.augment_with_noise = augment_with_noise
        self.augment_with_idx_repetition = augment_with_idx_repetition
        self.get_item_first_time = True
        self.random_augment = kwargs_augment_by_noise

        with h5py.File(self.path_data) as hf:
            ppm = hf['ppm'][()][:1]
            t = np.empty((1,ppm.shape[-1]))
            hf["t"].read_direct(t,source_sel=np.s_[0:1])
            fids = hf['transient_fids'][()][:1]
            self.fids_transient_qntty = fids.shape[-1]
            total_qntty = len(hf['transient_fids'])

        if start is not None and end is not None:
            self.start_pos = start
            self.end_pos = end
        elif start is not None and end is None:
            self.start_pos = start
            self.end_pos = total_qntty
        elif start is None and end is not None:
            self.start_pos = 0
            self.end_pos = end
        else:
            self.start_pos = 0
            self.end_pos = total_qntty

        if self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is not None:
            self.qntty_to_augment_by_idx = qntty_to_augment_by_idx
        elif self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is None:
            self.qntty_to_augment_by_idx = 200
        else:
            self.qntty_to_augment_by_idx = None

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

        if hop_size is not None and window_size is not None and window is not None:
            if window.shape[0] == window_size:
                self.SFT = ShortTimeFFT(
                    win=window,
                    hop=hop_size,
                    fs=self.fs,
                    mfft=window_size,
                    scale_to="magnitude",
                    fft_mode="centered",
                )
                self.hop_size = hop_size
                self.window_size = window_size
                self.window = window
            else:
                self.SFT = None
                self.hop_size = None
                self.window_size = None
                self.window = None
        else:
            self.SFT = None
            self.hop_size = None
            self.window_size = None
            self.window = None
        
        if self.augment_with_idx_repetition is True:
            self.idx_data = np.empty(self.qntty_to_augment_by_idx * (self.end_pos - self.start_pos), dtype="int")
            idx_counter = 0
            for i in range(self.start_pos, self.end_pos):
                for j in range(self.qntty_to_augment_by_idx):
                    self.idx_data[self.qntty_to_augment_by_idx * idx_counter + j] = i
                idx_counter=idx_counter+1
        else:
            self.idx_data = np.arange(self.start_pos, self.end_pos)

    def __len__(self) -> int:
        return self.idx_data.shape[0]
    
    def _get_interval_method_augment(self, key):
        if key in self.random_augment.keys():
            max = self.random_augment[key]["noise_level_base"]["max"] + 1
            min = self.random_augment[key]["noise_level_base"]["min"]

            noise_level_base = np.random.randint(min, max)

            max = self.random_augment[key]["noise_level_scan_var"]["max"] + 1
            min = self.random_augment[key]["noise_level_scan_var"]["min"]

            noise_level_scan_var = np.random.randint(min, max)

            return noise_level_base, noise_level_scan_var
        else:
            return None,None
    
    def _modulate_value(self,decimal_value, vector):
        max_value = np.max(np.abs(vector))
        order_of_magnitude = int(np.floor(np.log10(max_value))) if max_value != 0 else 0
        if order_of_magnitude == 0 or order_of_magnitude==1:
            modulated_value = decimal_value
        else:
            modulated_value = decimal_value * 10**(order_of_magnitude-1)
        return modulated_value
        
    def _get_random_transients(self,high):
        idx_transient_samples = np.random.randint(low=0,high=high,size=40)
        return idx_transient_samples

    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        filename = self.path_data
        idx_in_file = self.idx_data[idx]
        noise_amplitude_base, noise_amplitude_var = self._get_interval_method_augment("amplitude")
        noise_frequency_base, noise_frequency_var = self._get_interval_method_augment("frequency")
        noise_phase_base, noise_phase_var = self._get_interval_method_augment("phase")

        with h5py.File(self.path_data) as hf:
            ppm = hf["ppm"][()][idx_in_file]
            t = np.empty((1,ppm.shape[-1]))
            hf['t'].read_direct(t,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            
            aug_fids = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
            spectrum = np.empty((ppm.shape[-1]),dtype='float64')
            hf['target_spectra'].read_direct(spectrum,source_sel=np.s_[idx_in_file])
            if self.fids_transient_qntty > 40:
                fid_aux = np.empty((1,ppm.shape[-1],2,self.fids_transient_qntty),dtype='complex128')
                hf['transient_fids'].read_direct(fid_aux,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            else:
                hf['transient_fids'].read_direct(aug_fids,source_sel=np.s_[idx_in_file : idx_in_file + 1])

        if self.fids_transient_qntty > 40:
            if self.augment_with_idx_repetition is True:
                idx_transient_samples = self._get_random_transients(high=self.fids_transient_qntty)
                for i in range(len(idx_transient_samples)):
                    aug_fids[0,:,:,i] = fid_aux[0,:,:,idx_transient_samples[i]]
            else:
                aug_fids = fid_aux[:,:,:,:40]
        
        if self.augment_with_noise is True:
            transientmkr = TransientMaker(fids=aug_fids, t=t, create_transients=False)
            if noise_amplitude_base is not None and noise_amplitude_var is not None:
                noise_amplitude_base_mod = self._modulate_value(noise_amplitude_base, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                noise_amplitude_var_mod = self._modulate_value(noise_amplitude_var, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                transientmkr.add_random_amplitude_noise(noise_level_base=noise_amplitude_base_mod, 
                                                        noise_level_scan_var=noise_amplitude_var_mod)
            if noise_frequency_base is not None and noise_frequency_var is not None:
                transientmkr.add_random_frequency_noise(noise_level_base=noise_frequency_base, 
                                                        noise_level_scan_var=noise_frequency_var)
            if noise_phase_base is not None and noise_phase_var is not None:
                transientmkr.add_random_phase_noise(noise_level_base=noise_phase_base, 
                                                        noise_level_scan_var=noise_phase_var)
            aug_fids = transientmkr.fids
        
        spectrogram1 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 0:14],
            fid_on=aug_fids[0, :, 1, 0:14],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        spectrogram2 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 14:27],
            fid_on=aug_fids[0, :, 1, 14:27],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        spectrogram3 = PreProcessing.spgram_channel(
            fid_off=aug_fids[0, :, 0, 27:40],
            fid_on=aug_fids[0, :, 1, 27:40],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        if self.get_item_first_time == True:
            print("Generating Spectrograms of size: ", spectrogram1.shape)

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        if self.get_item_first_time == True:
            print("Zero padded to shape: ", spectrogram1.shape)
            self.get_item_first_time = False
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

        constant_factor = np.max(np.abs(spectrum))
        spectra_norm = spectrum / constant_factor
        target_spectrum = torch.from_numpy(np.real(spectra_norm))
        ppm = torch.from_numpy(ppm)

        return (
            three_channels_spectrogram.type(torch.FloatTensor),
            target_spectrum.type(torch.FloatTensor),
            ppm,
            constant_factor,
            filename,
        )


class DatasetSpgramRealDataOldSTFT(Dataset):
    def __init__(
        self,
        path_data,
        augment_with_noise,
        augment_with_idx_repetition,
        start=None,
        end=None,
        fs=None,
        larmorfreq=None,
        linear_shift=None,
        hop_size=None,
        window_size=None,
        window=None,
        qntty_to_augment_by_idx=None,
        **kwargs_augment_by_noise
    ):
        

        self.path_data = path_data
        self.augment_with_noise = augment_with_noise
        self.augment_with_idx_repetition = augment_with_idx_repetition
        self.get_item_first_time = True
        self.random_augment = kwargs_augment_by_noise

        with h5py.File(self.path_data) as hf:
            ppm = hf['ppm'][()][:1]
            t = np.empty((1,ppm.shape[-1]))
            hf["t"].read_direct(t,source_sel=np.s_[0:1])
            fids = hf['transient_fids'][()][:1]
            self.fids_transient_qntty = fids.shape[-1]
            total_qntty = len(hf['transient_fids'])

        if start is not None and end is not None:
            self.start_pos = start
            self.end_pos = end
        elif start is not None and end is None:
            self.start_pos = start
            self.end_pos = total_qntty
        elif start is None and end is not None:
            self.start_pos = 0
            self.end_pos = end
        else:
            self.start_pos = 0
            self.end_pos = total_qntty

        if self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is not None:
            self.qntty_to_augment_by_idx = qntty_to_augment_by_idx
        elif self.augment_with_idx_repetition is True and qntty_to_augment_by_idx is None:
            self.qntty_to_augment_by_idx = 100
        else:
            self.qntty_to_augment_by_idx = None

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

        if hop_size is not None and window_size is not None and window is not None:
            if window.shape[0] == window_size:
                self.hop_size = hop_size
                self.window_size = window_size
                self.window = window
            else:
                self.hop_size = None
                self.window_size = None
                self.window = None
        else:
            self.hop_size = None
            self.window_size = None
            self.window = None

        if self.augment_with_idx_repetition is True:
            self.idx_data = np.empty(self.qntty_to_augment_by_idx * (self.end_pos - self.start_pos), dtype="int")
            idx_counter = 0
            for i in range(self.start_pos, self.end_pos):
                for j in range(self.qntty_to_augment_by_idx):
                    self.idx_data[self.qntty_to_augment_by_idx * idx_counter + j] = i
                idx_counter=idx_counter+1
        else:
            self.idx_data = np.arange(self.start_pos, self.end_pos)


    def __len__(self) -> int:
        return self.idx_data.shape[0]
    
    def _get_interval_method_augment(self, key):
        if key in self.random_augment.keys():
            max = self.random_augment[key]["noise_level_base"]["max"] + 1
            min = self.random_augment[key]["noise_level_base"]["min"]

            noise_level_base = np.random.randint(min, max)

            max = self.random_augment[key]["noise_level_scan_var"]["max"] + 1
            min = self.random_augment[key]["noise_level_scan_var"]["min"]

            noise_level_scan_var = np.random.randint(min, max)

            return noise_level_base, noise_level_scan_var
        else:
            return None,None
        
    def _get_random_transients(self,high):
        idx_transient_samples = np.random.randint(low=0,high=high,size=40)
        return idx_transient_samples
    
    def _modulate_value(self,decimal_value, vector):
        max_value = np.max(np.abs(vector))
        order_of_magnitude = int(np.floor(np.log10(max_value))) if max_value != 0 else 0
        if order_of_magnitude == 0 or order_of_magnitude==1:
            modulated_value = decimal_value
        else:
            modulated_value = decimal_value * 10**(order_of_magnitude-1)
        return modulated_value


    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        filename = self.path_data
        idx_in_file = self.idx_data[idx]
        noise_amplitude_base, noise_amplitude_var = self._get_interval_method_augment("amplitude")
        noise_frequency_base, noise_frequency_var = self._get_interval_method_augment("frequency")
        noise_phase_base, noise_phase_var = self._get_interval_method_augment("phase")

        with h5py.File(self.path_data) as hf:
            ppm = hf["ppm"][()][idx_in_file]
            t = np.empty((1,ppm.shape[-1]))
            hf['t'].read_direct(t,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            
            aug_fids = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
            spectrum = np.empty((ppm.shape[-1]),dtype='float64')
            hf['target_spectra'].read_direct(spectrum,source_sel=np.s_[idx_in_file])
            if self.fids_transient_qntty > 40:
                fid_aux = np.empty((1,ppm.shape[-1],2,self.fids_transient_qntty),dtype='complex128')
                hf['transient_fids'].read_direct(fid_aux,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            else:
                hf['transient_fids'].read_direct(aug_fids,source_sel=np.s_[idx_in_file : idx_in_file + 1])

        if self.fids_transient_qntty > 40:
            if self.augment_with_idx_repetition is True:
                idx_transient_samples = self._get_random_transients(high=self.fids_transient_qntty)
                for i in range(len(idx_transient_samples)):
                    aug_fids[0,:,:,i] = fid_aux[0,:,:,idx_transient_samples[i]]
            else:
                aug_fids = fid_aux[:,:,:,:40]


        if self.augment_with_noise is True:
            transientmkr = TransientMaker(fids=aug_fids, t=t, create_transients=False)
            if noise_amplitude_base is not None and noise_amplitude_var is not None:
                noise_amplitude_base_mod = self._modulate_value(noise_amplitude_base, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                noise_amplitude_var_mod = self._modulate_value(noise_amplitude_var, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                transientmkr.add_random_amplitude_noise(noise_level_base=noise_amplitude_base_mod, 
                                                        noise_level_scan_var=noise_amplitude_var_mod)
            if noise_frequency_base is not None and noise_frequency_var is not None:
                noise_frequency_base_mod = self._modulate_value(noise_frequency_base, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                noise_frequency_var_mod = self._modulate_value(noise_frequency_var, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                transientmkr.add_random_frequency_noise(noise_level_base=noise_frequency_base_mod, 
                                                        noise_level_scan_var=noise_frequency_var_mod)
            if noise_phase_base is not None and noise_phase_var is not None:
                noise_phase_base_mod = self._modulate_value(noise_phase_base, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                noise_phase_var_mod = self._modulate_value(noise_phase_var, np.real(aug_fids[0,int(ppm.shape[0]/4):,:,:]))
                transientmkr.add_random_phase_noise(noise_level_base=noise_phase_base_mod, 
                                                        noise_level_scan_var=noise_phase_var_mod)
            aug_fids = transientmkr.fids


        spectrogram1 = PreProcessing.spgram_channel_old_STFT(
            fid_off=aug_fids[0, :, 0, 0:14],
            fid_on=aug_fids[0, :, 1, 0:14],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
        )

        spectrogram2 = PreProcessing.spgram_channel_old_STFT(
            fid_off=aug_fids[0, :, 0, 14:27],
            fid_on=aug_fids[0, :, 1, 14:27],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
        )

        spectrogram3 = PreProcessing.spgram_channel_old_STFT(
            fid_off=aug_fids[0, :, 0, 27:40],
            fid_on=aug_fids[0, :, 1, 27:40],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
        )

        if self.get_item_first_time == True:
            print("Generating Spectrograms of size: ", spectrogram1.shape)

        spectrogram1 = zero_padding(spectrogram1)
        spectrogram1 = spectrogram1[np.newaxis, ...]
        if self.get_item_first_time == True:
            print("Zero padded to shape: ", spectrogram1.shape)
            self.get_item_first_time = False
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

        constant_factor = np.max(np.abs(spectrum))
        spectra_norm = spectrum / constant_factor
        target_spectrum = torch.from_numpy(np.real(spectra_norm))
        ppm = torch.from_numpy(ppm)

        return (
            three_channels_spectrogram.type(torch.FloatTensor),
            target_spectrum.type(torch.FloatTensor),
            ppm,
            constant_factor,
            filename,
        )
    

class DatasetRealDataSeparateFiles(Dataset):
    def __init__(self,
        path_data,
        augment_with_noise,
        start=None,
        end=None,
        linear_shift=None,
        hop_size=None,
        window_size=None,
        window=None,
        **kwargs_augment_by_noise) -> None:


        self.path_data = path_data
        self.augment_with_noise = augment_with_noise
        self.random_augment = kwargs_augment_by_noise

        file_list = sorted(os.listdir(self.path_data))
        if start is not None and end is not None:
            self.file_list = file_list[start:end]
        elif start is not None and end is None:
            self.file_list = file_list[start:]
        elif start is None and end is not None:
            self.file_list = file_list[:end]
        else:
            self.file_list = file_list

        path_sample = os.path.join(self.path_data, self.file_list[0])
        transients, target_spectrum, ppm, fs, tacq, larmorfreq = (
            ReadDatasets.read_h5_complete(path_sample)
        )
        transients = np.expand_dims(transients, axis=0) 
        ppm = np.expand_dims(ppm,axis=0)

        self.fs = fs

        a_inv, b = get_Hz_ppm_conversion(
                gt_fids=transients, dwelltime=1/self.fs, ppm=ppm
                )
        if (round(a_inv) == round(larmorfreq)) and linear_shift == None:
            self.larmorfreq = larmorfreq
            self.linear_shift = b
        elif (round(a_inv) != round(larmorfreq)) and linear_shift == None:
            raise ValueError('no match between larmorfreq given and calculated. given: '+str(round(larmorfreq))+
                             'calculated: '+str(round(a_inv)))
        elif linear_shift != None:
            self.larmorfreq = larmorfreq
            self.linear_shift = linear_shift
        
        if hop_size is not None and window_size is not None and window is not None:
            if window.shape[0] == window_size:
                self.SFT = ShortTimeFFT(
                    win=window,
                    hop=hop_size,
                    fs=self.fs,
                    mfft=window_size,
                    scale_to="magnitude",
                    fft_mode="centered",
                )
                self.hop_size = hop_size
                self.window_size = window_size
                self.window = window
            else:
                self.SFT = None
                self.hop_size = None
                self.window_size = None
                self.window = None
        else:
            self.SFT = None
            self.hop_size = None
            self.window_size = None
            self.window = None
        

    def __len__(self) -> int:
        return len(self.file_list)

    def _get_interval_method_augment(self, key):
        if key in self.random_augment.keys():
            max = self.random_augment[key]["noise_level_base"]["max"] + 1
            min = self.random_augment[key]["noise_level_base"]["min"]
            noise_level_base = np.random.randint(min, max)
            max = self.random_augment[key]["noise_level_scan_var"]["max"] + 1
            min = self.random_augment[key]["noise_level_scan_var"]["min"]
            noise_level_scan_var = np.random.randint(min, max)
            return noise_level_base, noise_level_scan_var
        else:
            return None,None
        
    def _modulate_value(self,decimal_value, vector):
        max_value = np.max(np.abs(vector))
        order_of_magnitude = int(np.floor(np.log10(max_value))) if max_value != 0 else 0
        if order_of_magnitude == 0 or order_of_magnitude==1:
            modulated_value = decimal_value
        else:
            modulated_value = decimal_value * 10**(order_of_magnitude-1)
        return modulated_value
    

    def __getitem__(
        self, idx: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, float, str):

        path_sample = os.path.join(self.path_data, self.file_list[idx])
        filename = os.path.basename(path_sample)
        noise_amplitude_base, noise_amplitude_var = self._get_interval_method_augment("amplitude")
        noise_frequency_base, noise_frequency_var = self._get_interval_method_augment("frequency")
        noise_phase_base, noise_phase_var = self._get_interval_method_augment("phase")

        transients, target_spectrum, ppm, fs, tacq, larmorfreq = (
            ReadDatasets.read_h5_complete(path_sample)
        )
        transients = np.expand_dims(transients, axis=0) 

        constant_factor = np.max(np.abs(target_spectrum))
        target_spectrum /= constant_factor

        t = np.arange(0, tacq, 1 /self.fs)
        t = np.expand_dims(t, axis=0) 
        if self.augment_with_noise is True:
            transientmkr = TransientMaker(fids=transients, t=t, create_transients=False)
            if noise_amplitude_base is not None and noise_amplitude_var is not None:
                noise_amplitude_base_mod = self._modulate_value(noise_amplitude_base, np.real(transients[0,int(ppm.shape[0]/4):,:,:]))
                noise_amplitude_var_mod = self._modulate_value(noise_amplitude_var, np.real(transients[0,int(ppm.shape[0]/4):,:,:]))
                transientmkr.add_random_amplitude_noise(noise_level_base=noise_amplitude_base_mod, 
                                                        noise_level_scan_var=noise_amplitude_var_mod)
            if noise_frequency_base is not None and noise_frequency_var is not None:
                transientmkr.add_random_frequency_noise(noise_level_base=noise_frequency_base, 
                                                        noise_level_scan_var=noise_frequency_var)
            if noise_phase_base is not None and noise_phase_var is not None:
                transientmkr.add_random_phase_noise(noise_level_base=noise_phase_base, 
                                                        noise_level_scan_var=noise_phase_var)
            transients = transientmkr.fids

        
        spectrogram1 = PreProcessing.spgram_channel(
            fid_off=transients[0,:,0,0:14],
            fid_on=transients[0,:,1,0:14],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        spectrogram2 = PreProcessing.spgram_channel(
            fid_off=transients[0,:,0,14:27],
            fid_on=transients[0,:,1,14:27],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
        )

        spectrogram3 = PreProcessing.spgram_channel(
            fid_off=transients[0,:,0,27:40],
            fid_on=transients[0,:,1,27:40],
            fs=self.fs,
            larmorfreq=self.larmorfreq,
            linear_shift=self.linear_shift,
            hop_size=self.hop_size,
            window_size=self.window_size,
            window=self.window,
            SFT=self.SFT,
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
