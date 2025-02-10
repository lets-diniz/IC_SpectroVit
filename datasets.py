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

#---------------------ORIGINAL SPECTROVIT DATASET---------------------------
"""
Expects to receive path for a folder containing data in separate .h5 files
with:
    - transient array of size (2048,2,40) or (4096,2,40)
    - target spectrum of size (2048)
    - ppm array (2048)
    - fs - sampling frequency - (float)
    - tacq - acquisition time - (float)
    - larmorfreq - larmor frequency (float)
Inputs:
- path_data: path to folder with files
- kargs with keys: evaluation and random_augment - if evaluation False, use properties in random augment
                                                    to augment transients by adding noise (amplitude, frequency and phase)
                    - if random_augment is considered, it should present keys amplitude, frequency and phase, each one
                        containing dicts: {noise_level_base: {max: value, min: value},
                                          noise_level_scan_var: {max: value, min: value}}                       
Returns DS with following structure:
    - three_channels_spectrogram - 3 channel spectrogram (real part) of size (3, 224, 224) - torch.FloatTensor
                                    channels: 1- fid 0:14, 2- fid 14:27, 3- fid 27:40
                                    Spectrograms zero padded
    - target_spectrum - normalization of target spectrum present in the input file (size: 2048) - torch.FloatTensor
    - ppm - comes from ppm array present in the input file (size: 2048)
    - constant_factor - factor of normalization of target spectrum
    - filename - name of h5 file
OBS: Uses OLD VERSION of STFT
OBS: Preprocessing of spectrogram consists of eliminating lines for negative chemical shifts, for this
considers transformation ppm = f/larmorfreq + 4.65. There's however an error in spectrogram reorganization, so
some positive lines are also eleiminated. Normalizes spectrogram by the maximum absolute value. Check
normalized_stft in utils.py for more info.
"""
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

#---------------------DATASETS FOR SYNTHETIC DATA---------------------------
"""
Expects to receive path for unique h5 file
with:
    - N transients of size (2048,2) or (2048,2,40)
    - N ppm arrays of size (2048)
    - N time arrays of size (2048)
    - N target  spectrum arrays of size (2048) IF transients are of size (2048,2,40)
Inputs:
- path_data: path to h5 file
- augment_with_noise: bool, if True transients should be of size (2048,2), we then augment the quantity of transients
                            to have arrays of size (2048,2,40) containing noisy transients derived from the original one
                            these will latter be combined to form a spectrogram
- augment_with_idx_repetition: bool, if True, instead of N transients, we work with N*qntty_to_augment_by_idx by repeating
                               transients index in h5 file
- start: int, if None, we start considering transients from the first in h5 file, if not None, we consider [start:,]
- end: int, if None, we consider until the last transient from the h5 file, if not Nne, we consider [,:end]
- fs: float, if given is the sampling frequency used to define ppm from Hz (for the spectrogram) and in the STFT, if None, we calculate it using time arrays
- larmorfreq: float, if given is used to define ppm from Hz (ppm = f/larmofreq + linear_shift) (for the spectrogram), 
                if None, we calculate it from ppm array
- linear_shift: float, if given is used to define ppm from Hz (ppm = f/larmofreq + linear_shift) (for the spectrogram), 
                if None, we calculate it from ppm array
- hop_size: int, if given is the hop used in STFT (if too small one might find errors due to the size limitation 224x224), 
                if not given is 10 if transients are of size 2048, or 64 if 4096
- window_size: int, if given is the quantity of frequencies considered in the STFT (if too large might find errors due to the size limitation 224x224),
               if not given is 256
- window: if given should be an array of size (window_size) containing the expected window shape, if not given, we use the Hanning window in STFT
- qntty_to_augment_by_idx: if given and augment_with_idx_repetition is True, is the amount of times we repeat the same transient index in DS
- **kwargs_augment_by_noise: dict containing the arguments to define the noise added to transients if augment_with_noise is True,
                                - it should present keys amplitude, frequency and phase, each one
                                    containing dicts: {noise_level_base: {max: value, min: value},
                                                    noise_level_scan_var: {max: value, min: value}}                       
Returns DS with following structure:
    - three_channels_spectrogram - 3 channel spectrogram (real part) of size (3, 224, 224) - torch.FloatTensor
                                    channels: 1- fid 0:14, 2- fid 14:27, 3- fid 27:40
                                    Spectrograms zero padded
    - target_spectrum - comes from target spectrum present in the input file (1,2048) - torch.FloatTensor
    - ppm - comes from ppm array present in the input file (1,2048)
    - constant_factor - factor of normalization of target spectrum
    - filename - name of h5 file
OBS: Uses NEW VERSION of STFT
OBS: Preprocessing of spectrogram consists of eliminating lines for negative chemical shifts, for this
considers transformation ppm = f/larmorfreq + linear_shift. Normalizes spectrogram by the maximum absolute value. Check
get_normalized_spgram in utils.py for more info.
"""
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
        
        #get ppm and fids size, the total amount of transients available in h5 file
        #get ppm array sample
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

        #get larmorfreq and linear_shift if not given (uses ppm array sample)
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

        #prepare STFT class object if all parameters are available
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
        
        #create list of idx of transients considered in file. If we have repetition, we repeat each idx 
        # qntty_to_augment_by_idx times
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
        #get transient with index in h5 that in position idx in eht idx_data list
        filename = self.path_data
        idx_in_file = self.idx_data[idx]
        noise_amplitude_base, noise_amplitude_var = self._get_interval_method_augment("amplitude")
        noise_frequency_base, noise_frequency_var = self._get_interval_method_augment("frequency")
        noise_phase_base, noise_phase_var = self._get_interval_method_augment("phase")

        with h5py.File(self.path_data) as hf:
            ppm = hf["ppm"][()][idx_in_file : idx_in_file + 1]
            t = np.empty((1,ppm.shape[-1]))
            hf['t'].read_direct(t,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            #if we want to add noise: expect transients with size (2048,2)
            if self.augment_with_noise is True:
                fid = np.empty((1,ppm.shape[-1],2),dtype='complex128')
                hf["ground_truth_fids"].read_direct(fid,source_sel=np.s_[idx_in_file : idx_in_file + 1])
            else:
                #else we expect transients with size (2048,2,40)
                fid = np.empty((1,ppm.shape[-1],2,40),dtype='complex128')
                spectrum = np.empty((ppm.shape[-1]),dtype='complex128')
                hf['corrupted_fids'].read_direct(fid,source_sel=np.s_[idx_in_file : idx_in_file + 1])
                hf["spectrum"].read_direct(spectrum,source_sel=np.s_[idx_in_file])

        #if we add noise: our transients that used to be (2048,2) will become (2048,2,40)
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

        #if we add noise to transients, we are also assuming that 
        #we dont have in the dataset the target spectrum
        #so we must create it using the original transient of size (2048,2)
        if self.augment_with_noise is True:
            spectra_gt_fid = np.fft.fftshift(
                np.fft.fft(fid[0, :, :], n=fid.shape[1], axis=0), axes=0
            )
            spectrum = spectra_gt_fid[:, 1] - spectra_gt_fid[:, 0]

        #if we dont add noise to the transients we expect 
        #the target spectrum to be present in the dataset
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
    
"""
Equal to the previous class, however uses the old version of the STFT.
The same used in DatasetThreeChannelSpectrogram.
"""
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
    

#---------------------DATASETS FOR REAL DATA---------------------------
"""
Expects to receive path for unique h5 file
with:
    - N transients of size (2048,2,40) or (2048,2,320)
    - N ppm arrays of size (2048)
    - N time arrays of size (2048)
    - N target spectrum arrays of size (2048)
Inputs:
- path_data: path to h5 file
- augment_with_noise: bool, if True add noise to transients. Amplitude noise intensity is modulated depending on the intensity of values in transients.
- augment_with_idx_repetition: bool, if True, instead of N transients, we work with N*qntty_to_augment_by_idx by repeating
                               transients index in h5 file. Also, if transients are of size (2048,2,320) we randomly select
                               40 index of the third dimension to compose a new transient of size (2048,2,40) for each repetition.
- start: int, if None, we start considering transients from the first in h5 file, if not None, we consider [start:,]
- end: int, if None, we consider until the last transient from the h5 file, if not Nne, we consider [,:end]
- fs: float, if given is the sampling frequency used to define ppm from Hz (for the spectrogram) and in the STFT, if None, we calculate it using time arrays
- larmorfreq: float, if given is used to define ppm from Hz (ppm = f/larmofreq + linear_shift) (for the spectrogram), 
                if None, we calculate it from ppm array
- linear_shift: float, if given is used to define ppm from Hz (ppm = f/larmofreq + linear_shift) (for the spectrogram), 
                if None, we calculate it from ppm array
- hop_size: int, if given is the hop used in STFT (if too small one might find errors due to the size limitation 224x224), 
                if not given is 10 if transients are of size 2048, or 64 if 4096
- window_size: int, if given is the quantity of frequencies considered in the STFT (if too large might find errors due to the size limitation 224x224),
               if not given is 256
- window: if given should be an array of size (window_size) containing the expected window shape, if not given, we use the Hanning window in STFT
- qntty_to_augment_by_idx: if given and augment_with_idx_repetition is True, is the amount of times we repeat the same transient index in DS
- **kwargs_augment_by_noise: dict containing the arguments to define the noise added to transients if augment_with_noise is True,
                                - it should present keys amplitude, frequency and phase, each one
                                    containing dicts: {noise_level_base: {max: value, min: value},
                                                    noise_level_scan_var: {max: value, min: value}} 
                                - amplitude noise should have base and var values in between (0,100) due to noise modulation                      
Returns DS with following structure:
    - three_channels_spectrogram - 3 channel spectrogram (real part) of size (3, 224, 224) - torch.FloatTensor
                                    channels: 1- fid 0:14, 2- fid 14:27, 3- fid 27:40
                                    Spectrograms zero padded
    - target_spectrum - comes from target spectrum present in the input file (1,2048) - torch.FloatTensor
    - ppm - comes from ppm array present in the input file (1,2048)
    - constant_factor - factor of normalization of target spectrum
    - filename - name of h5 file
OBS: Uses NEW VERSION of STFT
OBS: Preprocessing of spectrogram consists of eliminating lines for negative chemical shifts, for this
considers transformation ppm = f/larmorfreq + linear_shift. Normalizes spectrogram by the maximum absolute value. Check
get_normalized_spgram in utils.py for more info.
OBS: Noise modulation is applied because depending on the fabricant, the data may cointain transients with intensity between 0 and 1, 0 and 100, or above 10^5. Modulation
should avoid the need to actively consider the characteristics of the transients when defining the noise values. Also, should allow
datasets with data mixed from different vendors.
"""
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

        #get transient sample for linear_shift and larmofreq calculation
        #get N and if third dimension is 40 or above
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

        #get linear_shift and larmorfreq if not given
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

        #prepare STFT class object if all info need is available
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
        
        #create list of idx of transients considered in file. If we have repetition, we repeat each idx 
        # qntty_to_augment_by_idx times
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
        #modulate noise intensity based on the order of magnitude
        #of transients values
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

        #get transient with index in h5 that in position idx in eht idx_data list
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

        #if each transient is size (2048,2,320) and we want to augment the amount os transients
        #we select 40 random indexes from 0 to 319 to build a transient of size (2048,2,40)
        if self.fids_transient_qntty > 40:
            if self.augment_with_idx_repetition is True:
                idx_transient_samples = self._get_random_transients(high=self.fids_transient_qntty)
                for i in range(len(idx_transient_samples)):
                    aug_fids[0,:,:,i] = fid_aux[0,:,:,idx_transient_samples[i]]
            else:
                aug_fids = fid_aux[:,:,:,:40]

        #adds noise to the transients of size (2048,2,40) -> they keep their dimensions
        #no transient is created, noise is simply added
        #AMPLITUDE noise intensity is defined by parameters given when defining the class and
        #is modulated by the intensity of the values in the transient so the 
        #noise level matches the intensity of the transients
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

        #expects dataset to have target spectrum
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

"""
Equal to the previous class, however uses the old version of the STFT.
The same used in DatasetThreeChannelSpectrogram.
"""
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
    
"""
Expects to receive path for a folder containing data in separate .h5 files
with:
    - transient array of size (2048,2,40) or (4096,2,40)
    - target spectrum of size (2048)
    - ppm array (2048)
    - fs - sampling frequency - (float)
    - tacq - acquisition time - (float)
    - larmorfreq - larmor frequency (float)
Inputs:
- path_data: path to folder with files
- augment_with_noise: bool, if True add noise to transients. Amplitude noise intensity is modulated depending on the intensity of values in transients.
- start: int, if None, we start considering transients from the first file, if not None, we consider from file with index "start"
- end: int, if None, we consider until the last file, if not None, we all files until the one with index "end"
- linear_shift: float, if given is used to define ppm from Hz (ppm = f/larmofreq + linear_shift) (for the spectrogram), 
                if None, we calculate it from ppm array
- hop_size: int, if given is the hop used in STFT (if too small one might find errors due to the size limitation 224x224), 
                if not given is 10 if transients are of size 2048, or 64 if 4096
- window_size: int, if given is the quantity of frequencies considered in the STFT (if too large might find errors due to the size limitation 224x224),
               if not given is 256
- window: if given should be an array of size (window_size) containing the expected window shape, if not given, we use the Hanning window in STFT
- **kwargs_augment_by_noise: dict containing the arguments to define the noise added to transients if augment_with_noise is True,
                                - it should present keys amplitude, frequency and phase, each one
                                    containing dicts: {noise_level_base: {max: value, min: value},
                                                    noise_level_scan_var: {max: value, min: value}} 
                                - amplitude noise should have base and var values in between (0,100) due to noise modulation                      
Returns DS with following structure:
    - three_channels_spectrogram - 3 channel spectrogram (real part) of size (3, 224, 224) - torch.FloatTensor
                                    channels: 1- fid 0:14, 2- fid 14:27, 3- fid 27:40
                                    Spectrograms zero padded
    - target_spectrum - comes from target spectrum present in the input file (1,2048) - torch.FloatTensor
    - ppm - comes from ppm array present in the input file (1,2048)
    - constant_factor - factor of normalization of target spectrum
    - filename - name of h5 file
OBS: Uses NEW VERSION of STFT
OBS: This Dataset is very similar to DatasetThreeChannelSpectrogram, changes are related to noise modulation and to be consistent
with usage of new STFT version from Scipy.
OBS: Preprocessing of spectrogram consists of eliminating lines for negative chemical shifts, for this
considers transformation ppm = f/larmorfreq + linear_shift. Normalizes spectrogram by the maximum absolute value. Check
get_normalized_spgram in utils.py for more info.
OBS: Noise modulation is applied because depending on the fabricant, the data may cointain transients with intensity between 0 and 1, 0 and 100, or above 10^5. Modulation
should avoid the need to actively consider the characteristics of the transients when defining the noise values. Also, should allow
datasets with data mixed from different vendors.
"""
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

        #get transient sample from h5 file
        path_sample = os.path.join(self.path_data, self.file_list[0])
        transients, target_spectrum, ppm, fs, tacq, larmorfreq = (
            ReadDatasets.read_h5_complete(path_sample)
        )
        transients = np.expand_dims(transients, axis=0) 
        ppm = np.expand_dims(ppm,axis=0)

        self.fs = fs

        #get linear_shift and larmorfreq
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
        
        #defines STFT class if all parameters are available
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

        #get transient from file
        transients, target_spectrum, ppm, fs, tacq, larmorfreq = (
            ReadDatasets.read_h5_complete(path_sample)
        )
        transients = np.expand_dims(transients, axis=0) 

        #get target spectrum
        constant_factor = np.max(np.abs(target_spectrum))
        target_spectrum /= constant_factor

        t = np.arange(0, tacq, 1 /self.fs)
        t = np.expand_dims(t, axis=0) 

        #if True, we add noise to the transients. No transient is created.
        #amplitude Noise is modulated by intensity of values in transient
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
