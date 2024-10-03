"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from utils import get_normalized_spgram, normalized_stft


class PreProcessing:
    @staticmethod
    def spectrogram_channel(
        fid_off: np.ndarray, fid_on: np.ndarray, fs: np.float64, larmorfreq: np.float64
    ) -> np.ndarray:

        fid_diff = fid_on - fid_off
        fid_result = np.mean(fid_diff, axis=1)

        if fid_result.shape[0] == 2048:
            hop_size = 10
        elif fid_result.shape[0] == 4096:
            hop_size = 64

        spectrogram = normalized_stft(
            fid=fid_result,
            fs=fs,
            larmorfreq=larmorfreq,
            window_size=256,
            hop_size=hop_size,
        )

        return spectrogram

    def spgram_channel(fid_off, fid_on, fs, larmorfreq, linear_shift, hop_size=None, window_size=None, window=None, SFT=None) -> np.ndarray:

        fid_diff = fid_on - fid_off
        fid_result = np.mean(fid_diff, axis=1)
        
        if SFT == None:
            if fid_result.shape[0] == 2048:
                hop_size = 10
            elif fid_result.shape[0] == 4096:
                hop_size = 64        
            
            window_size=256
            window = hann(window_size, sym=True)
            SFT = ShortTimeFFT(
                win=window,
                hop=hop_size,
                fs=fs,
                mfft=window_size,
                scale_to="magnitude",
                fft_mode="centered",
            )

        spectrogram = get_normalized_spgram(
            fid=fid_result,
            fs=fs,
            larmorfreq=larmorfreq,
            linear_shift=linear_shift,
            window_size=window_size,
            hop_size=hop_size,
            window=window,
            SFT=SFT
        )

        return spectrogram
