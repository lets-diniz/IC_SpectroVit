"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import numpy as np
import math


class TransientMaker:

    def __init__(self, fids=None, t=None, n_transients=80):
        self.ground_truth_fids=fids
        self.fids = np.expand_dims(self.ground_truth_fids,axis=3).copy()
        self.fids = np.repeat(self.fids,n_transients,axis=3)
        self.t=t

    def add_random_amplitude_noise(self, noise_level_base=10, noise_level_scan_var=3):
        base_noise = np.abs(
            noise_level_base * np.ones(self.fids.shape[0]) + np.random.uniform(low=-noise_level_scan_var,
                                                                               high=noise_level_scan_var,
                                                                               size=self.fids.shape[0]))

        noise_real = np.random.normal(0, base_noise.reshape(-1, 1, 1, 1), size=self.fids.shape)
        noise_imag = 1j * np.random.normal(0, base_noise.reshape(-1, 1, 1, 1), size=self.fids.shape)

        self.fids = self.fids + noise_real + noise_imag

    def add_random_frequency_noise(self, noise_level_base=7, noise_level_scan_var=3):
        base_noise = noise_level_base * np.ones(self.fids.shape[0]) + np.random.uniform(low=-noise_level_scan_var,
                                                                                        high=noise_level_scan_var,
                                                                                        size=self.fids.shape[0])

        noise = np.random.uniform(-base_noise.reshape(-1, 1, 1, 1), base_noise.reshape(-1, 1, 1, 1),
                                  size=(self.fids.shape[0], 1, self.fids.shape[2], self.fids.shape[3]))

        fs = self.t[0, 1] - self.t[0, 0]
        self.t = np.arange(0, self.fids.shape[1] * fs, fs)
        self.fids = self.fids * np.exp(
            1j * self.t.reshape(self.fids.shape[0], self.fids.shape[1], 1, 1) * noise * 2 * math.pi)

    def add_random_phase_noise(self, noise_level_base=5, noise_level_scan_var=3):
        base_noise = noise_level_base * np.ones(self.fids.shape[0]) + np.random.uniform(low=-noise_level_scan_var,
                                                                                        high=noise_level_scan_var,
                                                                                        size=self.fids.shape[0])

        noise = np.random.uniform(-base_noise.reshape(-1, 1, 1, 1), base_noise.reshape(-1, 1, 1, 1),
                                  size=(self.fids.shape[0], 1, self.fids.shape[2], self.fids.shape[3]))

        self.fids = self.fids * np.exp(1j * noise * math.pi / 180)
