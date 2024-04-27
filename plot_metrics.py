"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import numpy as np
import matplotlib.pyplot as plt


class PlotMetrics:
    @staticmethod
    def shape_score_comparison(prediction, target, ppm,
                               fig_name=None):

        prediction = np.real(prediction)
        target = np.real(target)

        gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
        glx_max_ind, glx_min_ind = np.amax(np.where(ppm >= 3.6)), np.amin(np.where(ppm <= 3.9))

        gaba_predict = prediction[gaba_min_ind:gaba_max_ind]
        gaba_predict = (gaba_predict - gaba_predict.min()) / (gaba_predict.max() - gaba_predict.min())

        gaba_ground_truth = target[gaba_min_ind:gaba_max_ind]
        gaba_ground_truth = (gaba_ground_truth - gaba_ground_truth.min()) / (
                gaba_ground_truth.max() - gaba_ground_truth.min())

        gaba_crop_ppm = ppm[gaba_min_ind:gaba_max_ind]

        glx_predict = prediction[glx_min_ind:glx_max_ind]
        glx_predict = (glx_predict - glx_predict.min()) / (glx_predict.max() - glx_predict.min())

        glx_ground_truth = target[glx_min_ind:glx_max_ind]
        glx_ground_truth = (glx_ground_truth - glx_ground_truth.min()) / (
                glx_ground_truth.max() - glx_ground_truth.min())

        glx_crop_ppm = ppm[glx_min_ind:glx_max_ind]

        gaba_corr = np.corrcoef(gaba_predict, gaba_ground_truth)[0, 1]
        glx_corr = np.corrcoef(glx_predict, glx_ground_truth)[0, 1]

        fig, ax = plt.subplots(1, 2, figsize=(17, 5))

        ax[0].plot(gaba_crop_ppm, gaba_ground_truth, label='ground-truth', c='b')
        ax[0].plot(gaba_crop_ppm, gaba_predict, label='reconstruction', c='r')
        ax[0].set_xlabel("ppm")
        ax[0].invert_xaxis()
        ax[0].set_title(f"GABA Peak - Correlation: {gaba_corr:.3f}")
        ax[0].legend()

        ax[1].plot(glx_crop_ppm, glx_ground_truth, label='ground-truth', c='b')
        ax[1].plot(glx_crop_ppm, glx_predict, label='reconstruction', c='r')
        ax[1].set_xlabel("ppm")
        ax[1].invert_xaxis()
        ax[1].set_title(f"GLX Peak - Correlation: {glx_corr:.3f}")
        ax[1].legend()

        if fig_name:
            plt.savefig(fig_name)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def spectra_comparison(prediction, target, ppm,
            fig_name=None):

        prediction = np.real(prediction)
        target = np.real(target)

        min_ppm = 2.5
        max_ppm = 4
        max_ind = np.amax(np.where(ppm >= min_ppm))
        min_ind = np.amin(np.where(ppm <= max_ppm))

        spec_predict = prediction[min_ind:max_ind]
        spec_ground_truth = target[min_ind:max_ind]

        ppm_crop = ppm[min_ind:max_ind]

        max_global = np.max(prediction)
        min_global = np.min(prediction)
        if max_global < np.max(target):
            max_global = np.max(target)
        if min_global > np.min(target):
            min_global = np.min(target)

        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax[0].plot(ppm, prediction, label="reconstruction", color="red")
        ax[0].plot(ppm, target, label="ground-truth", color="blue")

        ax[0].invert_xaxis()
        ax[0].set_xlabel("ppm")
        ax[0].set_yticks([])
        ax[0].set_yticklabels([])
        ax[0].set_ylim(min_global, max_global)
        ax[0].set_title("Spectra Comparison")
        ax[0].fill_between([2.5, 4], min_global, max_global, color="yellow", alpha=0.5)
        ax[0].legend(loc='upper right')

        ax[1].plot(ppm_crop, spec_predict, label="reconstruction", color="red")
        ax[1].plot(ppm_crop, spec_ground_truth, label="ground-truth", color="blue")
        ax[1].fill_between(ppm_crop, spec_predict, spec_ground_truth, color="yellow", alpha=0.5)
        ax[1].invert_xaxis()
        ax[1].set_xlabel("ppm")
        ax[1].set_yticks([])
        ax[1].set_yticklabels([])
        ax[1].set_title("Zoom (2.5ppm-4ppm)")

        plt.legend()
        if fig_name:
            plt.savefig(fig_name)
            plt.close(fig)
        else:
            plt.show()
