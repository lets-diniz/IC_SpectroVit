import numpy as np


def calculate_metrics(x, y, ppm):
    x = np.real(x)
    y = np.real(y)

    mse = []
    snr = []
    linewidth = []
    shape_score = []

    for i in range(x.shape[0]):
        mse.append(calculate_mse(x[i, :], y[i, :], ppm[i, :]))

        snr.append(calculate_snr(x[i, :], ppm[i, :]))

        linewidth.append(calculate_linewidth(x[i, :], ppm[i, :]))

        shape_score.append(calculate_shape_score(x[i, :], y[i, :], ppm[i, :]))

    output = {
        "mse": mse,
        "snr": snr,
        "linewidth": linewidth,
        "shape_score": shape_score
    }

    return output


def calculate_mse(x, y, ppm):
    max_ind = np.amax(np.where(ppm >= 2.5))
    min_ind = np.amin(np.where(ppm <= 4))

    x_crop = x[min_ind:max_ind]
    y_crop = y[min_ind:max_ind]

    x_crop_norm = (x_crop - x_crop.min()) / (x_crop.max() - x_crop.min())
    y_crop_norm = (y_crop - y_crop.min()) / (y_crop.max() - y_crop.min())

    mse = np.square(y_crop_norm - x_crop_norm).mean()

    return mse


def calculate_snr(x, ppm):
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    dt_max_ind, dt_min_ind = np.amax(np.where(ppm >= 10)), np.amin(np.where(ppm <= 12))

    max_peak = x[gaba_min_ind:gaba_max_ind].max()

    dt = np.polyfit(ppm[dt_min_ind:dt_max_ind], x[dt_min_ind:dt_max_ind], 2)
    sizeFreq = ppm[dt_min_ind:dt_max_ind].shape[0]
    stdev_Man = np.sqrt(
        np.sum(np.square(np.real(x[dt_min_ind:dt_max_ind] - np.polyval(dt, ppm[dt_min_ind:dt_max_ind])))) / (
                sizeFreq - 1))

    snr = np.real(max_peak) / (2 * stdev_Man)

    return snr


def calculate_linewidth(x, ppm):
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))

    spec = x[gaba_min_ind:gaba_max_ind]

    spec = (spec - spec.min()) / (spec.max() - spec.min())

    max_peak = spec.max()
    ind_max_peak = np.argmax(spec)

    try:
        left_side = spec[:ind_max_peak]
        left_ind = np.amin(np.where(left_side > max_peak / 2)) + gaba_min_ind
        left_ppm = ppm[left_ind]
    except:
        left_side = spec[:ind_max_peak + 1]
        left_ind = np.amin(np.where(left_side > max_peak / 2)) + gaba_min_ind
        left_ppm = ppm[left_ind]

    right_side = spec[ind_max_peak:]
    right_ind = np.amax(np.where(right_side > max_peak / 2)) + gaba_min_ind + ind_max_peak
    right_ppm = ppm[right_ind]

    linewidth = left_ppm - right_ppm

    return linewidth


def calculate_shape_score(x, y, ppm):
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    glx_max_ind, glx_min_ind = np.amax(np.where(ppm >= 3.6)), np.amin(np.where(ppm <= 3.9))

    gaba_spec_x = x[gaba_min_ind:gaba_max_ind]
    gaba_spec_x = (gaba_spec_x - gaba_spec_x.min()) / (gaba_spec_x.max() - gaba_spec_x.min())

    gaba_spec_y = y[gaba_min_ind:gaba_max_ind]
    gaba_spec_y = (gaba_spec_y - gaba_spec_y.min()) / (gaba_spec_y.max() - gaba_spec_y.min())

    glx_spec_x = x[glx_min_ind:glx_max_ind]
    glx_spec_x = (glx_spec_x - glx_spec_x.min()) / (glx_spec_x.max() - glx_spec_x.min())

    glx_spec_y = y[glx_min_ind:glx_max_ind]
    glx_spec_y = (glx_spec_y - glx_spec_y.min()) / (glx_spec_y.max() - glx_spec_y.min())

    gaba_corr = np.corrcoef(gaba_spec_x, gaba_spec_y)[0, 1]
    glx_corr = np.corrcoef(glx_spec_x, glx_spec_y)[0, 1]

    return (0.6 * gaba_corr + 0.4 * glx_corr)
