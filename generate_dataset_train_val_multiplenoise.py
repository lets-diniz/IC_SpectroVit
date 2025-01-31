import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal,stats
import os
import h5py

from data_augmentation import TransientMaker

###Definition:
path_to_original_data = '../simulated_ground_truths.h5'
random_augment = {'amplitude':{'noise_level_base':{'max':20,'min':2},
                                    'noise_level_scan_var':{'max':2,'min':0}},
                                'frequency':{'noise_level_base':{'max':40,'min':2},
                                    'noise_level_scan_var':{'max':2,'min':0}},
                                'phase':{'noise_level_base':{'max':40,'min':2},
                                    'noise_level_scan_var':{'max':2,'min':0}}}
start_train=0
stop_train=3500
augment_idx_train=4
start_val=3500
stop_val=3700
name_dataset_to_be_created = '../dataset_multiplenoise_SGT_train_val.h5'

def get_interval_method_augment(key,random_augment):
    if key in random_augment.keys():
        max = random_augment[key]["noise_level_base"]["max"] + 1
        min = random_augment[key]["noise_level_base"]["min"]

        noise_level_base = np.random.randint(min, max)

        max = random_augment[key]["noise_level_scan_var"]["max"] + 1
        min = random_augment[key]["noise_level_scan_var"]["min"]

        noise_level_scan_var = np.random.randint(min, max)

        return noise_level_base, noise_level_scan_var
    else:
        return None,None

corrupted_fids_train = np.empty(((stop_train-start_train)*augment_idx_train,2048,2,40),dtype='complex128')
ppm_of_corrupted_fids_train = np.empty(((stop_train-start_train)*augment_idx_train,2048))
t_of_corrupted_fids_train = np.empty(((stop_train-start_train)*augment_idx_train,2048))
target_of_corrupted_fids_train = np.empty(((stop_train-start_train)*augment_idx_train,2048),dtype='complex128')

corrupted_fids_val = np.empty((stop_val-start_val,2048,2,40),dtype='complex128')
ppm_of_corrupted_fids_val = np.empty((stop_val-start_val,2048))
t_of_corrupted_fids_val = np.empty((stop_val-start_val,2048))
target_of_corrupted_fids_val = np.empty((stop_val-start_val,2048),dtype='complex128')

stats_ds_base_train = [[],[],[]]
stats_ds_var_train = [[],[],[]]
counter_idx = 0
print('generating train data...')
for i in range(start_train,stop_train):
    for j in range(augment_idx_train):
        with h5py.File(path_to_original_data) as hf:
            fid = hf["ground_truth_fids"][()][i:i+1]
            ppm = hf["ppm"][()][i:i+1]
            t = hf["t"][()][i:i+1]

        noise_amplitude_base, noise_amplitude_var = get_interval_method_augment("amplitude",random_augment)
        noise_frequency_base, noise_frequency_var = get_interval_method_augment("frequency",random_augment)
        noise_phase_base, noise_phase_var = get_interval_method_augment("phase",random_augment)
        stats_ds_base_train[0].append(noise_amplitude_base)
        stats_ds_base_train[1].append(noise_frequency_base)
        stats_ds_base_train[2].append(noise_phase_base)
        stats_ds_var_train[0].append(noise_amplitude_var)
        stats_ds_var_train[1].append(noise_frequency_var)
        stats_ds_var_train[2].append(noise_phase_var)

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
        corrupted_fids_train[augment_idx_train*counter_idx+j,:,:,:] = aug_fids[0,:,:,:]
        ppm_of_corrupted_fids_train[augment_idx_train*counter_idx+j,:] = ppm[0,:]
        t_of_corrupted_fids_train[augment_idx_train*counter_idx+j,:] = t[0,:]

        spectra_gt_fid = np.fft.fftshift(np.fft.fft(fid[0, :, :], n=fid.shape[1], axis=0), axes=0)
        spectra_gt_diff = spectra_gt_fid[:, 1] - spectra_gt_fid[:, 0]
        target_of_corrupted_fids_train[augment_idx_train*counter_idx+j,:] = spectra_gt_diff
    counter_idx=counter_idx+1
    print('done with '+str(counter_idx+1)+'/'+str(stop_train-start_train))

stats_ds_base_val = [[],[],[]]
stats_ds_var_val = [[],[],[]]
print('generating validation data...')
counter_idx=0
for i in range(start_val,stop_val):
    with h5py.File(path_to_original_data) as hf:
        fid = hf["ground_truth_fids"][()][i:i+1]
        ppm = hf["ppm"][()][i:i+1]
        t = hf["t"][()][i:i+1]

    noise_amplitude_base, noise_amplitude_var = get_interval_method_augment("amplitude",random_augment)
    noise_frequency_base, noise_frequency_var = get_interval_method_augment("frequency",random_augment)
    noise_phase_base, noise_phase_var = get_interval_method_augment("phase",random_augment)
    stats_ds_base_val[0].append(noise_amplitude_base)
    stats_ds_base_val[1].append(noise_frequency_base)
    stats_ds_base_val[2].append(noise_phase_base)
    stats_ds_var_val[0].append(noise_amplitude_var)
    stats_ds_var_val[1].append(noise_frequency_var)
    stats_ds_var_val[2].append(noise_phase_var)

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
    corrupted_fids_val[i-start_val,:,:,:] = aug_fids[0,:,:,:]
    ppm_of_corrupted_fids_val[i-start_val,:] = ppm[0,:]
    t_of_corrupted_fids_val[i-start_val,:] = t[0,:]

    spectra_gt_fid = np.fft.fftshift(np.fft.fft(fid[0, :, :], n=fid.shape[1], axis=0), axes=0)
    spectra_gt_diff = spectra_gt_fid[:, 1] - spectra_gt_fid[:, 0]
    target_of_corrupted_fids_val[i-start_val,:] = spectra_gt_diff
    counter_idx=counter_idx+1
    print('done with '+str(counter_idx+1)+'/'+str(stop_val-start_val))

print('STD TRAIN')
print('amp',np.std(stats_ds_base_train[0]))
print('freq',np.std(stats_ds_base_train[1]))
print('phase',np.std(stats_ds_base_train[2]))
print('corr fids',np.std(corrupted_fids_train))
print('STD VAL')
print('amp',np.std(stats_ds_base_val[0]))
print('freq',np.std(stats_ds_base_val[1]))
print('phase',np.std(stats_ds_base_val[2]))
print('corr fids',np.std(corrupted_fids_val))

with h5py.File(name_dataset_to_be_created, 'w') as h5f:
    # Save inputs, targets, and labels into the HDF5 file
    h5f.create_dataset('corrupted_fids', data=np.concatenate((corrupted_fids_train, corrupted_fids_val), axis=0))
    h5f.create_dataset('ppm', data=np.concatenate((ppm_of_corrupted_fids_train, ppm_of_corrupted_fids_val), axis=0))
    h5f.create_dataset('t', data=np.concatenate((t_of_corrupted_fids_train, t_of_corrupted_fids_val), axis=0))
    h5f.create_dataset('spectrum', data=np.concatenate((target_of_corrupted_fids_train, target_of_corrupted_fids_val), axis=0))

with h5py.File(name_dataset_to_be_created, 'r') as h5f:
    print(h5f.keys())
    print(h5f['corrupted_fids'][()].shape)
    print(h5f['ppm'][()].shape)
    print(h5f['spectrum'][()].shape)
    print(h5f['t'][()].shape)