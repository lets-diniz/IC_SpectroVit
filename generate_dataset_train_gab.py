import numpy as np
import random
import os
import h5py
import glob as glob

def get_philips_scans(file_names):
    phi_files = []
    for file in file_names:
        if 'full_p' in file:
            phi_files.append(file)
    return phi_files

def get_random_transients():
        idx_transient_samples = np.random.randint(low=0,high=160,size=40)
        return idx_transient_samples

file_names_train = sorted(glob.glob(os.path.join('/home/leticia/Documentos/SpectroViT/full_320_transient_h5_samples', 'train', '*.h5')))
phi_files_train = get_philips_scans(file_names_train)

file_names_val = sorted(glob.glob(os.path.join('/home/leticia/Documentos/SpectroViT/full_320_transient_h5_samples', 'val', '*.h5')))
phi_files_val = get_philips_scans(file_names_val)

file_names_test = sorted(glob.glob(os.path.join('/home/leticia/Documentos/SpectroViT/full_320_transient_h5_samples', 'test', '*.h5')))
phi_files_test = get_philips_scans(file_names_test)

print(len(phi_files_train))
print(len(phi_files_val))
print(len(phi_files_test))

name_dataset_to_be_created_train = '../dataset_SpectVit_Paper_training_data.h5'
#data:
ppm_array_trainval = np.empty((len(phi_files_train)+len(phi_files_val),2048))
t_array_trainval = np.empty((len(phi_files_train)+len(phi_files_val),2048))
target_array_trainval = np.empty((len(phi_files_train)+len(phi_files_val),2048))
fids_array_trainval = np.empty((len(phi_files_train)+len(phi_files_val),2048,2,160),dtype='complex128')

for i in range(len(phi_files_train)):
    with h5py.File(phi_files_train[i], 'r') as h5f:
        ppm_array_trainval[i,:] = h5f['ppm'][()]
        tacq = h5f['tacq'][()]
        fs = h5f['fs'][()]
        t_array_trainval[i,:] = np.arange(0, tacq, 1 / fs)
        target_array_trainval[i,:] = h5f['target_spectra'][()]
        fids_array_trainval[i,:] = h5f['transient_specs'][()]
for i in range(len(phi_files_val)):
    with h5py.File(phi_files_val[i], 'r') as h5f:
        ppm_array_trainval[i+len(phi_files_train),:] = h5f['ppm'][()]
        tacq = h5f['tacq'][()]
        fs = h5f['fs'][()]
        t_array_trainval[i+len(phi_files_train),:] = np.arange(0, tacq, 1 / fs)
        target_array_trainval[i+len(phi_files_train),:] = h5f['target_spectra'][()]
        fids_array_trainval[i+len(phi_files_train),:] = h5f['transient_specs'][()]


with h5py.File(name_dataset_to_be_created_train, 'w') as h5f:
    h5f.create_dataset('transient_fids', data=fids_array_trainval)
    h5f.create_dataset('ppm', data=ppm_array_trainval)
    h5f.create_dataset('t', data=t_array_trainval)
    h5f.create_dataset('target_spectra', data=target_array_trainval)
#check
print('train dataset:')
with h5py.File(name_dataset_to_be_created_train, 'r') as h5f:
    print(h5f.keys())
    print(h5f['transient_fids'][()].shape)
    print(h5f['ppm'][()].shape)
    print(h5f['target_spectra'][()].shape)
    print(h5f['t'][()].shape)


name_dataset_to_be_created_test = '../dataset_SpectVit_Paper_test_data.h5'
qntty_to_repeat = 50
start=0
stop=12
#data:
ppm_array_test_aux = np.empty((12,2048))
t_array_test_aux = np.empty((12,2048))
target_array_test_aux = np.empty((12,2048))
fids_array_test_aux = np.empty((12,2048,2,160),dtype='complex128')

for i in range(len(phi_files_test)):
    with h5py.File(phi_files_test[i], 'r') as h5f:
        ppm_array_test_aux[i,:] = h5f['ppm'][()]
        tacq = h5f['tacq'][()]
        fs = h5f['fs'][()]
        t_array_test_aux[i,:] = np.arange(0, tacq, 1 / fs)
        target_array_test_aux[i,:] = h5f['target_spectra'][()]
        fids_array_test_aux[i,:] = h5f['transient_specs'][()]

corrupted_fids = np.empty(((stop-start)*qntty_to_repeat,2048,2,40),dtype='complex128')
ppm_of_corrupted_fids = np.empty(((stop-start)*qntty_to_repeat,2048))
t_of_corrupted_fids = np.empty(((stop-start)*qntty_to_repeat,2048))
target_of_corrupted_fids = np.empty(((stop-start)*qntty_to_repeat,2048))

counter=0
for j in range(qntty_to_repeat):
    for i in range(start,stop):
        ppm = ppm_array_test_aux[i:i+1,:]
        t = t_array_test_aux[i:i+1,:]
        target = target_array_test_aux[i:i+1,:]
        fid = fids_array_test_aux[i:i+1,:,:,:]
    
        idx_transient_samples = get_random_transients()
        for k in range(len(idx_transient_samples)):
            corrupted_fids[counter,:,:,k] = fid[0,:,:,idx_transient_samples[k]]
        ppm_of_corrupted_fids[counter,:]=ppm[0,:]
        t_of_corrupted_fids[counter,:]=t[0,:]
        target_of_corrupted_fids[counter,:]=target[0,:]
        counter=counter+1


with h5py.File(name_dataset_to_be_created_test, 'w') as h5f:
    h5f.create_dataset('transient_fids', data=corrupted_fids)
    h5f.create_dataset('ppm', data=ppm_of_corrupted_fids)
    h5f.create_dataset('t', data=t_of_corrupted_fids)
    h5f.create_dataset('target_spectra', data=target_of_corrupted_fids)

print('test dataset:')
with h5py.File(name_dataset_to_be_created_test, 'r') as h5f:
    print(h5f.keys())
    print(h5f['transient_fids'][()].shape)
    print(h5f['ppm'][()].shape)
    print(h5f['target_spectra'][()].shape)
    print(h5f['t'][()].shape)





