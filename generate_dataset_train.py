import numpy as np
import random
import os
import h5py

def get_random_transients():
        idx_transient_samples = np.random.randint(low=0,high=160,size=40)
        return idx_transient_samples

with h5py.File('../track_02_training_data.h5', 'r') as h5f:
    print(h5f.keys())
    ppm_tr2 = h5f['ppm'][()]
    t_tr2 = h5f['t'][()]
    target_tr2 = h5f['target_spectra'][()]
    fids_tr2 = h5f['transient_fids'][()]

with h5py.File('../track_03_training_data.h5', 'r') as h5f:
    print(h5f.keys())
    ppm_tr3 = h5f['data_2048']['ppm'][()]
    t_tr3 = h5f['data_2048']['t'][()]
    target_tr3 = h5f['data_2048']['target_spectra'][()]
    fids_tr3 = h5f['data_2048']['transient_fids'][()]

#----train----
#definitions:
name_dataset_to_be_created_train = '../dataset_tracks2and3_training_data.h5'
#data:
ppm_array_trainval = np.empty((19,2048))
t_array_trainval = np.empty((19,2048))
target_array_trainval = np.empty((19,2048))
fids_array_trainval = np.empty((19,2048,2,160),dtype='complex128')
idx_tr2 = [0,1,2,3,4,6,8,10,5,7]
idx_tr3 = [0,1,2,4,6,8,10,3,5]
count_idx_tr2 = 0
count_idx_tr3 = 0
for i in range(ppm_array_trainval.shape[0]):
    if i%2 == 0:
        ppm_array_trainval[i,:] = ppm_tr2[idx_tr2[count_idx_tr2],:]
        t_array_trainval[i,:] = t_tr2[idx_tr2[count_idx_tr2],:]
        target_array_trainval[i,:] = target_tr2[idx_tr2[count_idx_tr2],:]
        fids_array_trainval[i,:,:,:] = fids_tr2[idx_tr2[count_idx_tr2],:,:,:]
        #print('tr2:',idx_tr2[count_idx_tr2])
        count_idx_tr2 = count_idx_tr2 + 1
    else:
        ppm_array_trainval[i,:] = ppm_tr3[idx_tr3[count_idx_tr3],:]
        t_array_trainval[i,:] = t_tr3[idx_tr3[count_idx_tr3],:]
        target_array_trainval[i,:] = target_tr3[idx_tr3[count_idx_tr3],:]
        fids_array_trainval[i,:,:,:] = fids_tr3[idx_tr3[count_idx_tr3],:,:,:]
        #print('tr3:',idx_tr3[count_idx_tr3])
        count_idx_tr3 = count_idx_tr3 + 1

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

#----test----
#definitions:
name_dataset_to_be_created_test = '../dataset_test_realdata_augmented_from_tracks2and3.h5'
qntty_to_repeat = 50
start=0
stop=5
#data:
ppm_array_test_aux = np.empty((5,2048))
t_array_test_aux = np.empty((5,2048))
target_array_test_aux = np.empty((5,2048))
fids_array_test_aux = np.empty((5,2048,2,160),dtype='complex128')
idx_tr2 = [9,11]
idx_tr3 = [7,9,11]
count_idx_tr2 = 0
count_idx_tr3 = 0
for i in range(ppm_array_test_aux.shape[0]):
    if i%2 == 0:
        ppm_array_test_aux[i,:] = ppm_tr3[idx_tr3[count_idx_tr3],:]
        t_array_test_aux[i,:] = t_tr3[idx_tr3[count_idx_tr3],:]
        target_array_test_aux[i,:] = target_tr3[idx_tr3[count_idx_tr3],:]
        fids_array_test_aux[i,:,:,:] = fids_tr3[idx_tr3[count_idx_tr3],:,:,:]
        #print('tr3:',idx_tr3[count_idx_tr3])
        count_idx_tr3 = count_idx_tr3 + 1
    else:
        ppm_array_test_aux[i,:] = ppm_tr2[idx_tr2[count_idx_tr2],:]
        t_array_test_aux[i,:] = t_tr2[idx_tr2[count_idx_tr2],:]
        target_array_test_aux[i,:] = target_tr2[idx_tr2[count_idx_tr2],:]
        fids_array_test_aux[i,:,:,:] = fids_tr2[idx_tr2[count_idx_tr2],:,:,:]
        #print('tr2:',idx_tr2[count_idx_tr2])
        count_idx_tr2 = count_idx_tr2 + 1


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