import numpy as np
import os
import wfdb
from util import *
from tqdm import tqdm
from scipy import signal as sig

def get_ecg_filepaths(directory):
    ecg_id = [16265,16272,16273,16420,16483,16539,16773,16786,16795,17052,17453,18177,18184,19088,19090,19093,19140,19830]
    ecg_paths =[]
    for i in range(len(ecg_id)):
        ecg_paths.append(os.path.join(directory, str(ecg_id[i])))
    return ecg_paths

def read_ecg(ecg_path):
    ecg = wfdb.rdrecord(ecg_path).__dict__.get('p_signal')[:,0] # channel 1 ECG
    ecg_rate = 1000
    ecg = resample(ecg,128,ecg_rate)
    ecg = ecg.astype('float64')
    return ecg

Corpus_path = '../mit-bih-normal-sinus-rhythm-database-1.0.0'
# out_path ='../ECG_Ch1_fs1000_bp_training' # for training
out_path ='../ECG_Ch1_fs1000_bp_testing' # for testing

check_path(out_path)
file_paths = get_ecg_filepaths(Corpus_path)

#Filter
b_h, a_h = sig.butter(3, 10, 'hp',fs=1000)
b_l, a_l = sig.butter(3, 200, 'lp',fs=1000)

# Segment period
start,end = 0, 70000

for i in tqdm(range(len(file_paths))):
    ecg_file = read_ecg(file_paths[i])
    save_path = out_path
    if ecg_file.ndim>1:
        for j in range(ecg_file.shape[1]):
            ecg_save = sig.filtfilt(b_l,a_l,sig.filtfilt(b_h,a_h,ecg_file[start:end,j]))
            np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+str(j)),ecg_save)
    else:
        ecg_file = sig.filtfilt(b_l,a_l,sig.filtfilt(b_h,a_h,ecg_file[start:end]))
        np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]),ecg_file)
    