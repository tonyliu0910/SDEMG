import math
import os
import numpy as np
import random
import csv
import scipy.io
from scipy import signal
from tqdm import tqdm
import wfdb
from util import *


def get_filepaths_withSTI(directory,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename[-5] !='i' and filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
    return sorted(file_paths)

def add_noise(clean_path, noise_path, SNR, return_info=False, normalize=False):
    clean_rate = 1000
    y_clean = np.load(clean_path)
    noise_ori = np.load(noise_path)
    
    #if noise shorter than clean wav, extend
    if len(noise_ori) < len(y_clean):
        tmp = (len(y_clean) // len(noise_ori)) + 1
        y_noise = []
        for _ in range(tmp):
            y_noise.extend(noise_ori)
    else:
        y_noise = noise_ori

    # cut noise 
    start = random.randint(0,len(y_noise)-len(y_clean))
    end = start+len(y_clean)
    y_noise = y_noise[start:end]     
    y_noise = np.asarray(y_noise)

    y_clean_pw = np.dot(y_clean,y_clean) 
    y_noise_pw = np.dot(y_noise,y_noise) 

    scalar = np.sqrt(y_clean_pw/((10.0**(SNR/10.0))*y_noise_pw))
    noise = scalar * y_noise
    y_noisy = y_clean + noise
    if normalize: 
        norm_scalar = np.max(abs(y_noisy))
        y_noisy = y_noisy/norm_scalar

    if return_info is False:
        return y_noisy, clean_rate
    else:
        info = {}
        info['start'] = start
        info['end'] = end
        info['scalar'] = scalar
        return y_noisy, clean_rate, info




# Training and validation set parameters

# clean_paths,noise_paths = ["./data_E1_S40_Ch2_seg60s_ptb_ver2/train/clean","./data_E1_S40_Ch2_seg60s_ptb_ver2/val/clean"],["../ECG_Ch1_fs1000_bp_training","../ECG_Ch1_fs1000_bp_training"]
# exercise = 1
# channel = [2]
# # channel = [1,2]
# EMG_data_num = 40
# clean_paths,noise_paths = ["./data_E1_S40_Ch2_withSTI_seg60s_nsrd/train/clean","./data_E1_S40_Ch2_withSTI_seg60s_nsrd/val/clean"], ["../ECG_Ch1_fs1000_bp_training","../ECG_Ch1_fs1000_bp_training"]
# noise_paths =  ["../ECG_Ch1_fs1000_bp_training","../ECG_Ch1_fs1000_bp_training"]
# SNR_list = [-5,-7,-9,-11,-13,-15] # Training SNR
# num_of_copy = [10,5]
# cross_channel = False
# test = False
# normalize = False

# Testing set parameters
clean_paths,noise_paths = ["./data_E2_S40_Ch12_withSTI_seg60s_nsrd/test/clean"], ["../ECG_Ch1_fs1000_bp_testing"]
exercise = 2
# channel = [9,10,11,12]
channel = [12]
EMG_data_num = 40
SNR_list = [0,-2,-4,-6,-8,-10,-12,-14] #Testing SNR
num_of_copy = [4]
noise_paths = ["../ECG_Ch1_fs1000_bp_testing"]
test = True
cross_channel = False
###

normalize = False # Output noisy EMG without normalization. Set to False in building testing set.
sti = True
noisy_folder = 'noisy'
loop_break = False
for ch in channel:
    if loop_break == True:
        break
    if cross_channel == True:
        out_path = "./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(channel[0])+"_"+str(channel[1])+"_withSTI_seg60s_nsrd"
        loop_break = True
    else:
        out_path = "./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(ch)+"_withSTI_seg60s_nsrd"
    if test == True:
        clean_paths =[out_path+'/test/clean']
    else:
        clean_paths = [out_path+'/train/clean',out_path+'/val/clean']

    for i in range(len(clean_paths)):
        clean_path = clean_paths[i]
        noise_path = noise_paths[i]
        Noisy_path = clean_path.replace('clean',noisy_folder)
        root_path = clean_path.replace('clean',noisy_folder)
        
        check_path(Noisy_path)

        clean_list = get_filepaths(clean_path) if sti is False else get_filepaths_withSTI(clean_path)
        noise_list = get_filepaths(noise_path)
        
        sorted(clean_list)

        for snr in SNR_list:
                with open(root_path+str(snr)+'.csv', 'w', newline='') as csvFile:
                    fieldNames = ['EMG','ECG','start','end','scalar']
                    writer = csv.DictWriter(csvFile, fieldNames)
                    writer.writeheader()
                    for clean_emg_path in tqdm(clean_list):
                            noise_wav_path_list = random.sample(noise_list, num_of_copy[i])
                            for noise_ecg_path in noise_wav_path_list:
                                y_noisy, clean_rate, info = add_noise(clean_emg_path, noise_ecg_path, snr, True, normalize)
                                noise_name = noise_ecg_path.split(os.sep)[-1].split(".")[0]
                                output_dir = Noisy_path+os.sep+str(snr)+os.sep+noise_name
                                creat_dir(output_dir)
                                emg_name = clean_emg_path.split(os.sep)[-1].split(".")[0]
                                np.save(os.path.join(output_dir,emg_name),y_noisy)
                                writer.writerow({'EMG':emg_name,'ECG':noise_name, 'start':info['start'], 'end':info['end'],'scalar':info['scalar']})
      


