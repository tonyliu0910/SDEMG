import numpy as np
import os
import scipy.io
from scipy import signal
from util import *
from tqdm import tqdm

def get_emg_filepaths(directory,number,exercise):
    # import n(umber) of EMG signals
    emg_paths =[]
    for i in range(1,number+1):
        filename = "DB2_s"+str(i)+"/S"+str(i)+"_E"+str(exercise)+"_A1.mat"
        emg_paths.append(os.path.join(directory, filename))
    return emg_paths

def read_emg(emg_path,channel,restimulus=False):
    # extract nth channel EMG, bandpass,  down-sampling, normalize
    b, a = signal.butter(4, [20,500], 'bp',fs=2000) #bandpass = signal.butter(4, [20,500], 'bandpass',output='sos',fs=2000)
    emg_data = scipy.io.loadmat(emg_path)
    y_clean = emg_data.get('emg')[:,channel-1] #channel 
    y_clean = signal.filtfilt(b,a,y_clean)[::2]
    y_clean = y_clean/np.max(abs(y_clean))
    y_clean = y_clean.astype('float64')
    if restimulus:
        y_restimulus = emg_data.get('restimulus')[::2]
    else:
        y_restimulus = 0
    return y_clean, y_restimulus

# Corpus_path = '../EMG_DB2/'
# exercise = 2
# channel = [9,10,11,12]
# EMG_data_num = 40
# cross_channel = False#True
# segment = 60 # unit: second
# points_per_seg = segment * 1000 # fs = 1000 Hz

Corpus_path = '../EMG_DB2/'
exercise = 1
channel = [2]
EMG_data_num = 40
cross_channel = False#True
segment = 60 # unit: second
points_per_seg = segment * 1000 # fs = 1000 Hz

for ch in channel:
    if cross_channel == True:
        out_path ="./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(channel[0])+"_"+str(channel[1])+"_withSTI_seg60s_nsrd"
    else:
        out_path ="./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(ch)+"_withSTI_seg60s_nsrd"
    check_path(out_path)
    check_path(out_path+'/train/clean')
    check_path(out_path+'/test/clean')
    check_path(out_path+'/val/clean')

    file_paths = get_emg_filepaths(Corpus_path,EMG_data_num,exercise)

    for i in tqdm(range(len(file_paths))):
        test = False    
        if i<24:
            save_path = out_path+'/train/clean'
        elif i<30:
            save_path = out_path+'/val/clean'
        else:    
            save_path = out_path+'/test/clean'
            test = True
        emg_file,restimulus = read_emg(file_paths[i],ch,test)
        
        for j in range(emg_file.shape[0]//points_per_seg):
            np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)),emg_file[j*points_per_seg:(j+1)*points_per_seg])
            if test:
                np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)+"_sti"),restimulus[j*points_per_seg:(j+1)*points_per_seg])