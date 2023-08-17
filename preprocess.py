import numpy as np
import os
import wfdb
from tqdm import tqdm
from scipy import signal as sig
from scipy import signal
import scipy
import math
import random
import csv

from utils import check_path, resample, get_filepaths



class ECGdata:
    def __init__(self, corpus_path, train_path, test_path):
        self.corpus_path = corpus_path
        self.train_path = train_path
        self.test_path = test_path
        check_path(self.train_path)
        check_path(self.test_path)

        self.ecg_id =[]
        for f in os.listdir(self.corpus_path):
            f = os.path.splitext(os.path.basename(f))[0]
            if f.isdigit():
                self.ecg_id.append(f)

        self.ecg_id = sorted(list(set(self.ecg_id)))

        self.train_id = []
        self.test_id = []
        self.train_file = []
        self.test_file = []

        rand_list = np.random.randint(0, len(self.ecg_id)-1, int(len(self.ecg_id) / 5))

        for idx, id in enumerate(self.ecg_id):
            if idx in rand_list:
                self.test_id.append(id)
                self.test_file.append(os.path.join(self.corpus_path, str(id)))
            else:
                self.train_id.append(id)
                self.train_file.append(os.path.join(self.corpus_path, str(id)))

    def read_ecg(self, ecg_path):
        ecg = wfdb.rdrecord(ecg_path).__dict__.get('p_signal')[:,0] # channel 1 ECG
        ecg_rate = 1000
        ecg = resample(ecg,128,ecg_rate)
        ecg = ecg.astype('float64')
        return ecg

    def prepare(self):
        #Filter
        b_h, a_h = sig.butter(3, 10, 'hp',fs=1000)
        b_l, a_l = sig.butter(3, 200, 'lp',fs=1000)

        # Segment period
        start,end = 0, 70000

        print("training set ecg:")
        print(self.train_id)
        for i in tqdm(range(len(self.train_file))):
            file_path = self.train_file[i]
            ecg_file = self.read_ecg(file_path)
            if ecg_file.ndim>1:
                for j in range(ecg_file.shape[1]):
                    ecg_save = sig.filtfilt(b_l, a_l, sig.filtfilt(h_h, a_h, ecg_file[start:end, j]))
                    np.save(os.path.join(self.train_path, os.path.basename(file_path).split(".")[0] + str(j)), ecg_save)
            else:
                ecg_file = sig.filtfilt(b_l, a_l, sig.filtfilt(b_h, a_h, ecg_file[start:end]))
                np.save(os.path.join(self.train_path, os.path.basename(file_path).split(".")[0]), ecg_file)
            
        print("testing set ecg:")
        print(self.test_id)

        for i in tqdm(range(len(self.test_file))):
            file_path = self.test_file[i]
            ecg_file = self.read_ecg(file_path)
            if ecg_file.ndim>1:
                for j in range(ecg_file.shape[1]):
                    ecg_save = sig.filtfilt(b_l, a_l, sig.filtfilt(h_h, a_h, ecg_file[start:end, j]))
                    np.save(os.path.join(self.test_path, os.path.basename(file_path).split(".")[0] + str(j)), ecg_save)
            else:
                ecg_file = sig.filtfilt(b_l, a_l, sig.filtfilt(b_h, a_h, ecg_file[start:end]))
                np.save(os.path.join(self.test_path, os.path.basename(file_path).split(".")[0]), ecg_file)

class EMGdata:
    def __init__(self, corpus_path, train_path, test_path, train_noise_signal, test_noise_signal):
        self.corpus_path = corpus_path
        self.train_exercise = 1
        self.test_exercise = 2
        self.train_channel = [2]
        self.test_channel = [9]
        self.file_num = len(os.listdir(corpus_path))
        self.segment = 30 # (sec)
        self.points_per_seg = self.segment * 1000 # fs = 1000 Hz
        self.train_path = train_path
        self.test_path = test_path
        self.train_snr_list = [-5,-7,-9,-11,-13,-15]
        self.test_snr_list = [0,-2,-4,-6,-8,-10,-12,-14]
        self.train_num_of_copy = 10
        self.test_num_of_copy = 3
        self.train_noise_signal = train_noise_signal
        self.test_noise_signal = test_noise_signal

    def get_emg_filepaths(self, directory, number, exercise):
        # import n(umber) of EMG signals
        emg_paths =[]
        for i in range(1,number+1):
            filename = "DB2_s"+str(i)+"/S"+str(i)+"_E"+str(exercise)+"_A1.mat"
            emg_paths.append(os.path.join(directory, filename))
        return emg_paths

    def read_emg(self, emg_path, channel, restimulus=False):
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

    def prepare(self):
        for ch in self.train_channel:
            test=False
            self.train_path = f"{self.train_path}_E{self.train_exercise}_S{str(self.file_num)}_Ch{str(ch)}_withSTI_seg30s_nsrd"
            check_path(self.train_path)
            check_path(os.path.join(self.train_path,'clean'))
            file_paths = self.get_emg_filepaths(self.corpus_path, self.file_num, self.train_exercise)
            for i in tqdm(range(len(file_paths))):
                save_path = os.path.join(self.train_path,'clean')
                emg_file, restimulus = self.read_emg(file_paths[i], ch, test)     
                for j in range(emg_file.shape[0]//self.points_per_seg):
                    np.save(os.path.join(save_path, file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)), emg_file[j*self.points_per_seg:(j+1)*self.points_per_seg])
        
        for ch in self.test_channel:
            test=True
            self.test_path = f"{self.test_path}_E{self.test_exercise}_S{str(self.file_num)}_Ch{str(ch)}_withSTI_seg30s_nsrd"
            check_path(self.test_path)
            check_path(os.path.join(self.test_path,'clean'))
            file_paths = self.get_emg_filepaths(self.corpus_path, self.file_num, self.test_exercise)
            for i in tqdm(range(len(file_paths))):
                save_path = os.path.join(self.test_path,'clean')
                emg_file, restimulus = self.read_emg(file_paths[i], ch, test)     
                for j in range(emg_file.shape[0]//self.points_per_seg):
                    np.save(os.path.join(save_path, file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)), emg_file[j*self.points_per_seg:(j+1)*self.points_per_seg])
                    if test:
                        np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)+"_sti"), restimulus[j*self.points_per_seg:(j+1)*self.points_per_seg])

    def get_filepaths_withSTI(self, directory,ftype='.npy'):
        file_paths = []
        for root, directories, files in os.walk(directory):
            for filename in files:
                if filename[-5] !='i' and filename.endswith(ftype):
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)  # Add it to the list.
        return sorted(file_paths)

    def add_noise(self, clean_path, noise_path, SNR, return_info=False, normalize=False):
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
    
    def mixture(self):
        train_noisy_path = os.path.join(self.train_path, 'noisy')
        train_clean_path = os.path.join(self.train_path, 'clean')
        for ch in self.train_channel:
            clean_list = self.get_filepaths_withSTI(train_clean_path, '.npy')
            noise_list = get_filepaths(self.train_noise_signal)
            sorted(clean_list)
            for snr in self.train_snr_list:
                with open(train_noisy_path+str(snr)+'.csv', 'w', newline='') as csvFile:
                    fieldNames = ['EMG', 'ECG', 'start', 'end', 'scalar']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
                    writer.writeheader()
                    for clean_emg_path in tqdm(clean_list):
                        noise_wav_path_list = random.sample(noise_list, self.train_num_of_copy)
                        for noise_ecg_path in noise_wav_path_list:
                            y_noisy, clean_rate, info = self.add_noise(clean_emg_path, noise_ecg_path, snr, return_info=True, normalize=False)
                            noise_name = os.path.basename(noise_ecg_path).split(".")[0]
                            output_dir = os.path.join(train_noisy_path, str(snr), noise_name)
                            check_path(output_dir)
                            emg_name = os.path.basename(clean_emg_path).split(".")[0]
                            np.save(os.path.join(output_dir, emg_name), y_noisy)
                            writer.writerow({'EMG': emg_name, 'ECG': noise_name, 'start': info['start'], 'end': info['end'], 'scalar': info['scalar']})

        test_noisy_path = os.path.join(self.test_path, 'noisy')
        test_clean_path = os.path.join(self.test_path, 'clean')
        for ch in self.test_channel:
            clean_list = self.get_filepaths_withSTI(test_clean_path, '.npy')
            noise_list = get_filepaths(self.test_noise_signal)
            sorted(clean_list)
            for snr in self.test_snr_list:
                with open(test_noisy_path+str(snr)+'.csv', 'w', newline='') as csvFile:
                    fieldNames = ['EMG', 'ECG', 'start', 'end', 'scalar']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
                    writer.writeheader()
                    for clean_emg_path in tqdm(clean_list):
                        noise_wav_path_list = random.sample(noise_list, self.test_num_of_copy)
                        for noise_ecg_path in noise_wav_path_list:
                            y_noisy, clean_rate, info = self.add_noise(clean_emg_path, noise_ecg_path, snr, return_info=True, normalize=False)
                            noise_name = os.path.basename(noise_ecg_path).split(".")[0]
                            output_dir = os.path.join(test_noisy_path, str(snr), noise_name)
                            check_path(output_dir)
                            emg_name = os.path.basename(clean_emg_path).split(".")[0]
                            np.save(os.path.join(output_dir, emg_name), y_noisy)
                            writer.writerow({'EMG': emg_name, 'ECG': noise_name, 'start': info['start'], 'end': info['end'], 'scalar': info['scalar']})
            




if __name__ == '__main__':
    ecg_corpus_path = '/work/bigtony0910/mit-bih-normal-sinus-rhythm-database-1.0.0'
    ecg_train_path ='/work/bigtony0910/ECG_Ch1_fs1000_bp_training' # for training
    ecg_test_path ='/work/bigtony0910/ECG_Ch1_fs1000_bp_testing' # for testing

    ecg_data = ECGdata(ecg_corpus_path, ecg_train_path, ecg_test_path)
    ecg_data.prepare()
    
    emg_corpus_path = '/work/bigtony0910/EMG_DB2'
    emg_train_path = '/work/bigtony0910/dataset/train'
    emg_test_path = '/work/bigtony0910/dataset/test'
    emg_data = EMGdata(emg_corpus_path, emg_train_path, emg_test_path, ecg_train_path, ecg_test_path)
    emg_data.prepare()
    emg_data.mixture()
