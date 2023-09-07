from torch.utils.data import Dataset
from torch import Tensor
from glob import glob
import os
import numpy as np


class EMGDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.noisy_file_names = glob(os.path.join(self.file_path, 'noisy', '**', "*.npy"), recursive=True)
        self.clean_file_names = glob(os.path.join(self.file_path, 'clean', "*.npy"), recursive=True)
        self.snr_list = sorted(os.listdir(os.path.join(self.file_path, 'noisy')), key=int, reverse=True)

    def __len__(self):
        return len(self.noisy_file_names)
    
    def __getitem__(self, idx):
        noisy_file_name = self.noisy_file_names[idx]
        clean_file_name = os.path.join(self.file_path, 'clean', os.path.basename(noisy_file_name))

        noisy_data = np.load(noisy_file_name)
        clean_data = np.load(clean_file_name)
        # snr = noisy_file_name.split(os.sep)[-3]
        
        # data = np.vstack((clean_data, noisy_data))
        return Tensor(clean_data).unsqueeze(0), Tensor(noisy_data).unsqueeze(0)
        # return Tensor(noisy_data), Tensor(clean_data)

class EMGTestDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.noisy_file_names = glob(os.path.join(self.file_path, 'noisy', '**', "*.npy"), recursive=True)
        self.clean_file_names = glob(os.path.join(self.file_path, 'clean', "*.npy"), recursive=True)
        self.snr_list = sorted(os.listdir(os.path.join(self.file_path, 'noisy')), key=int, reverse=True)

    def __len__(self):
        return len(self.noisy_file_names)
    
    def __getitem__(self, idx):
        noisy_file_name = self.noisy_file_names[idx]
        clean_file_name = os.path.join(self.file_path, 'clean', os.path.basename(noisy_file_name))
        clean_sti_file_name = os.path.join(self.file_path, 'clean', os.path.basename(noisy_file_name).replace('.npy', '_sti.npy'))
        noisy_data = np.load(noisy_file_name)
        clean_data = np.load(clean_file_name)
        clean_sti = np.load(clean_sti_file_name)
        snr = noisy_file_name.split(os.sep)[-3]
        # data = np.vstack((clean_data, noisy_data))
        return Tensor(clean_data).unsqueeze(0), Tensor(noisy_data).unsqueeze(0), snr, Tensor(clean_sti).unsqueeze(0)
        
class CleanEMGDataset(Dataset):
    def __init__(self, clean_file_path):
        super().__init__()
        self.clean_file_path = clean_file_path
        self.file_names = glob(os.path.join(clean_file_path, "*.npy"), recursive=True)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        data = np.load(file_name)
        # print(f"file_name: {file_name}, data.shape: {data.shape}")
        return Tensor(data).unsqueeze(0)
    
# data

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()