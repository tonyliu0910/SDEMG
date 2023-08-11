from torch.utils.data import Dataset
from torch import Tensor
from glob import glob
import os
import numpy as np


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