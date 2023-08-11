import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import CleanEMGDataset
from ddpm_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from torch.utils.data import DataLoader
from utils import *


root_dir = '/home/zhengxiawu/GitHub/Bio-ASP/DiffusionForEMGDenoising'
clean_file_path = "/home/tony/Bio-ASP/Data/data_E1_S40_withSTI_seg60s_nsrd/train/clean"

dataset = CleanEMGDataset(clean_file_path)

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 60000,
    timesteps = 1000,
    objective = 'pred_noise'
)

# Or using trainer

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 1,
    train_lr = 8e-5,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
# trainer.train()
score_path = f'./results/denoise_UNet_epochs_optims_loss_batch_lr.csv'
test_path = "/home/tony/Bio-ASP/EMG/ECG-removal-from-sEMG-by-FCN/data_E2_S40_Ch9_withSTI_seg60s_nsrd/test"
trainer.test(test_path, score_path, milestone=1)