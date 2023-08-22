import os
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import Tensor

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from ddpm_1d import GaussianDiffusion1D
from score import cal_snr, cal_rmse, cal_ARV, cal_CC, cal_KR, cal_MF, cal_prd, cal_R2
from utils import check_folder, get_filepaths, exists, has_int_squareroot, cycle, num_to_groups, check_path
import matplotlib.pyplot as plt


class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        num_workers = 4
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        # dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = self.num_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model).to(device)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        count = 0
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    # count = count + 1

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                        #     batches = num_to_groups(self.num_samples, self.batch_size)
                        #     all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # all_samples = torch.cat(all_samples_list, dim = 0)

                        # torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)
        # accelerator.print(f"total file count: { count}")
        accelerator.print('training complete')

    def test(self, test_dataset, score_path, milestone, ddim, denoise_timesteps=None):
        accelerator = self.accelerator
        device = accelerator.device
        # load model
        self.load(milestone)
        self.score_path = score_path
        
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        
        test_dl = DataLoader(test_dataset, batch_size = self.train_batch_size, shuffle = False, pin_memory = True, num_workers = self.num_workers)
        test_dl = self.accelerator.prepare(test_dl)

        snr_list = test_dataset.snr_list

        df = pd.DataFrame(index=test_dataset.snr_list, columns=['SNR','loss','rmse','prd','arv','kr', 'r2', 'cc', 'file_count'])

        for col in df.columns:
            df[col].values[:] = 0
        if self.model.loss_function == 'l1':
            criterion = nn.L1Loss()
        elif self.model.loss_function == 'l2':
            criterion = nn.MSELoss()
        count = 0


        with tqdm(test_dl) as it:
            for batch_idx, batch in enumerate(it):
                data_batch = batch[0]
                snr_batch = batch[1]
                clean_batch = data_batch[:,:1]
                noisy_batch = data_batch[:,1:].to(device)
                # print(f"clean shape {clean_batch.shape}, nosiy shape {noisy_batch.shape}")
                if ddim:
                    pred = self.model.ddim_denoise(noisy_batch)
                else:
                    pred = self.model.denoise(noisy_batch, denoise_timesteps=denoise_timesteps)
                # print(f"pred shape {pred.shape}")
                clean_batch = clean_batch.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                snr_batch = np.array(snr_batch)
                for i, (pred_i, clean, snr) in enumerate(zip(pred, clean_batch, snr_batch)):
                    clean = clean.squeeze().squeeze()
                    enhanced = pred_i.squeeze().squeeze()
                    # print(f"clean: {clean.shape}, enhanced: {enhanced.shape}")
                    loss = criterion(torch.from_numpy(enhanced), torch.from_numpy(clean)).item()
                    SNR = cal_snr(clean,enhanced)
                    RMSE = cal_rmse(clean,enhanced)
                    PRD = cal_prd(clean,enhanced)
                    RMSE_ARV = cal_rmse(cal_ARV(clean),cal_ARV(enhanced))
                    KR = abs(cal_KR(clean)-cal_KR(enhanced))
                    R2 = cal_R2(clean,enhanced)
                    CC = cal_CC(clean,enhanced)
                    df.at[snr, 'SNR'] = df.at[snr, 'SNR'] + SNR
                    df.at[snr, 'loss'] = df.at[snr, 'loss'] + loss
                    df.at[snr, 'rmse'] = df.at[snr, 'rmse'] + RMSE
                    df.at[snr, 'prd'] = df.at[snr, 'prd'] + PRD
                    df.at[snr, 'arv'] = df.at[snr, 'arv'] + RMSE_ARV
                    df.at[snr, 'kr'] = df.at[snr, 'kr'] + KR
                    df.at[snr, 'r2'] = df.at[snr, 'r2'] + R2
                    df.at[snr, 'cc'] = df.at[snr, 'cc'] + CC
                    df.at[snr, 'file_count'] = df.at[snr, 'file_count'] + 1

        # print(f"Testing done! Test file count: {count}")
        for col in df.columns[:8]:
            df[col].values[:] = df[col].values[:]/df['file_count'].values[:]
        df = df.round(5)
        df.to_csv(self.score_path)

    def denoise_sample(self, file_paths, milestone, ddim, denoise_timesteps=None):
        accelerator = self.accelerator
        device = accelerator.device
        self.out_folder = os.path.join(self.results_folder, f'milestone_{milestone}_enhanced')
        check_path(self.out_folder)
        
        # load model
        self.load(milestone)
        
        # load data

        for filepath in file_paths:
            filename = os.path.basename(filepath)
            clean_file_name = os.path.join(*filepath.split(os.sep)[:5])
            clean_file_name = os.path.join('/'+clean_file_name, 'clean', filename)
            snr = filepath.split(os.sep)[-3]
            clean_data = np.load(clean_file_name)
            noisy_data = np.load(filepath)
            noisy_tensor = Tensor(noisy_data).unsqueeze(0).unsqueeze(0).to(device)
            if ddim:
                pred = self.model.ddim_denoise(noisy_tensor)
            else:
                pred = self.model.denoise(noisy_tensor, denoise_timesteps=denoise_timesteps)
            pred = pred.cpu().detach().numpy().squeeze().squeeze()
            
            np.save(os.path.join(self.out_folder, f"snr_{snr}_{filename}"), pred)
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,3))
            ax[0].plot(clean_data)
            ax[0].set_title("clean")
            ax[1].plot(noisy_data)
            ax[1].set_title("noisy")
            ax[2].plot(pred)
            ax[2].set_title("denoised")
            fig.suptitle(f"SNR: {snr}, filename: {filename}")
            fig.savefig(os.path.join(self.out_folder, f"snr_{snr}_{filename}.png"))
            print(f"denoised file {filename} saved to {self.out_folder}")
        