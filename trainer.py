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
from accelerate import DistributedDataParallelKwargs


class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        train_epochs = 50,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
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
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            kwargs_handlers=[ddp_kwargs]
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.train_epochs = train_epochs
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # dataset and dataloader

        # dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        # self.num_workers = cpu_count() // 2
        train_dl = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = self.num_workers)
        valid_dl = DataLoader(validation_dataset, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = self.num_workers)

        self.train_dl = self.accelerator.prepare(train_dl)
        self.valid_dl = self.accelerator.prepare(valid_dl)
        # self.dl = cycle(dl)

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

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.3, patience=3, verbose=True)

        self.model, self.opt= self.accelerator.prepare(self.model, self.opt)

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

        model = self.accelerator.unwrap_model(self.model)
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
        
        best_loss = 100000.

        for epoch in range(self.train_epochs):
            accelerator.print(f"Train epoch: {epoch}")
            # with tqdm(initial = 0, total = len(list(self.dl)), disable = not accelerator.is_main_process) as pbar:
            with tqdm(self.train_dl) as it:
                total_loss = 0.

                for batch_idx, (clean_batch, noisy_batch) in enumerate(it):
                    # print(clean_batch.shape, noisy_batch.shape)
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                    with self.accelerator.autocast():
                        loss = self.model(clean_batch, noisy_batch)
                        total_loss += loss.item()
                    
                    accelerator.backward(loss)
                
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    it.set_description(f'loss: {loss:.4f}')

                    accelerator.wait_for_everyone()
                    
                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()
            accelerator.print(f"Train loss: {total_loss/len(self.train_dl):.4f}")
            
            accelerator.print(f"Validation epoch: {epoch}")
            with tqdm(self.valid_dl) as it:
                total_loss = 0.
                for batch_idx, (clean_batch, noisy_batch) in enumerate(it):
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                    with self.accelerator.autocast():
                        loss = self.model(clean_batch, noisy_batch)
                        total_loss += loss.item()
                    it.set_description(f'loss: {loss:.4f}')
                    accelerator.wait_for_everyone()
            valid_loss = total_loss/len(self.valid_dl)
            self.lr_scheduler.step(valid_loss)
            accelerator.wait_for_everyone()
            accelerator.print(f"Validation loss: {valid_loss:.4f}")
            if valid_loss < best_loss and accelerator.is_main_process:
                self.ema.update()
                self.ema.ema_model.eval()
                self.save('best')
                best_loss = valid_loss
                accelerator.print(f"New best model saved at epoch {epoch} with loss {valid_loss:.4f}")
            
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

        # if self.model.loss_function == 'l1':
        #     criterion = nn.L1Loss()
        # elif self.model.loss_function == 'l2':
        #     criterion = nn.MSELoss()
        criterion = self.model.loss_function
        count = 0


        with tqdm(test_dl) as it:
            for batch_idx, (clean_batch, noisy_batch, snr_batch) in enumerate(it):
                if ddim:
                    pred = self.model.ddim_denoise(noisy_batch)
                else:
                    pred = self.model.denoise(noisy_batch, denoise_timesteps=denoise_timesteps)

                clean_batch = clean_batch.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                snr_batch = np.array(snr_batch)
                for i, (pred_i, clean, snr) in enumerate(zip(pred, clean_batch, snr_batch)):
                    clean = clean.squeeze().squeeze()
                    enhanced = pred_i.squeeze().squeeze()

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
                    count += 1

        print(f"Testing done! Test file count: {count}")
        for col in df.columns[:8]:
            df[col].values[:] = df[col].values[:]/df['file_count'].values[:]
        df = df.round(5)
        df.to_csv(self.score_path)

    def denoise_sample(self, file_paths, milestone, ddim, denoise_timesteps=None):
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:   
            self.out_folder = os.path.join(self.results_folder, f'milestone_{milestone}_enhanced')
            check_path(self.out_folder)
            # load model
            self.load(milestone)
            # load data
            ts = np.arange(0, self.model.sampling_timesteps+1, 1)
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize =(50, 30))
            fig.tight_layout()
            file_path = file_paths[0]
            filename = os.path.basename(file_path)
            clean_file_name = os.path.join(*file_path.split(os.sep)[:-4])
            clean_file_name = os.path.join('/'+clean_file_name, 'clean', filename)
            clean_data = np.load(clean_file_name)
            noisy_data = np.load(file_path)
            noisy_tensor = Tensor(noisy_data).unsqueeze(0).unsqueeze(0).to(device)
            snr = file_path.split(os.sep)[-3]
            for idx, denoise_ts in enumerate(ts):
                row = idx // 10
                col = idx % 10
                if ddim:
                    pred = self.model.ddim_denoise(noisy_tensor, denoise_timesteps=denoise_ts)
                else:
                    pred = self.model.denoise(noisy_tensor, denoise_timesteps=denoise_ts)
                pred = pred.cpu().detach().numpy().squeeze().squeeze()
                denoised_snr = cal_snr(clean_data, pred)
                denoised_rmse = cal_rmse(clean_data, pred)
                ax[row, col].plot(pred)
                ax[row, col].set_title(f"TS: {denoise_ts} SNRimp{denoised_snr - int(snr)} RMSE: {denoised_rmse}")
                ax[row, col].set_ylim(-1, 1)
                ax[row, col].set_xlim(0, 10000)
                print(f"step {denoise_ts} sample done!")
            fig.savefig(os.path.join(self.out_folder, f"denoise_ts_samples.png"))
            print(f"denoise_ts_samples saved to {self.out_folder}")

            
            ts = np.arange(0, self.model.num_timesteps+20, 10)
            fig, ax = plt.subplots(nrows=ts.shape[0], ncols=4, figsize=(20, 3*ts.shape[0]))
            fig.tight_layout()

            for idx, denoise_ts in enumerate(ts):
                for i, filepath in enumerate(file_paths):
                    filename = os.path.basename(filepath)
                    clean_file_name = os.path.join(*filepath.split(os.sep)[:-4])
                    clean_file_name = os.path.join('/'+clean_file_name, 'clean', filename)
                    snr = filepath.split(os.sep)[-3]
                    clean_data = np.load(clean_file_name)
                    noisy_data = np.load(filepath)
                    if idx == 0:
                        ax[idx, i].plot(clean_data)
                        ax[idx, i].set_title(f"clean")
                        ax[idx, i].set_ylim(-1, 1)
                        ax[idx, i].set_xlim(0, 10000)
                    elif idx == len(ts) - 1:
                        ax[idx, i].plot(noisy_data)
                        ax[idx, i].set_title(f"noisy {snr}")
                        ax[idx, i].set_ylim(-1, 1)
                        ax[idx, i].set_xlim(0, 10000)
                    else: 
                        noisy_tensor = Tensor(noisy_data).unsqueeze(0).unsqueeze(0).to(device)
                        if ddim:
                            pred = self.model.ddim_denoise(noisy_tensor, denoise_timesteps=denoise_ts)
                        else:
                            pred = self.model.denoise(noisy_tensor, denoise_timesteps=denoise_ts)
                        pred = pred.cpu().detach().numpy().squeeze().squeeze()
                        denoised_snr = cal_snr(clean_data, pred)
                        denoised_rmse = cal_rmse(clean_data, pred)
                        ax[idx, i].plot(pred)
                        if i == 0:
                            ax[idx, i].set_title(f"TS: {denoise_ts} SNR: {denoised_snr} RMSE: {denoised_rmse}")
                        else:
                            ax[idx, i].set_title(f"SNR: {denoised_snr} RMSE: {denoised_rmse}")
                        ax[idx, i].set_ylim(-1, 1)
                        ax[idx, i].set_xlim(0, 10000)
                        print(f"step {denoise_ts} snr {snr} sample done!")
            fig.savefig(os.path.join(self.out_folder, f"denoise_ts_snr_samples.png"))
            print(f"denoise_ts_samples saved to {self.out_folder}")

            file_num = len(file_paths)
            fig, ax = plt.subplots(nrows=file_num, ncols=3, figsize=(3*5, 3*file_num))
            fig.tight_layout()
            
            for idx, filepath in enumerate(file_paths):
                filename = os.path.basename(filepath)
                clean_file_name = os.path.join(*filepath.split(os.sep)[:-4])
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
                # np.save(os.path.join(self.out_folder, f"snr_{snr}_{filename}"), pred)
                denoised_snr = cal_snr(clean_data, pred)
                denoised_rmse = cal_rmse(clean_data, pred)

                ax[idx, 0].plot(clean_data)
                ax[idx, 0].set_title("Clean")
                ax[idx, 0].set_ylim(-1, 1)
                ax[idx, 0].set_xlim(0, 10000)
                ax[idx, 1].plot(noisy_data)
                ax[idx, 1].set_title(f"Noisy SNR: {snr}")
                ax[idx, 1].set_ylim(-1, 1)
                ax[idx, 1].set_xlim(0, 10000)
                ax[idx, 2].plot(pred)
                ax[idx, 2].set_title(f"Denoised SNRimp: {denoised_snr - int(snr)} RMSE: {denoised_rmse}")
                ax[idx, 2].set_ylim(-1, 1)
                ax[idx, 2].set_xlim(0, 10000)
                fig.suptitle(f"sEMG Denoising")
                fig.savefig(os.path.join(self.out_folder, f"signal_comparison.png"))
                print(f"denoised file: signal_comparison.png saved to {self.out_folder}")
        