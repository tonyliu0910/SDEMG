import os
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from ddpm_1d import GaussianDiffusion1D
from score import cal_snr, cal_rmse, cal_ARV, cal_CC, cal_KR, cal_MF, cal_prd, cal_R2
from utils import check_folder, get_filepaths, exists, has_int_squareroot, cycle, num_to_groups


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
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
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

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

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

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

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
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
    
    def write_score(self,test_file,test_path,output=False):   
        outname  = test_file.replace(f'{test_path}','').replace('/','_')
        c_file = os.path.join(self.test_clean,test_file.split('/')[-1])
        clean = np.load(c_file)
        stimulus = np.load(c_file.replace('.npy','_sti.npy'))
        noisy = np.load(test_file)
        # print(f"loading noisy file from {test_file}")
        # if self.args.STFT:
        #     n_emg,n_phase,n_len = make_spectrum(y = noisy)
        #     n_emg = torch.from_numpy(noisy).t()
        #     c_emg = make_spectrum(y = clean)[0]
        #     n_emg = torch.from_numpy(noisy).to(self.device).unsqueeze(0).type(torch.float32)
        #     c_emg = torch.from_numpy(c_emg).t().to(self.device).unsqueeze(0).type(torch.float32)
        # else: 
        #     n_emg = torch.from_numpy(noisy).to(self.device).unsqueeze(0).type(torch.float32)
        #     c_emg = torch.from_numpy(clean).to(self.device).unsqueeze(0).type(torch.float32)
        n_emg = torch.from_numpy(noisy).to(self.device).unsqueeze(0).unsqueeze(0).type(torch.float32)
        c_emg = torch.from_numpy(clean).to(self.device).unsqueeze(0).unsqueeze(0).type(torch.float32)
        # if self.inputdim_fix==True:
        #     n_emg = F.pad(n_emg,(0,2000-n_emg.shape[1]%2000)).view(-1,2000)
        #     c_emg = F.pad(c_emg,(0,2000-c_emg.shape[1]%2000))
        
        

        pred = self.model.denoise(n_emg)
        # pred = self.model.sample(batch_size=1)

        # if self.inputdim_fix==True:
        #     pred = pred.view(-1,1).squeeze()
        #     c_emg = c_emg.squeeze()

        criterion = nn.L1Loss()

        loss = criterion(pred, c_emg).item()
        pred = pred.cpu().detach().numpy()
        enhanced = pred.squeeze().squeeze()
            
        

        # Evaluation metrics
        SNR = cal_snr(clean,enhanced)
        RMSE = cal_rmse(clean,enhanced)
        PRD = cal_prd(clean,enhanced)
        RMSE_ARV = cal_rmse(cal_ARV(clean),cal_ARV(enhanced))
        KR = abs(cal_KR(clean)-cal_KR(enhanced))
        MF = cal_rmse(cal_MF(clean,stimulus),cal_MF(enhanced,stimulus))
        R2 = cal_R2(clean,enhanced)
        CC = cal_CC(clean,enhanced)
        
        with open(self.score_path, 'a') as f1:
            f1.write(f'{outname},{SNR},{loss},{RMSE},{PRD},{RMSE_ARV},{KR},{MF},{R2},{CC}\n')
        
        if output:
            emg_path = test_file.replace(f'{test_path}',f'./enhanced_data_E2_S40_Ch11_nsrd/{self.out_folder}') 
            check_folder(emg_path)
            np.save(emg_path,enhanced)

    def test(self, test_path, score_path, milestone):
        # load model
        self.load(milestone)
        self.test_clean = os.path.join(test_path,'clean')
        self.test_noisy = os.path.join(test_path,'noisy')
        self.score_path = score_path
        self.out_folder = os.path.join('test',f'milestone_{milestone}')
        test_folders = get_filepaths(self.test_noisy)

        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f1:
            f1.write('Filename,SNR,Loss,RMSE,PRD,RMSE_ARV,KR,MF,R2,CC\n')
        for test_file in tqdm(test_folders):
            self.write_score(test_file,test_path,output=True)
        
        data = pd.read_csv(self.score_path)
        snr_mean = data['SNR'].to_numpy().astype('float').mean()
        loss_mean = data['Loss'].to_numpy().astype('float').mean()
        rmse_mean = data['RMSE'].to_numpy().astype('float').mean()
        prd_mean = data['PRD'].to_numpy().astype('float').mean()
        arv_mean = data['RMSE_ARV'].to_numpy().astype('float').mean()
        kr_mean = data['KR'].to_numpy().astype('float').mean()
        mf_mean = data['MF'].to_numpy().astype('float').mean()
        r2_mean = data['R2'].to_numpy().astype('float').mean()
        cc_mean = data['CC'].to_numpy().astype('float').mean()
        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(snr_mean),str(loss_mean),str(rmse_mean),str(prd_mean),str(arv_mean),str(kr_mean),str(mf_mean),str(r2_mean),str(cc_mean)))+'\n')
    

        

    #     return 