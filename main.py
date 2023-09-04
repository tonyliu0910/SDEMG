import os
from argparse import ArgumentParser

from dataset import EMGDataset, EMGTestDataset
from ddpm_1d import GaussianDiffusion1D
from trainer import Trainer1D
from model import Unet1D
from deep_filter_model import ConditionalModel
from utils import default

def main(args):
    print(f"Running experiment {args.project_name}")
    train_path = args.train_dir
    validation_path = args.valid_dir
    result_path = os.path.join(args.result_dir, args.project_name)
    score_path = os.path.join(result_path, f'{args.project_name}.csv')
    test_path = args.test_dir
    
    train_dataset = EMGDataset(train_path)
    validation_dataset = EMGDataset(validation_path)
    # print(f"dataset sample shape: {dataset[0].shape}")
    # model = Unet1D(
    #     dim = 64,
    #     dim_mults = (1, 2, 4, 8),
    #     channels = 1,
    #     self_condition = args.condition,
    # )

    model = ConditionalModel(feats=64)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.seq_length,
        timesteps = args.sampling_steps,
        objective = args.objective,
        loss_function = args.loss_function,
        beta_schedule = args.beta_schedule,
        condition = args.condition,
    )

    trainer = Trainer1D(
        diffusion,
        train_dataset = train_dataset,
        validation_dataset = validation_dataset,
        train_epochs = args.train_epochs,
        train_batch_size = args.batch_size,
        train_lr = args.lr,        
        gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = args.ema_decay,                # exponential moving average decay
        amp = args.mix_precision,
        results_folder = result_path,                      # turn on mixed precision
        num_workers = args.num_workers
    )

    trainer.train()
    
    inference_milestone = default(args.inference_milestone, args.train_epochs-1)

    test_dataset = EMGTestDataset(test_path)
    # trainer.test(test_dataset, score_path, milestone=inference_milestone, ddim=args.ddim, denoise_timesteps=args.denoise_timesteps)

    #currently 10s
    file_paths = ['/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg10s_nsrd/noisy/0/16420/S1_E2_A1_ch9_3.npy',
                   '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg10s_nsrd/noisy/-4/16420/S1_E2_A1_ch9_3.npy',
                   '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg10s_nsrd/noisy/-8/16420/S1_E2_A1_ch9_3.npy',
                   '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg10s_nsrd/noisy/-12/16420/S1_E2_A1_ch9_3.npy']
    # file_paths = ['/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg10s_nsrd/noisy/-10/16420/S1_E2_A1_ch9_3.npy']

    # file_paths = ['/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/0/16272/S1_E2_A1_ch9_1.npy',
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-2/16272/S1_E2_A1_ch9_1.npy', 
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-4/16272/S1_E2_A1_ch9_1.npy',
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-6/16272/S1_E2_A1_ch9_1.npy',
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-8/16272/S1_E2_A1_ch9_1.npy',
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-10/16272/S1_E2_A1_ch9_1.npy',
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-12/16272/S1_E2_A1_ch9_1.npy',
    #             '/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg5s_nsrd/noisy/-14/16272/S1_E2_A1_ch9_1.npy']

    # trainer.denoise_sample(file_paths, milestone=inference_milestone, ddim=args.ddim, denoise_timesteps=args.denoise_timesteps)

if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a Diffusion model')
    parser.add_argument('--project_name', default='Sample_DF_10sec_EP30_SS50_quad', help='project name')
    parser.add_argument('--train_epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--root_dir', default='.', help='root directory for data and model storage')
    parser.add_argument('--train_dir', default='/work/t22302856/Tony_data/sEMG_Dataset/train_E1_S40_Ch2_withSTI_seg10s_nsrd', help='directory containing training EMG waveforms')
    parser.add_argument('--valid_dir', default='/work/t22302856/Tony_data/sEMG_Dataset/valid_E3_S10_Ch2_withSTI_seg10s_nsrd', help='directory containing validation EMG waveforms')
    parser.add_argument('--test_dir', default='/work/t22302856/Tony_data/sEMG_Dataset/test_E2_S10_Ch9_withSTI_seg10s_nsrd', help='directory containing testing EMG waveforms') 
    parser.add_argument('--result_dir', default='/work/t22302856/Tony_data/EMG_denoise', help='directory to store scores')
    parser.add_argument('--condition', default=True, type=bool, help='condition on noise')
    parser.add_argument('--sampling_steps', default=50, type=int, help='number of sampling steps')
    parser.add_argument('--beta_schedule', default='quad', help='diffusion process beta scheduler')
    parser.add_argument('--ddim', default=False, type=bool, help='use ddim sampling')
    parser.add_argument('--seq_length', default=10000, type=int, help='length of sequence')
    parser.add_argument('--objective', default='pred_noise', help='diffusion objective')
    parser.add_argument('--loss_function', default='l2', help='loss function')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--mix_precision', default=True, type=bool, help='turn on mixed precision')
    parser.add_argument('--gradient_accumulate_every', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--ema_decay', default=0.995, type=float, help='exponential moving average decay')
    parser.add_argument('--inference_milestone', default='best', help='select milestone model for inference')
    parser.add_argument('--denoise_timesteps', default=None, type=int, help='denoise step')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    main(parser.parse_args())