import os
from argparse import ArgumentParser

from dataset import CleanEMGDataset
from ddpm_1d import GaussianDiffusion1D
from trainer import Trainer1D
from model import Unet1D
from utils import default

def main(args):
    clean_file_path = args.clean_dir
    result_path = os.path.join(args.result_dir, args.project_name)
    score_path = os.path.join(result_path, f'{args.project_name}.csv')
    test_path = args.test_dir

    dataset = CleanEMGDataset(clean_file_path)

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.seq_length,
        timesteps = args.sampling_steps,
        objective = args.objective
    )

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = args.batch_size,
        train_lr = args.lr,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = args.ema_decay,                # exponential moving average decay
        amp = args.mix_precision,                    # turn on mixed precision
        results_folder = result_path                      
    )    
    inference_milestone = default(args.inference_milestone, int(args.train_steps / 1000))
    trainer.test(test_path, score_path, milestone=inference_milestone, denoise_timesteps=args.denoise_timesteps)


if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a DiffWave model')
    parser.add_argument('--project_name', default='DiffuEMG_UNet_epochs_optims_loss_batch_lr', help='project name')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--root_dir', default='.', help='root directory for data and model storage')
    parser.add_argument('--clean_dir', default='/home/tony/Bio-ASP/Data/data_E1_S40_withSTI_seg60s_nsrd/train/clean', help='directory containing clean EMG waveforms')
    parser.add_argument('--test_dir', default='/home/tony/Bio-ASP/Data/data_E2_S40_Ch9_withSTI_seg60s_nsrd/test', help='directory containing noisy EMG waveforms') 
    parser.add_argument('--result_dir', default='/work/t22302856/Tony_data/EMG_denoise', help='directory to store scores')
    parser.add_argument('--train_steps', default=700000, type=int, help='number of training steps')
    parser.add_argument('--sampling_steps', default=1000, type=int, help='number of sampling steps')
    parser.add_argument('--seq_length', default=60000, type=int, help='length of sequence')
    parser.add_argument('--objective', default='pred_noise', help='diffusion objective')
    parser.add_argument('--lr', default=8e-5, type=float, help='learning rate')
    parser.add_argument('--mix_precision', default=True, type=bool, help='turn on mixed precision')
    parser.add_argument('--gradient_accumulate_every', default=2, type=int, help='gradient accumulation steps')
    parser.add_argument('--ema_decay', default=0.995, type=float, help='exponential moving average decay')
    parser.add_argument('--inference_milestone', default=None, help='select milestone model for inference')
    parser.add_argument('--denoise_timesteps', default=100, type=int, help='denoise step')
    main(parser.parse_args())