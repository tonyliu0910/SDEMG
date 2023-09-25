import os
from argparse import ArgumentParser
import yaml

from dataset import EMGDataset, EMGTestDataset
from ddpm_1d import GaussianDiffusion1D
from trainer import Trainer1D
from model import Unet1D
from deep_filter_model import ConditionalModel
# from new_model import ConditionalModel
from utils import default


def main(args):
    with open(args.data_cfg, "r") as f:
        file_cfg = yaml.safe_load(f)
    with open(args.experiment_cfg, "r") as f:
        exp_cfg = yaml.safe_load(f)

    print(f"Running experiment {exp_cfg['project_name']}")

    train_path = file_cfg['train_dir']
    validation_path = file_cfg['valid_dir']
    result_path = os.path.join(file_cfg['result_dir'], exp_cfg['project_name'])
    score_path = os.path.join(result_path, f"{exp_cfg['project_name']}_ch12.csv")
    ptb_score_path = os.path.join(result_path, f"{exp_cfg['project_name']}_ptb.csv")
    test_path = file_cfg['test_dir']
    
    train_dataset = EMGDataset(train_path)
    validation_dataset = EMGDataset(validation_path)
    # print(f"dataset sample shape: {dataset[0].shape}")
    # model = Unet1D(
    #     dim = 64,
    #     dim_mults = (1, 2, 4, 8),
    #     channels = 1,
    #     self_condition = args.condition,
    # )

    model = ConditionalModel(feats=128)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = exp_cfg['seq_length'],
        timesteps = exp_cfg['sampling_steps'],
        objective = exp_cfg['objective'],
        loss_function = exp_cfg['loss_function'],
        beta_schedule = exp_cfg['beta_schedule'],
        condition = exp_cfg['condition'],
    )

    trainer = Trainer1D(
        diffusion,
        train_dataset = train_dataset,
        validation_dataset = validation_dataset,
        train_epochs = exp_cfg['train_epochs'],
        train_batch_size = exp_cfg['batch_size'],
        train_lr = exp_cfg['lr'],        
        gradient_accumulate_every = exp_cfg['gradient_accumulate_every'],    # gradient accumulation steps
        ema_decay = exp_cfg['ema_decay'],                # exponential moving average decay
        amp = exp_cfg['mix_precision'],
        results_folder = result_path,                      # turn on mixed precision
        num_workers = file_cfg['num_workers']
    )
    inference_milestone = default(exp_cfg['inference_milestone'], None)
    
    with open(os.path.join(result_path, 'experiment_cfg.yaml'), 'w') as yaml_file:
        yaml.dump(exp_cfg, yaml_file, default_flow_style=False)

    if args.train:
        print('start training')
        trainer.train()

    if args.test:
        print('testing normal condition')
        test_dataset = EMGTestDataset(test_path)
        trainer.test(test_dataset, score_path, milestone=inference_milestone, ddim=exp_cfg['ddim'], denoise_timesteps=exp_cfg['denoise_timesteps'])

    if args.sample:  
        print('sampling') 
        file_paths = ['demo file paths']
        trainer.denoise_sample(file_paths, milestone=inference_milestone, ddim=exp_cfg['ddim'], denoise_timesteps=exp_cfg['denoise_timesteps'], color='r')

    if args.test_mismatch:
        print('testing mismatch condition')
        mismatch_dataset = EMGTestDataset(file_cfg['mismatch_dir'])
        trainer.test(mismatch_dataset, ptb_score_path, milestone=inference_milestone, ddim=exp_cfg['ddim'], denoise_timesteps=exp_cfg['denoise_timesteps'])



if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a Diffusion model')
    parser.add_argument('--data_cfg', help='config for file paths on this machine')
    parser.add_argument('--experiment_cfg', default='cfg/default.yaml', help='experiment setting')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--test_mismatch', action='store_true')
    main(parser.parse_args())