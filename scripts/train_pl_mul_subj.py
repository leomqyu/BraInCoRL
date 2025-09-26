"""
The traning code for multiple object, withOUT validation
"""

from pathlib import Path
import torch
import numpy as np
import os
from datetime import datetime
import random

from dataset import IclDataModuleMultiSubj, IclDataModuleMultiSubjH5py
from model import HyperweightsPredictorTrainerComb

import argparse
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger

assert torch.cuda.is_available() 

seed = random.randint(0, 7777)
print(f'[INFO] seed set for random in train_pl.py is {seed}')
seed_everything(seed, workers=True)

def main(args):
    print('Initialization ...')
    ''''dir related'''
    job_name = args.job_name  # eg: 0205_subj1_unmasked
    print('[INFO] Working on job ', job_name)

    work_base = args.work_base
    print(f'[INFO] work_base: {work_base}')

    data_dir = args.data_dir
    print(f'[INFO] Getting data from: {data_dir}')

    results_dir = Path(work_base, 'results', 'train', job_name)
    model_dir = Path(results_dir, 'model')  # store model param checkpoints
    os.makedirs(model_dir, exist_ok=True)
    print(f'[INFO] Storing model checkpoint to: {model_dir}')

    ''''cuda'''
    cuda_visible_devices = ','.join(map(str, args.gpu))
    print(f'[INFO] cuda_visible_devices: ', cuda_visible_devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    ''''training hyper param'''
    default_struct = args.default_struct
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    epochs = args.epochs
    loss_type = args.loss_type

    ''''others'''
    train_subj_l = args.train_subj
    val_subj_l = args.val_subj
    max_in_context = args.max_in_context
    min_in_context = args.min_in_context
    max_unknown = args.max_unknown

    """init"""

    """train"""
    wandb_logger = WandbLogger(save_dir=str(results_dir), project=job_name)

    print('Loading data ...')
    data_module = IclDataModuleMultiSubj(data_dir=data_dir, train_sub_idx_l=train_subj_l, val_sub_idx_l=val_subj_l, 
                                         batch_size=batch_size, max_ic=max_in_context, max_uk=max_unknown)
    # data_module = IclDataModuleMultiSubjH5py(data_dir=data_dir, train_sub_idx_l=train_subj_l, val_sub_idx_l=val_subj_l, 
    #                                      batch_size=batch_size, max_ic=max_in_context, max_uk=max_unknown)

    print('Loading model ...')
    # model = HyperweightsPredictorTrainer(max_in_context=max_in_context, min_in_context=min_in_context, max_unknown=max_unknown,
    #                                      lr=lr, weight_decay=weight_decay, dropout=dropout, result_dir=results_dir)
    model = HyperweightsPredictorTrainerComb(default_struct=default_struct, max_in_context=max_in_context, min_in_context=min_in_context, max_unknown=max_unknown,
                                         lr=lr, weight_decay=weight_decay, loss_type=loss_type, dropout=dropout, result_dir=results_dir)

    if args.pretrained_model_path:
        print(f'[INFO] Loading pretrained model from {args.pretrained_model_path}')
        checkpoint = torch.load(args.pretrained_model_path, map_location=None, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print('[INFO] Pretrained model loaded successfully.')

    '''checkpoint save config'''
    print('Init checkpoint ...')
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch:03d}-{val_expl_var:.4f}',
        monitor='val_expl_var', mode='max', save_top_k=500, save_last=True)

    print('Start training ...')
    trainer = pl.Trainer(deterministic=False,
        logger=wandb_logger,
        max_epochs=epochs,
        default_root_dir=results_dir,  # save the lightning_log
        callbacks=[checkpoint_callback],  # save the checkpoint
        accelerator='cuda', devices='auto', strategy='fsdp',
        precision='16-mixed')  # Enable mixed precision (AMP) in Lightning)

    print('[DEBUG] model structure:\n')
    print(model)
    all_names = []
    for name, param in model.named_parameters():
        all_names.append(name)
    print('named parameters: ', all_names)

    if args.resume_ckpt_path:
        print(f'[INFO] Resuming from checkpoint {args.resume_ckpt_path}')
        trainer.fit(model=model, train_dataloaders=data_module, ckpt_path=args.resume_ckpt_path)
    else:
        trainer.fit(model=model, train_dataloaders=data_module)

if __name__ == '__main__':
    """read argument"""
    parser = argparse.ArgumentParser(description='simple distributed training job')

    parser.add_argument('--work_base', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--job_name', type=str, help='CUDA visible devices (default: 0)')

    parser.add_argument('--resume_ckpt_path', default=None, type=str, help='Path to the pretrained model checkpoint')
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Path to the pretrained model checkpoint')
    
    parser.add_argument('--default_struct', type=str, help='Input batch size on each device (default: 32)')

    parser.add_argument('--data_dir', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--img_feat_file', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--nrn_feat_file', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--train_subj', nargs='+', type=int, help='CUDA visible devices (default: 0)')
    parser.add_argument('--val_subj', nargs='+', type=int, help='CUDA visible devices (default: 0)')
    parser.add_argument('--max_in_context', default=100, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--min_in_context', default=100, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--max_unknown', default=10, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--loss_type', type=str, help='Input batch size on each device (default: 32)')

    parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='CUDA visible devices (default: 0)')

    parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Input batch size on each device (default: 32)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='Input batch size on each device (default: 32)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Input batch size on each device (default: 32)')

    parser.add_argument('--epochs', default=1000, type=int, help='Total epochs to train the model')

    args = parser.parse_args()

    main(args)
