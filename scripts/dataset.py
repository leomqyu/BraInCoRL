from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count

import lightning as pl
import numpy as np
import random
from pathlib import Path
import time 
import h5py
from utils import debug_info, is_unit_norm

seed = random.randint(0, 7777)
print(f'[INFO] seed set for random in dataset.py is {seed}')
random.seed(seed)
       
class IclDataMultiSubj(Dataset):
    """
    Data loader for training on multiple subject
    """
    def __init__(self, img_feats_path_l, nrn_feats_path_l, gt_arr_path_l,
                 max_in_context=100, max_unknown=10):

        self.img_feats_l = []
        self.nrn_feats_l = []
        self.gt_arr_l = []
        
        # Initialize timing variables
        self.load_times = []
        total_start = time.time()

        '''load nrn by mmap'''
        for subj_idx, (img_feats_path, nrn_feats_path, gt_arr_path) in enumerate(zip(img_feats_path_l, nrn_feats_path_l, gt_arr_path_l)):
            print(f'\nDataloader loading subject {subj_idx} data...')
            subj_start = time.time()
            
            # Load with timing for each file type
            start = time.time()
            img_feats = np.load(img_feats_path, mmap_mode='r').astype(np.float32)
            img_time = time.time() - start
            self.img_feats_l.append(img_feats)
            print(f'Loaded image feats in {img_time:.2f} seconds', flush=True)
            
            start = time.time()
            nrn_feats = np.load(nrn_feats_path, mmap_mode='r').astype(np.float32)
            nrn_time = time.time() - start
            self.nrn_feats_l.append(nrn_feats)
            print(f'Loaded nrn feats in {nrn_time:.2f} seconds', flush=True)
            
            start = time.time()
            gt_arr = np.load(gt_arr_path, mmap_mode='r').astype(np.float32)
            gt_time = time.time() - start
            self.gt_arr_l.append(gt_arr)
            print(f'Loaded gt feats in {gt_time:.2f} seconds')
            
            subj_time = time.time() - subj_start
            self.load_times.append(subj_time)
            print(f'Subject {subj_idx} loaded in {subj_time:.2f} seconds', flush=True)

        for img_feats in self.img_feats_l:
            assert img_feats.dtype == np.float32, "img_feats must be of type float32"
        for nrn_feats in self.nrn_feats_l:
            assert nrn_feats.dtype == np.float32, "nrn_feats must be of type float32"
        for gt_arr in self.gt_arr_l:
            assert gt_arr.dtype == np.float32, "gt_arr must be of type float32"
      
        print('len(self.img_feats_l), len(self.nrn_feats_l), len(self.gt_arr_l): ', len(self.img_feats_l), len(self.nrn_feats_l), len(self.gt_arr_l))
        
        '''check img normed'''
        for img in self.img_feats_l:
            assert is_unit_norm(img, dim=1)

        self.max_in_context = max_in_context
        self.max_unknown = max_unknown

        '''generate num voxel list from gt list'''
        self.num_voxel_l = []
        for gt in self.gt_arr_l:
            self.num_voxel_l.append(gt.shape[0])

    def map_glob_vxl_idx_to_subj_idx(self, glob_idx):
        '''
        give a global idx, map to the idx of the subj_idx (computational) and the vxl index in that subj
        eg glob_idx = 45, num_vxl_l = [10, 20, 30]; subj_idx = 2, s_vxl_idx = 45 - 10 -20 = 15
        '''
        num_vxl_cumsum = np.cumsum([0] + self.num_voxel_l)  # Prefix sum with zero for easier indexing
        assert 0 <= glob_idx < num_vxl_cumsum[-1], f"glob_idx = {glob_idx} is out of range"
        # print(num_vxl_cumsum)       # [ 0 10 30 60]
        subj_idx = np.searchsorted(num_vxl_cumsum, glob_idx, side='right') - 1
        s_vxl_idx = glob_idx - num_vxl_cumsum[subj_idx]
        return subj_idx, s_vxl_idx

    def __len__(self):
        return sum(self.num_voxel_l)

    def __getitem__(self, glob_idx):
        """
        return ic_img_embs, ic_vxl_resp, uk_ic_img_embs, uk_vxl_resp, voxel_gt_weigts
        """
        subj_idx, s_vxl_idx = self.map_glob_vxl_idx_to_subj_idx(glob_idx)

        img_feats = self.img_feats_l[subj_idx]
        nrn_feats = self.nrn_feats_l[subj_idx]
        gt_weights = self.gt_arr_l[subj_idx]

        '''random sample images'''
        num_img, emb_len = img_feats.shape
        rand_img_indices = random.sample(range(num_img), self.max_in_context + self.max_unknown)
        ic_img_idices = rand_img_indices[:self.max_in_context]
        uk_img_idices = rand_img_indices[self.max_in_context:]

        '''get img emb'''
        ic_img_embs = torch.tensor(img_feats[ic_img_idices, :])      # shape (self.max_in_context, E)
        uk_img_embs = torch.tensor(img_feats[uk_img_idices, :])      # shape (self.max_unknown, E)

        '''get voxel responce'''
        ic_vxl_resp = torch.tensor(nrn_feats[ic_img_idices, s_vxl_idx])         # shape (self.max_in_context)
        uk_vxl_resp = torch.tensor(nrn_feats[uk_img_idices, s_vxl_idx])          # shape (self.max_unknown)

        '''get weights'''
        gt_weights = torch.tensor(gt_weights[s_vxl_idx, :])        # shape (E+1)
        
        return ic_img_embs, ic_vxl_resp, uk_img_embs, uk_vxl_resp, gt_weights


class IclDataModuleMultiSubj(pl.LightningDataModule):
    def __init__(self, data_dir, train_sub_idx_l, val_sub_idx_l, batch_size=64, max_ic=100, max_uk=10, num_workers=8):
        """
        Only used for training and validation on multiple subjects 
        """
        super().__init__()

        self.num_workers = num_workers if num_workers else cpu_count()
        self.train_sub_idx_l = train_sub_idx_l
        self.val_sub_idx_l = val_sub_idx_l
        self.data_dir = Path(data_dir)

        print(f'[INFO] traning on subjects: {self.train_sub_idx_l}', flush=True)
        print(f'[INFO] val on subjects: {self.val_sub_idx_l}', flush=True)
        print(f'[INFO] getting data from {self.data_dir}', flush=True)

        self.batch_size = batch_size
        self.max_ic = max_ic
        self.max_uk = max_uk

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):

        '''train'''
        img_feats_path_l, nrn_feats_path_l, gt_path_l = [], [], []
        print('[DEBUG] train img')
        for subj_idx in self.train_sub_idx_l:
            img_feats_path_l.append(self.data_dir / f's{subj_idx}_unique_img_feats_normed.npy')
            nrn_feats_path_l.append(self.data_dir / f's{subj_idx}_unique_nrn_feats_msk.npy')
            gt_path_l.append(self.data_dir / f's{subj_idx}_gt.npy')
        self.train_dataset = IclDataMultiSubj(img_feats_path_l, nrn_feats_path_l, gt_path_l, self.max_ic, self.max_uk)


        '''val'''
        img_feats_path_l, nrn_feats_path_l, gt_path_l = [], [], []
        for subj_idx in self.val_sub_idx_l:
            img_feats_path_l.append(self.data_dir / f's{subj_idx}_unique_img_feats_normed_val.npy')
            nrn_feats_path_l.append(self.data_dir / f's{subj_idx}_unique_nrn_feats_msk_val.npy')
            gt_path_l.append(self.data_dir / f's{subj_idx}_gt_val.npy')
        self.val_dataset = IclDataMultiSubj(img_feats_path_l, nrn_feats_path_l, gt_path_l, self.max_ic, self.max_uk)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class IclDataMultiSubjH5py(Dataset):
    """
    Data loader for training on multiple subject
    """
    def __init__(self, hdf5_path, img_feats_keys, nrn_feats_keys, gt_keys,
                 max_in_context=100, max_unknown=10):


        self.hdf5_path = hdf5_path
        self.img_feats_keys = img_feats_keys
        self.nrn_feats_keys = nrn_feats_keys
        self.gt_keys = gt_keys
        self.max_in_context = max_in_context
        self.max_unknown = max_unknown
        self.hdf5_file = h5py.File(hdf5_path, 'r', libver='latest', swmr=True)

        self.num_voxel_l = []
        self.num_images_l = []  # List to store image counts per subject
        self.cumulative_voxels = [0]
        
        for img_key, gt_key in zip(img_feats_keys, gt_keys):
            self.num_images_l.append(self.hdf5_file[img_key].shape[0])
            self.num_voxel_l.append(self.hdf5_file[gt_key].shape[0])
        self.num_vxl_cumsum = np.cumsum([0] + self.num_voxel_l)

    def map_glob_vxl_idx_to_subj_idx(self, glob_idx):
        '''
        give a global idx, map to the idx of the subj_idx (computational) and the vxl index in that subj
        eg glob_idx = 45, num_vxl_l = [10, 20, 30]; subj_idx = 2, s_vxl_idx = 45 - 10 -20 = 15
        '''
        assert 0 <= glob_idx < self.num_vxl_cumsum[-1], f"glob_idx = {glob_idx} is out of range"
        subj_idx = np.searchsorted(self.num_vxl_cumsum, glob_idx, side='right') - 1
        s_vxl_idx = glob_idx - self.num_vxl_cumsum[subj_idx]
        return subj_idx, s_vxl_idx

    def __len__(self):
        return sum(self.num_voxel_l)

    def __getitem__(self, glob_idx):
        """
        return ic_img_embs, ic_vxl_resp, uk_ic_img_embs, uk_vxl_resp, voxel_gt_weigts
        """
        subj_idx, s_vxl_idx = self.map_glob_vxl_idx_to_subj_idx(glob_idx)

        img_feats = self.hdf5_file[self.img_feats_keys[subj_idx]]
        nrn_feats = self.hdf5_file[self.nrn_feats_keys[subj_idx]]
        gt_weights = self.hdf5_file[self.gt_keys[subj_idx]]

        '''random sample images'''
        num_img = self.num_images_l[subj_idx]

        rand_img_indices = random.sample(range(num_img), self.max_in_context + self.max_unknown)
        ic_img_idices = sorted(rand_img_indices[:self.max_in_context])
        uk_img_idices = sorted(rand_img_indices[self.max_in_context:])

        '''get img emb'''
        ic_img_embs = torch.from_numpy(img_feats[ic_img_idices, :])      # shape (self.max_in_context, E)
        uk_img_embs = torch.from_numpy(img_feats[uk_img_idices, :])      # shape (self.max_unknown, E)

        '''get voxel responce'''
        ic_vxl_resp = torch.from_numpy(nrn_feats[ic_img_idices, s_vxl_idx])         # shape (self.max_in_context)
        uk_vxl_resp = torch.from_numpy(nrn_feats[uk_img_idices, s_vxl_idx])          # shape (self.max_unknown)

        '''get weights'''
        gt_weights = torch.from_numpy(gt_weights[s_vxl_idx, :])        # shape (E+1)
        return ic_img_embs, ic_vxl_resp, uk_img_embs, uk_vxl_resp, gt_weights

    def __del__(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file:
            self.hdf5_file.close()


class IclDataModuleMultiSubjH5py(pl.LightningDataModule):
    def __init__(self, data_dir, train_sub_idx_l, val_sub_idx_l, batch_size=64, max_ic=100, max_uk=10, num_workers=8):
        """
        Only used for training and validation on multiple subjects 
        """
        super().__init__()

        self.num_workers = num_workers if num_workers else cpu_count()
        self.train_sub_idx_l = train_sub_idx_l
        self.val_sub_idx_l = val_sub_idx_l
        self.h5_path = Path(data_dir) / 'h5py_data.h5py'

        print(f'[INFO] traning on subjects: {self.train_sub_idx_l}', flush=True)
        print(f'[INFO] val on subjects: {self.val_sub_idx_l}', flush=True)
        print(f'[INFO] getting data from {self.h5_path}', flush=True)

        self.batch_size = batch_size
        self.max_ic = max_ic
        self.max_uk = max_uk

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):
        '''train'''
        img_feats_key_l, nrn_feats_key_l, gt_key_l = [], [], []
        for subj_idx in self.train_sub_idx_l:
            img_feats_key_l.append(f's{subj_idx}_unique_img_feats_normed')
            nrn_feats_key_l.append(f's{subj_idx}_unique_nrn_feats_msk')
            gt_key_l.append(f's{subj_idx}_gt')
        self.train_dataset = IclDataMultiSubjH5py(self.h5_path, img_feats_key_l, nrn_feats_key_l, gt_key_l, self.max_ic, self.max_uk)


        '''val'''
        img_feats_key_l, nrn_feats_key_l, gt_key_l = [], [], []
        for subj_idx in self.val_sub_idx_l:
            img_feats_key_l.append(f's{subj_idx}_unique_img_feats_normed_val')
            nrn_feats_key_l.append(f's{subj_idx}_unique_nrn_feats_msk_val')
            gt_key_l.append(f's{subj_idx}_gt_val')
        self.val_dataset = IclDataMultiSubjH5py(self.h5_path, img_feats_key_l, nrn_feats_key_l, gt_key_l, self.max_ic, self.max_uk)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.num_workers > 0, pin_memory=True)

        