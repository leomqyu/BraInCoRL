import numpy as np
from pathlib import Path
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

from model import *
from torchmetrics.regression import ExplainedVariance, CosineSimilarity, MeanSquaredError
from utils import load_weights_and_predict
from utils import *

seed = random.randint(0, 7777)
# print(f'[INFO] seed set for random in pipeline.py is {seed}')
random.seed(seed)

class PLTrainVal(pl.LightningModule):
    def __init__(self, max_in_context, min_in_context=10, max_unknown=10,
            lr=1e-3, weight_decay=1e-4, lambda_l2=1, 
            sched_step_size=30, sched_factor=0.1,
            result_dir=None
        ):

        super().__init__()

        self.max_in_context = max_in_context
        self.min_in_context = min_in_context
        assert self.max_in_context >= self.min_in_context, f'self.max_in_context must >= self.min_in_context, but {self.max_in_context} and {self.min_in_context} actually'
        self.max_unknown = max_unknown

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2

        print(f'[INFO] wd={self.weight_decay:e}')

        self.sched_step_size = sched_step_size
        self.sched_factor = sched_factor

        self.result_dir = result_dir

        self.train_loss = []
        self.val_loss = []

        '''criterion and metrics'''
        self.train_gtw_mse = []

        self.val_gtw_mse = []

        self.criterion_mse = nn.MSELoss()
        self.criterion_cos = nn.CosineEmbeddingLoss()

        self.train_ele_expl_var_values = []
        self.val_ele_expl_var_values = []

        self.train_cos_sim = CosineSimilarity(reduction='mean')
        self.val_cos_sim = CosineSimilarity(reduction='mean')

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()

        print('='*50)
        print(f'[INFO] Initialized PLTrainVal with hyperparameters:')
        print(f'  max_in_context={self.max_in_context}')
        print(f'  min_in_context={self.min_in_context}')
        print(f'  max_unknown={self.max_unknown}')
        print(f'  lr={self.lr}')
        print(f'  weight_decay={self.weight_decay}')
        print(f'  lambda_l2={self.lambda_l2}')
        print(f'  sched_step_size={self.sched_step_size}')
        print(f'  sched_factor={self.sched_factor}')
        print(f'  result_dir={self.result_dir}')
        print('='*50)

    def training_step(self, batch, batch_idx):
        assert self.training

        '''get pred results'''
        max_ic_img_embs, max_ic_vxl_resp, uk_img_embs, uk_vxl_resp, gt_weights = batch

        actual_ic_indices = random.sample(
            range(self.max_in_context),
            random.randint(self.min_in_context, self.max_in_context)
        )
        ic_img_embs = max_ic_img_embs[:, actual_ic_indices, :]
        ic_vxl_resp = max_ic_vxl_resp[:, actual_ic_indices]

        output, weights = self(ic_img_embs, ic_vxl_resp, uk_img_embs)
        assert output.shape == uk_vxl_resp.shape, f'output.shape != uk_vxl_resp.shape: {output.shape} != {uk_vxl_resp.shape}'

        train_ele_expl_var = manual_expl_var(output, uk_vxl_resp)

        cosine_loss = self.criterion_cos(F.normalize(weights[:, :-1], p=2, dim=-1), F.normalize(gt_weights[:, :-1], p=2, dim=-1),
                                         torch.ones(weights[:, :-1].size(0), device=self.device))
        l2_loss = self.criterion_mse(weights, gt_weights)
        loss = cosine_loss + self.lambda_l2 * l2_loss

        self.train_gtw_mse.append(l2_loss.item())

        self.train_loss.append(loss.item())
        self.train_ele_expl_var_values.extend(train_ele_expl_var.detach().cpu().numpy())
        self.train_cos_sim.update(F.normalize(weights, p=2, dim=-1),
                                  F.normalize(gt_weights, p=2, dim=-1))  # (B,)
        self.train_mse.update(output, uk_vxl_resp)

        return loss

    def on_train_epoch_end(self):
        avg_train_loss = np.mean(self.train_loss)
        avg_train_expl_var_manual = np.mean(self.train_ele_expl_var_values)        
        train_cos_sim = self.train_cos_sim.compute()
        train_mse = self.train_mse.compute()

        train_gtw_mse = np.mean(self.train_gtw_mse)
        

        self.log('train_loss', avg_train_loss, sync_dist=True, prog_bar=True)
        self.log('train_expl_var', avg_train_expl_var_manual, sync_dist=True, prog_bar=True)
        self.log('train_cos_sim', train_cos_sim, sync_dist=True, prog_bar=True)
        self.log('train_mse', train_mse, sync_dist=True, prog_bar=False)
        self.log('train_gtw_mse', train_gtw_mse, sync_dist=True, prog_bar=False)

        self.train_loss = []
        self.train_ele_expl_var_values = []
        self.train_cos_sim.reset()
        self.train_mse.reset()
        self.train_gtw_mse = []

    def validation_step(self, batch, batch_idx):
        assert not self.training

        max_ic_img_embs, max_ic_vxl_resp, uk_img_embs, uk_vxl_resp, gt_weights = batch

        variable_ic_num = random.randint(self.min_in_context, self.max_in_context)
        actual_ic_indices = random.sample(range(self.max_in_context), variable_ic_num)
        # print(f'[DEBUG] variable_ic_num', variable_ic_num)

        ic_img_embs = max_ic_img_embs[:, actual_ic_indices, :]
        ic_vxl_resp = max_ic_vxl_resp[:, actual_ic_indices]

        output, weights = self(ic_img_embs, ic_vxl_resp, uk_img_embs)

        val_ele_expl_var = manual_expl_var(output, uk_vxl_resp)
        # loss = torch.mean(val_ele_expl_var)

        # loss = self.criterion_mse(output, uk_vxl_resp)
        cosine_loss = self.criterion_cos(F.normalize(weights[:, :-1], p=2, dim=-1), F.normalize(gt_weights[:, :-1], p=2, dim=-1),
                                         torch.ones(weights[:, :-1].size(0), device=self.device))
        l2_loss = self.criterion_mse(weights, gt_weights)
        loss = cosine_loss + self.lambda_l2 * l2_loss
        
        self.val_gtw_mse.append(l2_loss.item())

        self.val_loss.append(loss.item())
        self.val_ele_expl_var_values.extend(val_ele_expl_var.cpu().numpy())
        self.val_cos_sim.update(F.normalize(weights, p=2, dim=-1), F.normalize(gt_weights, p=2, dim=-1))
        self.val_mse.update(output, uk_vxl_resp)

        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = np.mean(self.val_loss)
        avg_val_expl_var_manual = np.mean(self.val_ele_expl_var_values)        
        val_cos_sim = self.val_cos_sim.compute()
        val_mse = self.val_mse.compute()

        val_gtw_mse = np.mean(self.val_gtw_mse)


        self.log('val_loss', avg_val_loss, sync_dist=True, prog_bar=True)
        self.log('val_expl_var', avg_val_expl_var_manual, sync_dist=True, prog_bar=True)
        self.log('val_cos_sim', val_cos_sim, sync_dist=True, prog_bar=True)
        self.log('val_mse', val_mse, sync_dist=True, prog_bar=False)
        self.log('val_gtw_mse', val_gtw_mse, sync_dist=True, prog_bar=False)

        self.val_loss = []
        self.val_ele_expl_var_values = []
        self.val_cos_sim.reset()
        self.val_mse.reset()
        self.val_gtw_mse = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=self.sched_factor, patience=5, threshold=0.0001, cooldown=2, min_lr=1e-5)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': { 
                'scheduler': scheduler, 'monitor': 'val_loss',  # Metric to monitor for ReduceLROnPlateau
                'interval': 'epoch', 'frequency': 1,
            },
        }

# class PLTester(pl.LightningModule):
#     """
#     args:
#         store_all: 
#             if it is false, just store some metrics, including:
#                 a. element-wise (voxel-wise) cos similarity (num_voxel, )
#                 b. element-wise (voxel-wise) explained variance (num_voxel, )
#             if it is true, will store all the predicted results, except for above, also including
#                 c. the neuro response predicted (num_test_img, num_voxel) 
#                 d. the projection weights predicted (num_voxel, emb_len+1)
#     """
#     def __init__(self, max_in_context, min_in_context=10, max_unknown=10, 
#                  lambda_l2=1, result_dir=None, store_all=False):

#         super().__init__()

#         """init"""
#         self.max_in_context = max_in_context
#         self.min_in_context = min_in_context
#         assert self.max_in_context >= self.min_in_context, f'self.max_in_context must >= self.min_in_context, but {self.max_in_context} and {self.min_in_context} actually'
#         self.max_unknown = max_unknown

#         self.lambda_l2 = lambda_l2

#         self.result_dir = result_dir

#         self.store_all = store_all
        
#         '''criterion and metrics'''
#         self.criterion_mse = nn.MSELoss()
#         self.criterion_cos = nn.CosineEmbeddingLoss()

#         self.test_loss = []
#         self.test_ele_expl_var_values = []
#         self.test_cos_sim = CosineSimilarity(reduction='none')
#         self.test_mse = MeanSquaredError()

#         if self.store_all:
#             self.pred_weights = []      # the predicted weights for each voxel
#             self.pred_nrn = []          # the predicted nrn response for each voxel

#         print('='*50) 
#         print(f'[INFO] Initialized PLTester with hyperparameters:')
#         print(f'  max_in_context={self.max_in_context}')
#         print(f'  min_in_context={self.min_in_context}')
#         print(f'  max_unknown={self.max_unknown}')
#         print(f'  lambda_l2={self.lambda_l2}')
#         print(f'  result_dir={self.result_dir}')
#         print(f'  store_all={self.store_all}')
#         print('='*50) 

#     def test_step(self, batch, batch_idx):
#         max_ic_img_embs, max_ic_vxl_resp, uk_img_embs, uk_vxl_resp, gt_weights = batch

#         variable_ic_num = random.randint(self.min_in_context, self.max_in_context)
#         actual_ic_indices = random.sample(range(self.max_in_context), variable_ic_num)

#         ic_img_embs = max_ic_img_embs[:, actual_ic_indices, :]
#         ic_vxl_resp = max_ic_vxl_resp[:, actual_ic_indices]

#         output, weights = self(ic_img_embs, ic_vxl_resp, uk_img_embs)

#         # loss = self.criterion_mse(output, uk_vxl_resp)
#         cosine_loss = self.criterion_cos(F.normalize(weights[:, :-1], p=2, dim=-1), F.normalize(gt_weights[:, :-1], p=2, dim=-1),
#                                          torch.ones(weights[:, :-1].size(0), device=self.device))
#         l2_loss = self.criterion_mse(weights, gt_weights)
#         loss = cosine_loss + self.lambda_l2 * l2_loss

#         '''store'''
#         self.test_loss.append(loss.item())

#         test_ele_expl_var = manual_expl_var(output, uk_vxl_resp)
#         self.test_ele_expl_var_values.extend(test_ele_expl_var.cpu().numpy())
        
#         self.test_cos_sim.update(F.normalize(weights, p=2, dim=-1), F.normalize(gt_weights, p=2, dim=-1))
#         self.test_mse.update(output, uk_vxl_resp)

#         if self.store_all:
#             self.pred_weights.append(weights)
#             self.pred_nrn.append(output)

#         return loss

#     def on_test_epoch_end(self):
#         print('\n')
#         avg_test_loss = np.mean(self.test_loss)

#         avg_test_expl_var_manual = np.mean(self.test_ele_expl_var_values)        
#         test_cos_sim = self.test_cos_sim.compute()
#         avg_test_cos_sim = torch.mean(test_cos_sim)
        
#         test_mse = self.test_mse.compute()

#         # print(f'[DEBUG] test_cos_sim: ', test_cos_sim)
#         # print(f'[DEBUG] test_cos_sim.shape: ', test_cos_sim.shape)      # [22961, ] 
#         # print(f'[DEBUG] len(self.test_ele_expl_var_values): ', len(self.test_ele_expl_var_values))      # [22961, ]
#         # print(f'[INFO] max(self.test_ele_expl_var_values): ', max(self.test_ele_expl_var_values))

#         self.log('test_loss', avg_test_loss, sync_dist=True, prog_bar=True)
#         self.log('test_expl_var', avg_test_expl_var_manual, sync_dist=True, prog_bar=True)
#         self.log('avg_test_cos_sim', avg_test_cos_sim, sync_dist=True, prog_bar=True)
#         self.log('test_mse', test_mse, sync_dist=True, prog_bar=True)

#         np.save(Path(self.result_dir, 'test_ele_expl_var_values.npy'), np.array(self.test_ele_expl_var_values))
#         np.save(Path(self.result_dir, 'test_ele_cos_sim_values.npy'), self.test_cos_sim.compute().cpu().numpy())

#         self.test_loss = []
#         self.test_cos_sim.reset()
#         self.test_mse.reset()

#         if self.store_all:
#             np.save(Path(self.result_dir, 'pred_voxel_weights.npy'), np.concatenate(self.pred_weights, axis=0))
#             np.save(Path(self.result_dir, 'pred_voxel_nrn.npy'), np.concatenate(self.pred_nrn, axis=0))

#             self.pred_weights = []
#             self.pred_nrn = []

            