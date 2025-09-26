from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import lightning as pl

from torchmetrics.regression import ExplainedVariance, CosineSimilarity, MeanSquaredError

from utils import load_weights_and_predict
from pipeline import PLTrainVal

import random 
import torch.distributed as dist

from utils import *



'''
modified form huggingface Dinov2SwiGLUFFN: 
'''
class SwiGLUFFN(nn.Module):
    '''no auto determined hidden size'''
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear1(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.linear2(hidden)

class SwigluAttentionBlock(nn.Module):
    def __init__(self, embed_dim, tsfm_hidden_dim, num_heads, dropout=0.0, need_weights=False):
        """Conventional attention + swiglu and attention residual
        """

        super().__init__()
        self.need_weights=need_weights

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, tsfm_hidden_dim, embed_dim)

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        log_scale = np.log(inp_x.shape[-2])

        '''store attn weights for external access'''
        attn_output, attn_w = self.attn(log_scale * inp_x, inp_x, inp_x, need_weights=self.need_weights)
        # print('[DEBUG] attn_w.shape',attn_w.shape)      # attn_w.shape torch.Size([64, 34, 34])

        self.last_attn = attn_w
        x = x + self.attn_dropout(attn_output)  # Apply dropout to the attention output
        x = x + self.ffn(self.layer_norm_2(x))
        return x

class ResidualBlock(nn.Module):
    # Follows "Identity Mappings in Deep Residual Networks", uses LayerNorm instead of BatchNorm, and LeakyReLU instead of ReLU
    def __init__(self, feat_in=128, feat_out=128, feat_hidden=256, drop_out=0.0, use_norm=True):
        super().__init__()
        # Define the residual block with or without normalization
        if use_norm:
            self.block = nn.Sequential(
                nn.LayerNorm(feat_in),  # Layer normalization on input features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_in, feat_hidden),  # Linear layer transforming input to hidden features
                nn.LayerNorm(feat_hidden),  # Layer normalization on hidden features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_hidden, feat_out)  # Linear layer transforming hidden to output features
            )
        else:
            self.block = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_in, feat_hidden),  # Linear layer transforming input to hidden features
                nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU activation
                nn.Dropout(p=drop_out),
                nn.Linear(feat_hidden, feat_out)  # Linear layer transforming hidden to output features
            )

        # Define the bypass connection
        if feat_in != feat_out:
            self.bypass = nn.Linear(feat_in, feat_out)  # Linear layer to match dimensions if they differ
        else:
            self.bypass = nn.Identity()  # Identity layer if input and output dimensions are the same

    def forward(self, input_data):
        # Forward pass: apply the block and add the bypass connection
        return self.block(input_data) + self.bypass(input_data)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ONLY MODIFY THE MODEL STRUCT AT HERE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class HyperweightsPredictorModel(nn.Module):
    def __init__(self, backbone_type):
        
        super().__init__()

        backbone_configs = {
            "DINO": {
                "embed_dim": 768, "internal_emb_dim": 800,
                "num_tsfm_layers": 20, "tsfm_hidden_dim": 2048, "num_reg_tok": 4, "num_heads": 10,
                "num_early_lyr": 1, "num_w_pred_layers": 1, "early_hidden_dim": 800*2, "w_pred_hidden_dim": 800*2,
                "dropout": 0
            },
            "SIGLIP": {
                "embed_dim": 1152, "internal_emb_dim": 1200,
                "num_tsfm_layers": 20, "tsfm_hidden_dim": 2048, "num_reg_tok": 4, "num_heads": 10,
                "num_early_lyr": 1, "num_w_pred_layers": 1, "early_hidden_dim": 1200*2, "w_pred_hidden_dim": 1200*2,
                "dropout": 0
            },
            "CLIP": {
                "embed_dim": 512, "internal_emb_dim": 560,
                "num_tsfm_layers": 20, "tsfm_hidden_dim": 2048,"num_reg_tok": 4, "num_heads": 10,
                "num_early_lyr": 1, "num_w_pred_layers": 1, "early_hidden_dim": 560*2, "w_pred_hidden_dim": 560*2,
                "dropout": 0
            },
            "vis": {
                "embed_dim": 512, "internal_emb_dim": 560,
                "num_tsfm_layers": 3, "tsfm_hidden_dim": 2048,"num_reg_tok": 4, "num_heads": 10,
                "num_early_lyr": 1, "num_w_pred_layers": 1, "early_hidden_dim": 560*2, "w_pred_hidden_dim": 560*2,
                "dropout": 0
            }
        }
        
        if backbone_type not in backbone_configs:
            raise ValueError(f"Invalid backbone_type: {backbone_type}. Must be one of {list(backbone_configs.keys())}")
        
        config = backbone_configs[backbone_type]
        
        self.embed_dim = config["embed_dim"]
        self.internal_emb_dim = config["internal_emb_dim"]
        self.num_tsfm_layers = config["num_tsfm_layers"]
        self.tsfm_hidden_dim = config["tsfm_hidden_dim"]
        self.num_reg_tok = config["num_reg_tok"]
        self.num_heads = config["num_heads"]
        self.num_early_lyr = config["num_early_lyr"]
        self.num_w_pred_layers = config["num_w_pred_layers"]
        self.early_hidden_dim = config["early_hidden_dim"]
        self.w_pred_hidden_dim = config["w_pred_hidden_dim"]
        self.dropout = config["dropout"]
        self.backbone_type = backbone_type

        # Print all hyperparameters during initialization
        print("\n" + "="*50)
        print("HyperweightsPredictorModel INITIALIZATION PARAMETERS")
        print("="*50)
        print(f"embed_dim: {self.embed_dim}")
        print(f"internal_emb_dim: {self.internal_emb_dim}")
        print(f"num_tsfm_layers: {self.num_tsfm_layers}")
        print(f"tsfm_hidden_dim: {self.tsfm_hidden_dim}")
        print(f"num_reg_tok: {self.num_reg_tok}")
        print(f"num_heads: {self.num_heads}")
        print(f"num_early_lyr: {self.num_early_lyr}")
        print(f"num_w_pred_layers: {self.num_w_pred_layers}")
        print(f"early_hidden_dim: {self.early_hidden_dim}")
        print(f"w_pred_hidden_dim: {self.w_pred_hidden_dim}")
        print(f"dropout: {self.dropout}")
        print("="*50 + "\n")

        '''model struct'''
        # the first layer also used as map hidden_dim to internal hidden dim
        self.early_layers = nn.Sequential(ResidualBlock(feat_in=self.embed_dim + 1, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True),
                                          *(ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True) for _ in range(self.num_early_lyr-1)))

        
        # class tokens
        cls_tensor = torch.randn(1, self.num_reg_tok, self.internal_emb_dim)
        cls_tensor = cls_tensor / (float(self.internal_emb_dim + 1)**0.5)
        self.cls_token = nn.Parameter(cls_tensor, requires_grad=True)

        # Transformer Layers
        self.input_dropout = nn.Dropout(self.dropout)
        self.transformer = nn.Sequential(*(SwigluAttentionBlock(self.internal_emb_dim, self.tsfm_hidden_dim, self.num_heads, dropout=self.dropout, need_weights=True)
                                           for _ in range(self.num_tsfm_layers)))

        # weight prediction
        # the last layer also used as map hidden_dim to internal_hidden dim
        self.weight_pred = nn.Sequential(*(ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True) for _ in range(self.num_early_lyr-1)),
                                         ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.embed_dim + 1, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True))

    def forward(self, ic_img, ic_nrn, unknown_img):
        """
        x is a batch of image embeddings, returned is the predicted activation for the batch of image in a single voxel
        x.shape is (B, S, E) where B is batch size, S is the num of images for in context learning, E is the length of image embeddings (512)
        
        ic_img is the image embeddings for incontext learning: (B, S_ic, E)
        ic_nrn is the neural activation for incontext learning: (B, S_ic)
        unknown_img is the img embedding to predict: (B, S_uk, E)
        """

        # print('[DEBUG] self.dropout', self.dropout)

        B, S_ic, E = ic_img.shape # batch, in context samples, 512
        S_uk = unknown_img.shape[1]

        # debug_info()
        # print(f'[DEBUG] B, S, E: {ic_img.shape}')   # B, S, E: torch.Size([1, 100, 512])

        x = self.early_layers(torch.cat([ic_img, ic_nrn.unsqueeze(-1)], dim=-1))  # [B, S, E+1]  batch, in context samples, 512

        # # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, N, E+1]
        x = torch.cat([cls_token, x], dim=1)  # [B, S+N, E+1]

        '''Apply Transformer'''
        # print('[DEBUG] type(x)', x.dtype)       # torch.float32
        x = self.input_dropout(x)
        x = self.transformer(x)

        # Perform hyperweights prediction
        pred_tok = x[:, 0, :]  # [B, E+1]

        weights = self.weight_pred(pred_tok)  # [B, E+1]

        pred = load_weights_and_predict(weights, unknown_img)

        return pred, weights

class HyperweightsPredictorTrainerComb(pl.LightningModule):
    def __init__(self,    
                default_struct=None,      
                embed_dim=None, internal_emb_dim=None,   
                num_tsfm_layers=None, tsfm_hidden_dim=None, num_reg_tok=None, num_heads=None, 
                num_early_lyr=None, num_w_pred_layers=None, early_hidden_dim=None, w_pred_hidden_dim=None,
                dropout=None,
                ############## above is the model struct, below is pipeline related ##############
                max_in_context=500, min_in_context=30, max_unknown=10,
                lr=1e-3, weight_decay=1e-4, loss_type='wsim', lambda_l2=1, # loss type is wsim or pmse
                sched_step_size=30, sched_factor=0.1,
                result_dir=None):
        
        super().__init__()
        self.save_hyperparameters()  

        ARCHITECTURE_PRESETS = {
            'CLIP': {
                'embed_dim': 512, 'internal_emb_dim': 560,
                'num_tsfm_layers': 20, 'tsfm_hidden_dim': 2048, 'num_reg_tok': 4, 'num_heads': 10,
                'num_early_lyr': 1, 'early_hidden_dim': 560*2, 'num_w_pred_layers': 1, 'w_pred_hidden_dim': 560*2
            },
            'DINO': {
                'embed_dim': 768, 'internal_emb_dim': 800,
                'num_tsfm_layers': 20, 'tsfm_hidden_dim': 2048, 'num_reg_tok': 4, 'num_heads': 10,
                'num_early_lyr': 1,
                'early_hidden_dim': 800*2, 'num_w_pred_layers': 1, 'w_pred_hidden_dim': 800*2
            },
            'SIGLIP': {
                'embed_dim': 1152, 'internal_emb_dim': 1200,
                'num_tsfm_layers': 20, 'tsfm_hidden_dim': 2048, 'num_reg_tok': 4, 'num_heads': 10,
                'num_early_lyr': 1, 'early_hidden_dim': 1200*2, 'num_w_pred_layers': 1, 'w_pred_hidden_dim': 1200*2
            }
        }

        # Set architecture parameters based on default_struct or individual arguments
        self.default_struct = default_struct
        if self.default_struct in ARCHITECTURE_PRESETS:
            print(f'[INFO] Using {default_struct} architecture preset. Will OVERWRITE!!!')
            preset = ARCHITECTURE_PRESETS[default_struct]
            self.embed_dim = preset['embed_dim']
            self.internal_emb_dim = preset['internal_emb_dim']
            self.num_tsfm_layers = preset['num_tsfm_layers']
            self.tsfm_hidden_dim = preset['tsfm_hidden_dim']
            self.num_reg_tok = preset['num_reg_tok']
            self.num_heads = preset['num_heads']
            self.num_early_lyr = preset['num_early_lyr']
            self.early_hidden_dim = preset['early_hidden_dim']
            self.num_w_pred_layers = preset['num_w_pred_layers']
            self.w_pred_hidden_dim = preset['w_pred_hidden_dim']
        else:
            # Use individual arguments with some basic validation
            self.embed_dim = embed_dim if embed_dim is not None else 768
            self.internal_emb_dim = internal_emb_dim if internal_emb_dim is not None else 800
            self.num_tsfm_layers = num_tsfm_layers if num_tsfm_layers is not None else 20
            self.tsfm_hidden_dim = tsfm_hidden_dim if tsfm_hidden_dim is not None else 2048
            self.num_reg_tok = num_reg_tok if num_reg_tok is not None else 4
            self.num_heads = num_heads if num_heads is not None else 10
            self.num_early_lyr = num_early_lyr if num_early_lyr is not None else 1
            self.early_hidden_dim = early_hidden_dim if early_hidden_dim is not None else self.internal_emb_dim*2
            self.num_w_pred_layers = num_w_pred_layers if num_w_pred_layers is not None else 1
            self.w_pred_hidden_dim = w_pred_hidden_dim if w_pred_hidden_dim is not None else self.internal_emb_dim*2

        assert self.num_early_lyr > 0, "Number of early layers must be > 0"
        assert self.num_w_pred_layers > 0, "Number of weight prediction layers must be > 0"
        assert self.internal_emb_dim % self.num_heads == 0, "internal_emb_dim must be divisible by num_heads"

        self.dropout = dropout
        print(f'[DEBUG] dropout={self.dropout}')

        '''model struct'''
        # the first layer also used as map hidden_dim to internal hidden dim
        self.early_layers = nn.Sequential(ResidualBlock(feat_in=self.embed_dim + 1, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True),
                                          *(ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True) for _ in range(self.num_early_lyr-1)))

        
        # class tokens
        cls_tensor = torch.randn(1, self.num_reg_tok, self.internal_emb_dim)
        cls_tensor = cls_tensor / (float(self.internal_emb_dim + 1)**0.5)
        self.cls_token = nn.Parameter(cls_tensor, requires_grad=True)

        # Transformer Layers
        self.input_dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(*(SwigluAttentionBlock(self.internal_emb_dim, self.tsfm_hidden_dim, self.num_heads, dropout=self.dropout)
                                           for _ in range(self.num_tsfm_layers)))

        # weight prediction
        # the last layer also used as map hidden_dim to internal_hidden dim
        self.weight_pred = nn.Sequential(*(ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True) for _ in range(self.num_early_lyr-1)),
                                         ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.embed_dim + 1, feat_hidden=self.early_hidden_dim, 
                                                         drop_out=self.dropout, use_norm=True))

        ############## init pipeline ##############
        
        self.max_in_context = max_in_context
        self.min_in_context = min_in_context
        assert self.max_in_context >= self.min_in_context, f'self.max_in_context must >= self.min_in_context, but {self.max_in_context} and {self.min_in_context} actually'
        self.max_unknown = max_unknown

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        assert self.loss_type in ['wsim', 'pmse'], f'{self.loss_type} unknown'
        self.lambda_l2 = lambda_l2

        print(f'[DEBUG] wd={self.weight_decay:e}')

        self.sched_step_size = sched_step_size
        self.sched_factor = sched_factor

        self.result_dir = result_dir

        self.train_loss = []
        self.val_loss = []
        # self.test_loss = []

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

        ################### print all hyperparameters during initialization ###################
        params_to_print = [
            'default_struct',
            'embed_dim', 'internal_emb_dim', 
            'num_tsfm_layers', 'tsfm_hidden_dim', 'num_reg_tok', 'num_heads',
            'num_early_lyr', 'num_w_pred_layers', 'early_hidden_dim', 'w_pred_hidden_dim', 
            'dropout', 
            'max_in_context', 'min_in_context', 'max_unknown', 
            'lr', 'weight_decay', 'loss_type', 'lambda_l2', 
            'sched_step_size', 'sched_factor', 
            'result_dir'
        ]
        print("\n" + "="*50)
        print("HyperweightsPredictorModel INITIALIZATION PARAMETERS")
        print("="*50)
        for param in params_to_print:
            # if hasattr(self, param):
            print(f"{param}: {getattr(self, param)}")
        print('='*50)

    def forward(self, ic_img, ic_nrn, unknown_img):
        """
        x is a batch of image embeddings, returned is the predicted activation for the batch of image in a single voxel
        x.shape is (B, S, E) where B is batch size, S is the num of images for in context learning, E is the length of image embeddings (512)
        
        ic_img is the image embeddings for incontext learning: (B, S_ic, E)
        ic_nrn is the neural activation for incontext learning: (B, S_ic)
        unknown_img is the img embedding to predict: (B, S_uk, E)
        """

        # print('[DEBUG] self.dropout', self.dropout)

        B, S_ic, E = ic_img.shape # batch, in context samples, 512
        S_uk = unknown_img.shape[1]

        # debug_info()
        # print(f'[DEBUG] B, S, E: {ic_img.shape}')   # B, S, E: torch.Size([1, 100, 512])

        x = self.early_layers(torch.cat([ic_img, ic_nrn.unsqueeze(-1)], dim=-1))  # [B, S, E+1]  batch, in context samples, 512

        # # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, N, E+1]
        x = torch.cat([cls_token, x], dim=1)  # [B, S+N, E+1]

        '''Apply Transformer'''
        # print('[DEBUG] type(x)', x.dtype)       # torch.float32
        x = self.input_dropout(x)
        x = self.transformer(x)

        # Perform hyperweights prediction
        pred_tok = x[:, 0, :]  # [B, E+1]

        weights = self.weight_pred(pred_tok)  # [B, E+1]

        pred = load_weights_and_predict(weights, unknown_img)

        return pred, weights

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
        # loss = torch.mean(train_ele_expl_var)

        
        l2_loss = self.criterion_mse(weights, gt_weights)
        if self.loss_type == 'wsim':
            cosine_loss = self.criterion_cos(F.normalize(weights[:, :-1], p=2, dim=-1), F.normalize(gt_weights[:, :-1], p=2, dim=-1),
                                         torch.ones(weights[:, :-1].size(0), device=self.device))
            loss = cosine_loss + self.lambda_l2 * l2_loss
        elif self.loss_type == 'pmse':
            loss = self.criterion_mse(output, uk_vxl_resp)
        else:
            raise ValueError()

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

        cosine_loss = self.criterion_cos(F.normalize(weights[:, :-1], p=2, dim=-1), F.normalize(gt_weights[:, :-1], p=2, dim=-1),
                                         torch.ones(weights[:, :-1].size(0), device=self.device))
        l2_loss = self.criterion_mse(weights, gt_weights)
        loss = cosine_loss + self.lambda_l2 * l2_loss
        # loss = self.criterion_mse(output, uk_vxl_resp)
        
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

