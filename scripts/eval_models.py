import numpy as np
from pathlib import Path
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from typing import override
from sklearn.linear_model import RidgeCV

from model import HyperweightsPredictorModel
from torchmetrics.regression import ExplainedVariance, CosineSimilarity, MeanSquaredError
from utils import load_weights_and_predict

seed = random.randint(0, 7777)
print(f'[INFO] seed set for random in eval_models.py is {seed}')
random.seed(seed)

class RidgeCVBaseline:
    def __init__(self, alphas=None, cv=2):
        self.alphas = alphas if alphas is not None else [0.01, 1.0, 10.0, 100.0]
        self.cv = cv

    def __call__(self, ic_img, ic_nrn, unknown_img):
        return self.forward(ic_img, ic_nrn, unknown_img)

    def forward(self, ic_img, ic_nrn, unknown_img):
        '''
        ic_img: (B, num_ic, emb_dim)
        ic_nrn: (B, num_ic)
        unknown_img: (B, num_uk, emb_dim)
        '''
        B, num_ic, emb_dim = ic_img.shape
        num_uk = unknown_img.shape[1]

        ones = torch.ones((B, num_ic, 1), device=ic_img.device)
        X = torch.cat((ic_img, ones), dim=-1).cpu().numpy()  # Shape: (B, num_ic, emb_dim + 1)
        Y = ic_nrn.cpu().numpy()  # Shape: (B, num_ic)
        unknown_img = unknown_img.cpu().numpy()  # Shape: (B, num_uk, emb_dim)

        predictions = []
        weights = []

        for batch_idx in range(B):
            ridge = RidgeCV(alphas=self.alphas, fit_intercept=False, cv=self.cv).fit(X[batch_idx], Y[batch_idx])

            weight = ridge.coef_  # Shape: (emb_dim + 1,)
            pred = np.dot(
                np.concatenate([unknown_img[batch_idx], np.ones((num_uk, 1))], axis=-1), weight.T
            )  # Shape: (num_uk,)

            predictions.append(pred)
            weights.append(weight)

        predictions = torch.tensor(np.stack(predictions), device=ic_img.device)  # Shape: (B, num_uk)
        weights = torch.tensor(np.stack(weights), device=ic_img.device)  # Shape: (B, emb_dim + 1)

        return predictions, weights