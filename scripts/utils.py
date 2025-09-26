import inspect
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import explained_variance_score 


##################### metrics ########################
def manual_point_mse(pred, tar):
    """
    in shape of (batch_size, num_smpl) or (num_voxel, num_testing_img)
    calculate along dim=1, which is the average mse loss of a voxel

    return shape: (batch_size,)
    """
    if isinstance(pred, np.ndarray) and isinstance(tar, np.ndarray):
        return np.mean((tar - pred) ** 2, axis=1)
    elif isinstance(pred, torch.Tensor) and isinstance(tar, torch.Tensor):
        return torch.mean((tar - pred) ** 2, dim=1)
    else:
        raise TypeError("Inputs must be either both NumPy arrays or both PyTorch tensors.")

def manual_normed_cos_similarity(pred, tar):
    """
    in shape of (batch_size, emb_len+1) or (num_voxel, emb_len+1)
    calculate along dim=1, which is the cosine similarity weights of a voxel

    return shape: (batch_size,)
    """
    if isinstance(pred, np.ndarray) and isinstance(tar, np.ndarray):
        pred_norm = pred / np.linalg.norm(pred, axis=1, keepdims=True)
        tar_norm = tar / np.linalg.norm(tar, axis=1, keepdims=True)
        return np.sum(pred_norm * tar_norm, axis=1)
    elif isinstance(pred, torch.Tensor) and isinstance(tar, torch.Tensor):
        pred_norm = pred / torch.norm(pred, dim=1, keepdim=True)
        tar_norm = tar / torch.norm(tar, dim=1, keepdim=True)
        return torch.sum(pred_norm * tar_norm, dim=1)
    else:
        raise TypeError("Inputs must be either both NumPy arrays or both PyTorch tensors.")

def skl_expl_var(pred, tar):
    return explained_variance_score(tar.T, pred.T, multioutput='raw_values')

def manual_expl_var(pred, tar, has_nan=False):
    """
    in shape of (batch_size, num_smpl) or (num_voxel, num_testing_img)
    calculate along dim=1, which is the ev of a voxel

    return shape: (batch_size,)
    """
    if isinstance(pred, np.ndarray) and isinstance(tar, np.ndarray):
        return 1 - np.var(tar - pred, axis=1) / np.var(tar, axis=1)
        if has_nan:
            return 1 - np.nanvar(tar - pred, axis=1) / np.nanvar(tar, axis=1)
    elif isinstance(pred, torch.Tensor) and isinstance(tar, torch.Tensor):
        return 1 - torch.var(tar - pred, dim=1) / torch.var(tar, dim=1)
    else:
        raise TypeError("Inputs must be either both NumPy arrays or both PyTorch tensors.")

def manual_r2(pred, tar):
    """in shape of (batch_size, num_smpl)"""
    if isinstance(pred, np.ndarray) and isinstance(tar, np.ndarray):
        ss_res = np.sum((tar - pred) ** 2, axis=1)
        tar_mean = np.mean(tar, axis=1, keepdims=True)
        ss_tot = np.sum((tar - tar_mean) ** 2, axis=1)
        return 1 - (ss_res / ss_tot)
    elif isinstance(pred, torch.Tensor) and isinstance(tar, torch.Tensor):
        ss_res = torch.sum((tar - pred) ** 2, dim=1)
        tar_mean = torch.mean(tar, dim=1, keepdim=True)
        ss_tot = torch.sum((tar - tar_mean) ** 2, dim=1)
        return 1 - (ss_res / ss_tot)
    else:
        raise TypeError("Inputs must be either both NumPy arrays or both PyTorch tensors.")

#################### eval ############################
def from_img_idx_to_img_key(img_idx, img_type, json_path):
    """
    input: 
        image index in the img_feats.npy (eg: 10)
        img type (eg: 's1_unique': just any key in the json file)
        json path: path to 'data/results/8_subj_split/img_idx.json
    
    output: 
        the padded img key string

    idea:
        The ith image in img_feats is the ith key in img_idx.json
    """
    with open(json_path, 'r') as f:
        img_keys_l = json_path[img_type]
    return img_keys_l[img_idx]

def from_img_key_to_img_idx(img_key, img_type, json_path):
    """
    input: 
        padded img key string (eg: '000000000370')
        img type (eg: 's1_unique': just any key in the json file)
        json path: path to 'data/results/8_subj_split/img_idx.json
    
    output: 
        the img index in the img_feats.npy

    idea:
        The ith image in img_feats is the ith key in img_idx.json
    """
    with open(json_path, 'r') as f:
        img_keys_l = json_path[img_type]
    return img_keys_l.index(target_string)

#################### prediction ######################
def load_weights_and_predict(weights, x):
    '''
    weights: [B, 513]
    x: [B, num_uk, 512]

    output: [B, num_uk]
    '''

    weight = weights[:, :-1]
    bias = weights[:, -1]

    # print(weight.shape, bias.shape, x.shape)        # torch.Size([16, 512]) torch.Size([16]) torch.Size([16, 10, 512])
    weight = weight.unsqueeze(1)  # Shape: [16, 1, 512]
    bias = bias.unsqueeze(-1)  # Shape: [16, 1]

    pred = torch.sum(weight * x, dim=-1) + bias  # torch.Size([16, 10])

    return pred

def convert_dict_w_file_to_numpy(gt_path):
    """
    input is a torch gt as a dict
    output is the numpy weights used for training
    """
    # gt['network']['linear.weight'].shape
    # gt['network']['linear.bias'].shape
    dict_gt = torch.load(gt_path, map_location='cpu', weights_only=True)
    return torch.cat([dict_gt['network']['linear.weight'], dict_gt['network']['linear.bias'].unsqueeze(-1)], dim=1).cpu().numpy()

#################### debug ###########################
def debug_info():
    # Get the current frame
    current_frame = inspect.currentframe()
    # Get the caller's frame (one level up in the stack)
    caller_frame = current_frame.f_back
    # Get the file name and line number from the caller's frame
    file_name = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    # Print the debug information
    print(f"[DEBUG] File '{file_name}', Line {line_number}")
    # Clean up the frame to avoid reference cycles
    del current_frame
    del caller_frame

def unit_norm(a):
    """
    input is of shape (num_img, img_emb), normalize along axis=-1
    """
    if isinstance(a, np.ndarray):
        return a / np.linalg.norm(a, axis=-1, keepdims=True)
    elif isinstance(a, torch.Tensor):
        return a / torch.norm(a, p=2, dim=-1, keepdim=True)
    else:
        raise TypeError("Not numpy array or tensor")

def is_unit_norm(a, dim=-1, tolerance = 1e-5):
    if isinstance(a, np.ndarray):
        norms = np.linalg.norm(a, axis=dim)
        return np.allclose(norms, np.ones_like(norms), atol=tolerance)
    elif isinstance(a, torch.Tensor):
        norms = torch.norm(a, p=2, dim=dim)
        return torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
    else:
        raise TypeError("Not numpy array or tensor")

def cal_gt_avg_norm(gt):
    """
    input is of shape (num_vxl, emb_len+1), cal norm along axis=1, then avg
    """
    if isinstance(gt, np.ndarray):
        return np.mean(np.linalg.norm(gt, axis=1))
    elif isinstance(gt, torch.Tensor):
        return torch.mean(torch.norm(gt, p=2, dim=1))
    else:
        raise TypeError("Not numpy array or tensor")
    
def inspect_gt_performance(gt, img, nrn):
    # print('gt.shape:', gt.shape)       # (num_vxl, 513)
    # print('img.shape:', img.shape)     # (num_img, 512)
    # print('nrn.shape:', nrn.shape)     # (num_img, num_vxl)

    if is_unit_norm(img):
        print('Image is unit norm')
    else:
        raise ValueError("Image is not unit normed")
    
    if isinstance(gt, np.ndarray):
        pred_nrn = np.concatenate((img, np.ones((img.shape[0], 1))), axis=1) @ gt.T         # (num_img, num_vxl)
    elif isinstance(gt, torch.Tensor):
        pred_nrn = torch.cat((img, torch.ones((img.shape[0], 1))), dim=1) @ gt.T
    
    pmse = manual_point_mse(pred_nrn.T, nrn.T)
    ev = manual_expl_var(pred_nrn.T, nrn.T)
    cos_sim = manual_normed_cos_similarity(pred_nrn.T, nrn.T)

    return pmse, ev, cos_sim

def inspect_gt_performance_batch(gt, img, nrn, batch_size=10000):
    """
    Computes performance metrics between predicted and actual neural responses in batches.
    
    Args:
        gt: Ground truth weights (num_voxels × 513)
        img: Image features (num_images × 512)
        nrn: Actual neural responses (num_images × num_voxels)
        batch_size: Number of voxels to process at once
        
    Returns:
        Tuple of (pmse, ev, cos_sim) averaged across voxels
    """
    # Validation checks
    assert gt.shape[1] == img.shape[1] + 1, "GT dim should be img_dim + 1 (for bias)"
    assert nrn.shape == (img.shape[0], gt.shape[0]), f"NRN shape mismatch: {nrn.shape} vs expected {(img.shape[0], gt.shape[0])}"
    
    if not is_unit_norm(img):
        raise ValueError("Image features must be unit normed")

    # Convert to torch if needed (for GPU acceleration)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
        img = torch.from_numpy(img)
        nrn = torch.from_numpy(nrn)
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    gt = gt.to(device)
    img = img.to(device)
    nrn = nrn.to(device)

    # Pre-allocate results
    pmse_total = 0
    ev_total = 0
    cos_total = 0
    cnt = 0


    # Process in batches
    for i in tqdm(range(0, gt.shape[0], batch_size)):
        batch_end = min(i + batch_size, gt.shape[0])
        
        # Get current batch
        gt_batch = gt[i:batch_end]
        nrn_batch = nrn[:, i:batch_end]  # Transpose to (batch_voxels × num_images)
        
        # Compute predictions
        ones = torch.ones((img.shape[0], 1), device=device)
        img_aug = torch.cat((img, ones), dim=1)
        pred_batch = img_aug @ gt_batch.T
        
        # Compute metrics
        # print(pred_batch.shape, nrn_batch.shape)
        mpse = manual_point_mse(pred_batch, nrn_batch)
        mev = manual_expl_var(pred_batch, nrn_batch)
        mcossim = manual_normed_cos_similarity(pred_batch, nrn_batch)

        pmse_total += mpse.sum()
        ev_total += mev.sum()
        cos_total += mcossim.sum()
        cnt += mpse.shape[0]
        print('current average pmse:', pmse_total / cnt)
        print('current average ev:', ev_total / cnt)
        print('current average cos:', cos_total / cnt)

    # Convert back to numpy if input was numpy
    if isinstance(gt, np.ndarray):
        return pmse_results.cpu().numpy(), ev_results.cpu().numpy(), cos_results.cpu().numpy()
    return pmse_results, ev_results, cos_results

#################### plot ############################
def plot_from_csv(result_dir, first_n = None, marker = None, store_pdf=False,
                  types = ['train', 'val'],
                  metrics = ['r2', 'expl_var', 'cos_sim', 'mse']):
    """init"""
    csv_path = os.path.join(result_dir, 'csv_logs/version_0/metrics.csv')
    data = pd.read_csv(csv_path)

    """loss"""
    data_train_loss = data[['epoch', 'train_loss']].dropna()
    data_val_loss = data[['epoch', 'val_loss']].dropna()

    plt.figure(figsize=(10, 6))
    if first_n is None:
        plt.plot(data_train_loss["epoch"], data_train_loss["train_loss"], label="Training Loss", marker=marker)
        plt.plot(data_val_loss["epoch"], data_val_loss["val_loss"], label="Validation Loss", marker=marker)
    else:
        plt.plot(data_train_loss["epoch"][:first_n], data_train_loss["train_loss"][:first_n], label="Training Loss", marker=marker)
        plt.plot(data_val_loss["epoch"][:first_n], data_val_loss["val_loss"][:first_n], label="Validation Loss", marker=marker)

    '''annotate'''
    train_loss_max = data_train_loss['train_loss'].max()
    train_loss_min = data_train_loss['train_loss'].min()
    train_loss_max_epoch = data_train_loss.loc[data_train_loss['train_loss'].idxmax(), 'epoch']
    train_loss_min_epoch = data_train_loss.loc[data_train_loss['train_loss'].idxmin(), 'epoch']
    plt.annotate(f'Max: {train_loss_max:.2f}', xy=(train_loss_max_epoch, train_loss_max), xytext=(train_loss_max_epoch, train_loss_max))
                #  arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Min: {train_loss_min:.2f}', xy=(train_loss_min_epoch, train_loss_min), xytext=(train_loss_min_epoch, train_loss_min))
                #  arrowprops=dict(facecolor='black', shrink=0.05))
    val_loss_max = data_val_loss['val_loss'].max()
    val_loss_min = data_val_loss['val_loss'].min()
    val_loss_max_epoch = data_val_loss.loc[data_val_loss['val_loss'].idxmax(), 'epoch']
    val_loss_min_epoch = data_val_loss.loc[data_val_loss['val_loss'].idxmin(), 'epoch']
    plt.annotate(f'Max: {val_loss_max:.2f}', xy=(val_loss_max_epoch, val_loss_max), xytext=(val_loss_max_epoch, val_loss_max))
                #  arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Min: {val_loss_min:.2f}', xy=(val_loss_min_epoch, val_loss_min), xytext=(val_loss_min_epoch, val_loss_min))
                #  arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    if store_pdf:
        output_pdf_path = os.path.join(result_dir, "loss_plot.pdf")
        plt.show()
        plt.savefig(output_pdf_path, format="pdf")
    output_pdf_path = os.path.join(result_dir, "loss_plot.png")
    plt.show()
    plt.savefig(output_pdf_path, format="png")
    plt.close()


    """metrics"""
    plt.figure(figsize=(10, 6))

    for metric in metrics:
        for tp in types:
            metric_col_name = f'{tp}_{metric}'

            data_metric = data[['epoch', metric_col_name]].dropna()

            if first_n is None:
                plt.plot(data_metric["epoch"], data_metric[metric_col_name], label=metric_col_name, marker=marker)
            else:
                plt.plot(data_train_loss["epoch"][:first_n], data_train_loss[metric_col_name][:first_n], label="Training Loss", marker=marker)

            '''annotate'''
            metric_max = data_metric[metric_col_name].max()
            metric_min = data_metric[metric_col_name].min()
            metric_max_epoch = data_metric.loc[data_metric[metric_col_name].idxmax(), 'epoch']
            metric_min_epoch = data_metric.loc[data_metric[metric_col_name].idxmin(), 'epoch']
            plt.annotate(f'Max: {metric_max:.2f}', xy=(metric_max_epoch, metric_max), xytext=(metric_max_epoch, metric_max))
                        #  arrowprops=dict(facecolor='black', shrink=0.05))
            plt.annotate(f'Min: {metric_min:.2f}', xy=(metric_min_epoch, metric_min), xytext=(metric_min_epoch, metric_min))
                        #  arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel("Epoch")
    plt.ylabel("Metrics Value")
    plt.title("Training and Validation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)

    if store_pdf:
        output_pdf_path = os.path.join(result_dir, "metrics_plot.pdf")
        plt.show()
        plt.savefig(output_pdf_path, format="pdf")
    output_pdf_path = os.path.join(result_dir, "metrics_plot.png")
    plt.show()
    plt.savefig(output_pdf_path, format="png")
    plt.close()



def ridge_fast_brainwise(X, y, lambdas=None, num_fold=5, verbose=False):
    """
    X is of shape (n_sample, n_feat): note in here all X is in shape (n_sample, n_feat)
    y is of shape (n_sample, ) / (n_sample, 1)
    b is of shape (n_feat, ) / (n_feat, 1)

    so that torch.matmul(X, b) is closest to y

    ======
    Brain-wise: for a single brain, the x is the same (for a single fold), but the y is not.

    array: 
        summ: (num_fold, num_lamb, num_vxl)
        b: (num_fold, num_lamb, num_vxl, emb_len+1)

    alg:
    for fold:
        do svd
        for lambda:
            for y:
                update summ and b

    mean_summ = mean(summ, dim=0): (num_lamb, num_vxl)
    best_lamb_idx = argmax(mean_summ, dim=0): (num_vxl)
    b_lamb = b[:, best_lamb_idx, :, :]: (num_fold, num_vxl, emb_len+1)
    b_mean = mean(b_lamb, dim=0): (num_vxl, emb_len+1)
    

    Args: all input as torch
        X: (num_img, emb_len)
        y: (num_vxl, num_img) (this is actually batched. actually for a single one is (num_img,))
        lambdas: a list
    """
    lambdas = lambdas if lambdas is not None else [0.01, 1.0, 10.0, 100.0]

    '''init'''
    device = X.device
    num_lamb = len(lambdas)
    num_img, emb_len = X.shape   # (num_img, emb_len)
    num_vxl = y.shape[0]
    ev_stats = torch.empty((num_fold, num_lamb, num_vxl), device=device)
    b_stats = torch.empty((num_fold, num_lamb, num_vxl, emb_len+1), device=device)

    '''fold split'''
    fold_size = num_img // num_fold
    indices = torch.arange(num_img, device=device)  # Simple sequential indices

    for fold in range(num_fold):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        test_indices = indices[test_start:test_end]
        train_indices = torch.cat([indices[:test_start], indices[test_end:]])
        # print('torch.max(train_indices)', torch.max(train_indices))
        # print('train_indices.shape', train_indices.shape)
        
        X_train = X[train_indices]     # (num_img_tr, emb_len)
        y_train = y[:, train_indices, None]     # (num_vxl, num_img_tr, 1)
        X_test = X[test_indices]      # (num_img_te, emb_len)
        y_test = y[:, test_indices]      # (num_vxl, num_img_te)
        # print('X_train.shape, y_train.shape, X_test.shape, y_test.shape', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        '''cal X svd'''
        X_train_aug = torch.cat([X_train, torch.ones((X_train.shape[0], 1), device=device)], dim=1)     # (num_img_tr, emb_len+1)
        X_train_aug_t = X_train_aug.transpose(-2, -1)       # (emb_len+1, num_img_tr)
        X_test_aug = torch.cat([X_test, torch.ones((X_test.shape[0], 1), device=device)], dim=1)
        # print('X_train_aug.shape, X_train_aug_t.shape, X_test_aug.shape', X_train_aug.shape, X_train_aug_t.shape, X_test_aug.shape)

        # _, S, Vh = torch.svd(X_train_aug_t)
        _, S, Vh = torch.linalg.svd(X_train_aug_t, full_matrices = False)
        e = S ** 2  # Eigenvalues
        p = torch.matmul(X_train_aug_t, y_train)
        Q = torch.matmul(X_train_aug_t, Vh.transpose(-2, -1))  # (num_vxl, n_feats, n_feats)
        r = torch.matmul(Q.transpose(-2, -1), p)

        '''for each lambda and y calculate and store b, summ'''
        for lamb_idx, lambd in enumerate(lambdas):
            b = (1.0 / lambd) * (p - torch.matmul(Q, r / (e.unsqueeze(-1) + lambd)))  # (num_vxl, n_feats, 1)
            # print('b.shape', b.shape)
            b = b.squeeze(-1)  # (num_vxl, emb_len+1)
            # print('b.shape', b.shape)
            
            b_stats[fold, lamb_idx] = b         # (num_fold, num_lamb, num_vxl, emb_len+1)

            y_pred = b @ X_test_aug.T
            # print('y_pred.shape, y_test.shape', y_pred.shape, y_test.shape)
            ev = manual_expl_var(y_pred, y_test)       # (num_nrn,)
            # print('ev.shape',ev.shape)
            ev_stats[fold, lamb_idx] = ev       # (num_fold, num_lamb, num_vxl)

    '''find the best lambda and find the best ev'''
    mean_ev = torch.mean(ev_stats, dim=0)           # (num_lamb, num_vxl)
    # print('mean_ev.shape',mean_ev.shape)        # [4, 200]
    best_lamb_idx = torch.argmax(mean_ev, dim=0)    # (num_vxl,) 
    # print('best_lamb_idx.shape',best_lamb_idx.shape) 
    fold_idx = torch.arange(num_fold, device=device)[:, None]  # (num_fold, 1)
    voxel_idx = torch.arange(num_vxl, device=device)          # (num_vxl,)
    b_best_lamb = b_stats[fold_idx, best_lamb_idx, voxel_idx, :]  # (num_fold, num_vxl, emb_len+1)
    # print('b_best_lamb.shape',b_best_lamb.shape)
    b_mean = torch.mean(b_best_lamb, dim=0)         # (num_vxl, emb_len+1)

    return b_mean
