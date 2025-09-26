from pathlib import Path
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm
import pandas as pd
from model import HyperweightsPredictorModel
from eval_models import RidgeCVBaseline
from utils import manual_expl_var, manual_normed_cos_similarity, manual_point_mse, load_weights_and_predict, is_unit_norm, unit_norm, ridge_fast_brainwise
import json

metrics = ['expl_var', 'cos_sim', 'pt_mse']

class IcltTestTime(Dataset):
    def __init__(self, ic_img, ic_nrn, uk_img, uk_nrn, gt):
        self.ic_img = torch.tensor(ic_img)
        self.ic_nrn = torch.tensor(ic_nrn)
        self.uk_img = torch.tensor(uk_img)
        self.uk_nrn = torch.tensor(uk_nrn)
        self.gt = torch.tensor(gt)

        self.num_nrn = self.uk_nrn.shape[1]

    def __len__(self):
        return self.num_nrn      # num of voxels

    def __getitem__(self, idx):
        """
        return ic_img, ic_nrn, uk_img for a single voxel
        """
        ic_img_embs = self.ic_img
        ic_vxl_resp = self.ic_nrn[:, idx]
        uk_img_embs = self.uk_img
        uk_vxl_resp = self.uk_nrn[:, idx]
        gt_weights = self.gt[idx, :]

        return ic_img_embs, ic_vxl_resp, uk_img_embs, uk_vxl_resp, gt_weights

def main(args):
    '''1. init'''
    print('Initialization ...')
    # basics
    if args.gpu is not None:
        assert torch.cuda.is_available()
        cuda_visible_devices = ','.join(map(str, args.gpu))
        print(f'[INFO] cuda_visible_devices: ', cuda_visible_devices)
    device = f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    torch.cuda.set_device(device)
    data_dir = Path(args.data_dir)
    test_subj_idx = args.test_subj_idx
    test_epoch = args.test_epoch
    batch_size = args.batch_size
    num_ic_l = args.num_ic
    print('[DEBUG] num_ic_l', num_ic_l)
    backbone_type = args.backbone_type
    print(f'[INFO] Test for different number of in context learning images: ', num_ic_l, f', Each test for{test_epoch} times')

    # results
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_csv_path = result_dir / 'test_results.csv'
    all_metrics_pd_l = []

    # model
    model_type = args.model_type
    print('[INFO] Testing model type: ', model_type)
    model_ckpt_path = args.model_ckpt_path

    # num_ic
    if args.custom_icl_idx is None:
        print('[INFO] Using default test_time_icl_idx')
        test_time_icl_idx_path = data_dir / 'test_time_icl_idx.json' 
        with open(test_time_icl_idx_path) as f:
            test_time_icl_idx_full = json.load(f)
        test_time_icl_idx_full = test_time_icl_idx_full['large_shuffled']
        # print(f'[DEBUG] len(test_time_icl_idx_full): ', len(test_time_icl_idx_full))
    else:
        print('[INFO] Using custom test_time_icl_idx')
        test_time_icl_idx_full = args.custom_icl_idx
        # print(f'[DEBUG] len(test_time_icl_idx_full): ', len(test_time_icl_idx_full))
    
    save_all = args.save_all
    if save_all:
        print('[INFO] saving voxel wise metrics')
    save_img_wise = args.save_img_wise
    if save_img_wise:
        print('[INFO] saving image wise metrics')

    '''2. load data'''
    print('Loading data ...')
    ic_img_path = data_dir / f's{test_subj_idx}_unique_img_feats_normed.npy'
    ic_nrn_path = data_dir / f's{test_subj_idx}_unique_nrn_feats_msk.npy'
    uk_img_path = data_dir / f'common_img_feats_normed.npy'
    uk_nrn_path = data_dir / f's{test_subj_idx}_common_nrn_feats_msk.npy'
    gt_path = data_dir / f's{test_subj_idx}_gt.npy'


    ic_img_full = np.load(ic_img_path).astype(np.float32)   
    ic_nrn_full = np.load(ic_nrn_path).astype(np.float32)   
    uk_img = np.load(uk_img_path).astype(np.float32)
    uk_nrn = np.load(uk_nrn_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)

    ic_img_full = unit_norm(ic_img_full)
    uk_img = unit_norm(uk_img)
    assert is_unit_norm(ic_img_full), is_unit_norm(uk_img)

    # print('[DEBUG] ic_img_full.shape, ic_nrn_full.shape, uk_img.shape, uk_nrn.shape, gt.shape',
    #                ic_img_full.shape, ic_nrn_full.shape, uk_img.shape, uk_nrn.shape, gt.shape)


    '''3. test'''
    print('Start testing ...')
    for num_ic in num_ic_l:
        diff_ep_ev_list = []
        diff_ep_cos_sim_list = []
        diff_ep_pt_mse_list = []
        diff_ep_pred_weights_mse_list = []

        diff_ep_ev_list_img_wise = []
        diff_ep_cos_sim_list_img_wise = []
        diff_ep_pt_mse_list_img_wise = []
        diff_ep_pred_weights_mse_list_img_wise = []
        test_time_icl_idx_start = 0

        '''for 5 times of testing'''
        for ep in range(test_epoch):
            print('='*40, f'NUM_IC={num_ic} - Epoch {ep} / {test_epoch-1}', '='*40)

            '''1. load data'''
            test_time_icl_idx = test_time_icl_idx_full[test_time_icl_idx_start: test_time_icl_idx_start+num_ic]
            test_time_icl_idx_start = test_time_icl_idx_start+num_ic
            ic_img = ic_img_full[test_time_icl_idx, :]   # slice by icl_img_idx
            ic_nrn = ic_nrn_full[test_time_icl_idx, :]   # slice


            '''2. predict: gets the pred nrn: (num_vxl, num_img) and pred weights (num_vxl, emb_len+1)'''
            if model_type == 'pretrained_ICL':
                true_nrn_l = []     # each ele: real response for 1000 true test images of a voxel: shape: (1000, 1)
                pred_nrn_l = []     # each element is the predicted response for a single voxel, np_array of shape (num_uk_img,), total num of ele: num_nrn
                true_weights_l = []     
                pred_weights_l = [] # each element is the predicted weights for a single voxel, np_array of shape (emb_len+1,), total num of ele: num_nrn
                
                dataset = IcltTestTime(ic_img, ic_nrn, uk_img, uk_nrn, gt)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                print('Loading model ...')
                model = HyperweightsPredictorModel(backbone_type=backbone_type).to(device)
                checkpoint = torch.load(model_ckpt_path, weights_only=False)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                '''each time of prediction'''
                print('Predicting ...') 
                with torch.inference_mode():
                    with torch.autocast(device_type="cpu" if device=="cpu" else "cuda", dtype=torch.bfloat16):
                        for ic_img_te, ic_nrn_te, uk_img_te, uk_nrn_te, gt_te in tqdm(dataloader):
                            
                            ic_img_te = ic_img_te.to(device)
                            ic_nrn_te = ic_nrn_te.to(device)
                            uk_img_te = uk_img_te.to(device)
                            uk_nrn_te = uk_nrn_te.to(device)
                            gt_te = gt_te.to(device)

                            if not save_img_wise:
                                pred_nrn, pred_weights = model(ic_img_te, ic_nrn_te, uk_img_te)
                            
                            pred_nrn, pred_weights = pred_nrn.float(), pred_weights.float()  # Convert output to float
                                
                            pred_nrn_l.append(pred_nrn)
                            pred_weights_l.append(pred_weights)
                            true_nrn_l.append(uk_nrn_te.detach().clone())
                            true_weights_l.append(gt_te.detach().clone())

                pred_nrn = torch.cat(pred_nrn_l, dim=0)     # (num_nrn, num_uk_img): [22961, 1000]
                pred_weights = torch.cat(pred_weights_l, dim=0)     # (num_nrn, emb_len+1)
                true_nrn = torch.cat(true_nrn_l, dim=0)     # (num_nrn, num_uk_img)
                true_weights = torch.cat(true_weights_l, dim=0)     # (num_nrn, emb_len+1)
                assert pred_nrn.shape == true_nrn.shape and pred_weights.shape == true_weights.shape, f'{pred_nrn.shape} == {true_nrn.shape} and {pred_weights.shape} == {true_weights.shape}'
            

            elif model_type == 'gt_weights':
                evice = torch.device('cuda')
                
                true_weights = pred_weights = torch.tensor(gt, device=device)

                uk_img_ts = torch.tensor(uk_img, device=device)
                pred_nrn = torch.cat((uk_img_ts, torch.ones((uk_img_ts.shape[0], 1), device=device)), dim=1) @ pred_weights.T

                true_nrn = uk_nrn_ts = torch.tensor(uk_nrn, device=device)
                
                pred_nrn = pred_nrn.T
                true_nrn = true_nrn.T
            
            elif model_type == 'ridge_regression':
                device = torch.device('cuda')

                '''predict weights'''
                # X: (num_img, emb_len)
                # y: (num_vxl, num_img) (this is actually batched. actually for a single one is (num_img,))
                X = torch.tensor(ic_img, device=device)
                y = torch.tensor(ic_nrn.T, device=device)

                print('lamdas ', [1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5])
                pred_weights = ridge_fast_brainwise(X, y, num_fold=5, lambdas=[1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5])

                '''true test'''
                uk_img_ts = torch.tensor(uk_img, device=device)
                pred_nrn = torch.cat((uk_img_ts, torch.ones((uk_img_ts.shape[0], 1), device=device)), dim=1) @ pred_weights.T

                '''pack for testing'''
                true_nrn = uk_nrn_ts = torch.tensor(uk_nrn, device=device)
                true_weights = gt_ts = torch.tensor(gt, device=device)

                pred_nrn = pred_nrn.T
                true_nrn = true_nrn.T

            else:
                raise ValueError('Unknown Model Type')

            '''calculate metrics'''

            '''cal metrics'''
            print('Calculating metrics ...')
            # print('[DEBUG] pred_nrn.shape, true_nrn.shape', pred_nrn.shape, true_nrn.shape)
            vxl_wise_ev = manual_expl_var(pred_nrn, true_nrn)       #（num_vxl,)
            print('vxl_wise_ev.shape', vxl_wise_ev.shape)
            vxl_wise_mse = manual_point_mse(pred_nrn, true_nrn)     # (num_vxl,)
            vxl_wise_cos_sim = manual_normed_cos_similarity(pred_weights, true_weights) # (num_vxl,)
            pred_weights_mse = manual_point_mse(pred_weights, true_weights) # (num_vxl,)
            diff_ep_ev_list.append(vxl_wise_ev.unsqueeze(0))
            diff_ep_cos_sim_list.append(vxl_wise_cos_sim.unsqueeze(0))
            diff_ep_pt_mse_list.append(vxl_wise_mse.unsqueeze(0))
            diff_ep_pred_weights_mse_list.append(pred_weights_mse.unsqueeze(0))

            if save_img_wise:
                img_wise_ev = manual_expl_var(pred_nrn.T, true_nrn.T)       #（num_vxl,)
                img_wise_mse = manual_point_mse(pred_nrn.T, true_nrn.T)     # (num_vxl,)
                img_wise_cos_sim = manual_normed_cos_similarity(pred_weights.T, true_weights.T) # (num_vxl,)
                # print(img_wise_ev.shape)
                # assert False
                # assert img_wise_ev.shape[0] == 907, img_wise_ev.shape
                diff_ep_ev_list_img_wise.append(img_wise_ev[None, :])
                diff_ep_cos_sim_list_img_wise.append(img_wise_cos_sim[None, :])
                diff_ep_pt_mse_list_img_wise.append(img_wise_mse[None, :])

                pred_nrn_np = pred_nrn.cpu().numpy()
                np.save(result_dir / f'num_ic={num_ic}_ep={ep}_pred_nrn.npy', pred_nrn_np)

            print('torch.mean(vxl_wise_ev): ', torch.mean(vxl_wise_ev))

            '''store pred weights'''
            pred_weights_np = pred_weights.cpu().numpy()
            np.save(result_dir / f'num_ic={num_ic}_ep={ep}_pred_weights.npy', pred_weights_np)

        '''save at the end of many epochs for same num_ic'''
        print('Saving ...')
        same_num_ic_diff_epoch_ev = torch.cat(diff_ep_ev_list, dim=0)     # (num_test_epoch, num_voxel)
        same_num_ic_diff_epoch_cos_sim = torch.cat(diff_ep_cos_sim_list, dim=0)
        same_num_ic_diff_epoch_pt_mse = torch.cat(diff_ep_pt_mse_list, dim=0)
        same_num_ic_diff_epoch_pred_weights_mse = torch.cat(diff_ep_pred_weights_mse_list, dim=0)
        same_num_ic_diff_epoch_avg_ev = torch.mean(same_num_ic_diff_epoch_ev, dim=1)    # (num_test_epoch,)
        same_num_ic_diff_epoch_avg_cos_sim = torch.mean(same_num_ic_diff_epoch_cos_sim, dim=1)    # (num_test_epoch,)
        same_num_ic_diff_epoch_avg_pt_mse = torch.mean(same_num_ic_diff_epoch_pt_mse, dim=1)    # (num_test_epoch,)
        same_num_ic_diff_epoch_avg_pred_weights_mse = torch.mean(same_num_ic_diff_epoch_pred_weights_mse, dim=1)    # (num_test_epoch,)

        if save_img_wise:
            same_num_ic_diff_epoch_ev_img_wise = torch.cat(diff_ep_ev_list_img_wise, dim=0)     # (num_test_epoch, num_voxel)
            same_num_ic_diff_epoch_cos_sim_img_wise = torch.cat(diff_ep_cos_sim_list_img_wise, dim=0)
            same_num_ic_diff_epoch_pt_mse_img_wise = torch.cat(diff_ep_pt_mse_list_img_wise, dim=0)


        '''write mean value to csv'''
        for ep in range(test_epoch):
            all_metrics_pd_l.append({
                'num_ic': num_ic,
                'epoch': ep,
                'avg_explained_variance': same_num_ic_diff_epoch_avg_ev[ep].cpu().item(),
                'avg_cosine_similarity': same_num_ic_diff_epoch_avg_cos_sim[ep].cpu().item(),
                'avg_point_mse': same_num_ic_diff_epoch_avg_pt_mse[ep].cpu().item()
            })
        results_df = pd.DataFrame(all_metrics_pd_l)
        results_df = results_df[['num_ic', 'epoch', 'avg_explained_variance', 'avg_cosine_similarity', 'avg_point_mse']]
        results_df.to_csv(result_csv_path, index=False)
        print(f'[INFO]: results_df:\n', results_df)

        if save_all:
            np.save(result_dir / f'num_ic={num_ic}_ev.npy', same_num_ic_diff_epoch_ev.cpu().numpy())
            np.save(result_dir / f'num_ic={num_ic}_cos_sim.npy', same_num_ic_diff_epoch_cos_sim.cpu().numpy())
            np.save(result_dir / f'num_ic={num_ic}_weights_mse.npy', same_num_ic_diff_epoch_avg_pred_weights_mse.cpu().numpy())
            np.save(result_dir / f'num_ic={num_ic}_pt_mse.npy', same_num_ic_diff_epoch_pt_mse.cpu().numpy())
        else:
            raise ValueError('save_all is False, not saving all results')

        if save_img_wise:
            np.save(result_dir / f'num_ic={num_ic}_ev_img_wise.npy', same_num_ic_diff_epoch_ev_img_wise.cpu().numpy())
            np.save(result_dir / f'num_ic={num_ic}_cos_sim_img_wise.npy', same_num_ic_diff_epoch_cos_sim_img_wise.cpu().numpy())
            np.save(result_dir / f'num_ic={num_ic}_pt_mse_img_wise.npy', same_num_ic_diff_epoch_pt_mse_img_wise.cpu().numpy())

        
if __name__ == '__main__':
    """read argument"""
    parser = argparse.ArgumentParser(description='simple distributed training job')

    parser.add_argument('--data_dir', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--backbone_type', type=str, default=None, help='CUDA visible devices (default: 0)')
    parser.add_argument('--test_subj_idx', type=int, help='CUDA visible devices (default: 0)')
    parser.add_argument('--test_epoch', type=int, help='CUDA visible devices (default: 0)')
    parser.add_argument('--result_dir', type=str, default=None, help='CUDA visible devices (default: 0)')
    parser.add_argument('--custom_icl_idx', type=int, nargs='+', default=None, help='CUDA visible devices (default: 0)')

    parser.add_argument('--num_ic', nargs='+', type=int, help='CUDA visible devices (default: 0)')

    parser.add_argument('--model_type', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--model_ckpt_path', type=str, help='CUDA visible devices (default: 0)')
    parser.add_argument('--gpu', nargs='+', type=int, default=None, help='CUDA visible devices (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='CUDA visible devices (default: 0)')
    
    parser.add_argument('--save_all', action='store_true', help='Save all results')
    parser.add_argument('--save_img_wise', action='store_true', default=False, help='Save all results')

    args = parser.parse_args()

    main(args)