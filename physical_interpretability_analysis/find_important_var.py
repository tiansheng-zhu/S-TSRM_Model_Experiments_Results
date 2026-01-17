from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from dataloader_thermohaline import Thermohaline_dataset
from model_2 import Thermohaline_Model
import numpy as np
from collections import OrderedDict
from utils import *
import os.path as osp
import torch.nn as nn
from dataloader_thermohaline import Thermohaline_dataset
from functions import train_one_epoch, valid_one_epoch,test_one_epoch
from torch.utils.data import DataLoader
from typing import Dict, List, Union
import torch
depths1 = [0., 5., 10.]
depths2 = [0., 5., 10., 20., 30., 50.]
depths3 = [0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.]
import xarray as xr
device=torch.device('cpu')
# from train import get_input_dataset
def get_input_dataset(add_noise_var):
    data_root = r'./data'
    # var_list1 = ['adt', 'east_stress', 'north_stress', 'sla', 'ssd', 'sss', 'sss_err', 'sst_oi', 'sst_anom',
    #              'ugos', 'vgos']
    # var_list2 = ['adt', 'east_stress', 'north_stress', 'sla', 'ssd', 'sss', 'sss_err', 'sst_mo', 'sst_anom',
    #              'ugos', 'vgos']
    var_list1 = ['lon_lat_cor', 'east_wind', 'north_wind',
                 'adt', 'sss','noaa_sst']
    dataset_input = xr.open_dataset(data_root + r'/map_processed_input_2005-2019.nc')
    # dataset_label = xr.open_dataset(data_root + r'/processed_label_2005-2019_2000m.nc')
    dataset_input_sel = dataset_input.sel(time=range(168,180))
    # dataset_label_sel = dataset_label.sel(time=range(start_index, end_index))
    data_list = []
    for var in var_list1:
        if add_noise_var == var:
            sel_var_data=dataset_input_sel[var].values[:, np.newaxis, :, :]
            noise = np.random.normal(0.0, 0.5, sel_var_data.shape)
            sel_var_data += noise
            sel_var_data_min = np.nanmin(sel_var_data, axis=(2, 3))
            sel_var_data_max = np.nanmax(sel_var_data, axis=(2, 3))
            sel_var_data_diff = sel_var_data_max - sel_var_data_min
            sel_var_data_min = sel_var_data_min[:, :,np.newaxis, np.newaxis]
            sel_var_data_diff = sel_var_data_diff[:, :,np.newaxis, np.newaxis]
            limited_var = (sel_var_data - sel_var_data_min) / sel_var_data_diff
            fill_nan_limited_var = np.nan_to_num(limited_var, nan=0.0)
            data_list.append(sel_var_data)
        else:
            data_list.append(dataset_input_sel[var].values[:, np.newaxis, :, :])
    features = np.concatenate(data_list, axis=1)[:, :, np.newaxis, :, :]
    return features

def main():

    model = Thermohaline_Model(in_shape=(6, 1, 180, 360), hid_S=64, hid_T=256, N_S=2, mlp_ratio=4, drop=0,
                               drop_path=0.1, spatio_kernel_dec=(3, 3), spatio_kernel_enc=(3, 3),
                               Depth_out1=len(depths1), Depth_out2=len(depths2), Depth_out3=len(depths3)).to(
        device)
    # model_dir, log_dir, save_dir, plot_dir, best_checkpoint, interval_checkpoint = make_dir('./results_exp2_E_64_T_256_2_3_4_no_pred_train_cosine_lr')
    print('>' * 35 + ' testing ' + '<' * 35)

    best_model_path = r'your_trained_model_state_dict'
    model.load_state_dict(torch.load(best_model_path))
    print('>' * 35 + 'find_important_var' + '<' * 35)
    model.eval()
    var_list1 = ['east_wind', 'north_wind',
                 'adt', 'noaa_sst', 'sss','lon_lat_cor']
    for var in var_list1:
        for t in range(10):
            preds_list = []
            features = get_input_dataset(add_noise_var=var)
            for i in range(12):
                f = features[i, :, :, :, :]
                fea = torch.tensor(f[np.newaxis, :, :, :, :]).float().to(device)
                _,_,pred_y = model(fea)
                preds_list.append(pred_y.detach().cpu().numpy())
                print(f'{var}_{t + 1}次_{i + 1}月')
            preds = np.concatenate(preds_list, axis=0)
            np.save(f'./data/salt/{var}_{t + 1}_{i + 1}.npy', preds)
        print(f'{var}_complete!')
    print('complete!')

if __name__ == '__main__':
    main()