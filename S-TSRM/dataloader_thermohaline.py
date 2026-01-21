from torch.utils.data import Dataset
import numpy as np
import torch
import os
import xarray as xr


class Thermohaline_dataset(Dataset):
    def __init__(self, data_root, data_name, training_time, step,
                 mean=None, std=None):
        super().__init__()

        self.data_name = data_name
        self.start_index = training_time[0]
        self.end_index = training_time[1]
        self.step = step
        self.mean = mean
        self.std = std
        # self.var_list1 = ['adt', 'east_stress', 'north_stress', 'sla', 'ssd', 'sss', 'sss_err', 'sst_oi', 'sst_anom', 'ugos','vgos']
        # self.var_list2 = ['adt', 'east_stress', 'north_stress', 'sla', 'ssd', 'sss', 'sss_err', 'sst_mo', 'sst_anom','ugos','vgos']
        # data_list = []
        # for var in self.var_list1:
        #     var_path = os.path.join(data_root, var + r'.npy')
        #     var_data = np.load(var_path)
        #     sel_var_data = var_data[self.start_index:self.end_index, :, :]
        #     sel_var_data_min = sel_var_data.min(axis=(1, 2))
        #     sel_var_data_max = sel_var_data.max(axis=(1, 2))
        #     sel_var_data_diff = sel_var_data_max - sel_var_data_min
        #     for i in range(len(sel_var_data_diff)):
        #         if sel_var_data_diff[i] == 0:
        #             sel_var_data_diff[i] = 1
        #     sel_var_data_min=sel_var_data_min[:,np.newaxis,np.newaxis]
        #     sel_var_data_diff = sel_var_data_diff[:, np.newaxis, np.newaxis]
        #     limited_var = (sel_var_data - sel_var_data_min) / sel_var_data_diff
        #     fill_nan_limited_var = np.nan_to_num(limited_var, nan=0.0)
        #     data_list.append(fill_nan_limited_var)
        #
        # lon_lat_cor_path = os.path.join(data_root, r'lon_lat_cor.npy')
        # lon_lat_cor = np.load(lon_lat_cor_path)[self.start_index:self.end_index, :, :, :]
        # cor_min = lon_lat_cor.min(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
        # cor_max = lon_lat_cor.max(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
        # cor_diff = cor_max - cor_min
        # limited_cor = (lon_lat_cor - cor_min) / cor_diff
        #
        # data_list_np = np.array(data_list)
        # data_list_np = data_list_np.transpose((1, 0, 2, 3))
        # features = np.concatenate((data_list_np, limited_cor), axis=1)
        # features=features[:,np.newaxis,:,:,:]
        #
        # temp_label_path = os.path.join(data_root, r'temp_arr.npy')
        # salt_label_path = os.path.join(data_root, r'salt_arr.npy')
        # label_temp = np.load(temp_label_path)[self.start_index:self.end_index, :, :, :]
        # label_salt = np.load(salt_label_path)[self.start_index:self.end_index, :, :, :]
        #
        # temp_min = label_temp.min(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
        # temp_max = label_temp.max(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
        # temp_diff = temp_max - temp_min
        # label_temp = (label_temp - temp_min) / temp_diff
        # _label_temp = np.nan_to_num(label_temp, nan=0.0)
        #
        # salt_min = label_salt.min(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
        # salt_max = label_salt.max(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
        # salt_diff = salt_max - salt_min
        # label_salt = (label_salt - salt_min) / salt_diff
        # _label_salt = np.nan_to_num(label_salt, nan=0.0)
        #
        # target = np.concatenate((_label_temp, _label_salt), axis=1)
        # target=target[:,np.newaxis,:,:,:]
        dataset_input = xr.open_dataset(data_root + r'/map_processed_input_2005-2019.nc')
        dataset_label = xr.open_dataset(data_root + r'/processed_label_2005-2019_2000m.nc')
        dataset_input_sel = dataset_input.sel(time=range(self.start_index, self.end_index))
        dataset_label_sel = dataset_label.sel(time=range(self.start_index, self.end_index))
        # lon_lat_cor=np.load(data_root+r'/pro_map_lon_lat_cor.npy')
        inputs_data = []
        # inputs_data.append(lon_lat_cor[self.start_index:self.end_index,np.newaxis,:,:])
        var_list1 = ['lon_lat_cor','east_wind','north_wind', 'east_stress','north_stress',
                     'adt', 'sla', 'ssd','sss','noaa_sst']
        map_list1 = ['adt', 'east_stress', 'north_stress', 'sla', 'ssd', 'sss', 'sst_oi', 'ugos',
                     'vgos', 'east_wind_map', 'north_wind_map']
        var_list2 = ['adt', 'east_stress', 'north_stress', 'sla', 'ssd', 'sss','sst_mo', 'ugos',
                     'vgos','east_wind','north_wind']
        for var in var_list1:
            inputs_data.append(dataset_input_sel[var].values[:, np.newaxis, :, :])
        features = np.concatenate(inputs_data, axis=1)[:,:,np.newaxis, :, :]
        label_data = []
        # label_list = ['temp_arr', 'salt_arr']
        label_list = ['temp_arr_2000m']
        for label in label_list:
            label_data.append(dataset_label_sel[label].values)
        target = np.concatenate(label_data, axis=1)[:, :,np.newaxis, :, :]
        self.data = [features, target]
    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        data = torch.tensor(self.data[0][index])
        label = torch.tensor(self.data[1][index])
        return data, label