"""
Temp
"""

preds_path=r'preds.npy'
trues_path=r'trues.npy'

import numpy as np
import matplotlib.pyplot as plt
preds_data=np.load(preds_path,'r')
trues_data=np.load(trues_path,'r')
print(preds_data.shape,trues_data.shape)
preds=np.squeeze(preds_data,axis=2)
trues=np.squeeze(trues_data,axis=2)

salt_label_path=r'F:\label_data\salt_arr_2000m.npy'
temp_label_path=r'F:\label_data\temp_arr_2000m.npy'

salt_data=np.load(salt_label_path,'r')
temp_data=np.load(temp_label_path,'r')

salt_tar=salt_data[168:180]
temp_tar=temp_data[168:180]
salt_min=np.nanmin(salt_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
salt_max=np.nanmax(salt_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
salt_diff=salt_max-salt_min
temp_min=np.nanmin(temp_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
temp_max=np.nanmax(temp_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
temp_diff=temp_max-temp_min

total_diff=np.concatenate((temp_diff,salt_diff),axis=1)[:,:27]

total_min=np.concatenate((temp_min,salt_min),axis=1)[:,:27]

total_tar=np.concatenate((temp_tar,salt_tar),axis=1)

map_temp_path=r'F:\label_data\map_temp_nan.npy'
map_salt_path=r'F:\label_data\map_salt_nan.npy'
map_temp=np.load(map_temp_path,'r')
map_salt=np.load(map_salt_path,'r')
map_temp_tar=map_temp[168:180]
map_salt_tar=map_salt[168:180]
map_tar=np.concatenate((map_temp_tar,map_salt_tar),axis=1)[:,:27]

preds_nan=preds*map_tar
trues_nan=trues*map_tar

preds_out=preds_nan*total_diff+total_min
trues_out=trues_nan*total_diff+total_min

differs_out=trues_out-preds_out

def R2(pred,true):
    from sklearn.metrics import r2_score
    mask = ~np.isnan(pred) & ~np.isnan(true)
    pred_no_nan=pred[mask]
    true_no_nan=true[mask]
    return r2_score(true_no_nan,pred_no_nan)
temp_r2=[[] for _ in range(12)]
for i in range(12):
    for j in range(27):
        temp_r2_unit=R2(preds_out[i,j,:,:],trues_out[i,j,:,:])
        temp_r2[i].append(temp_r2_unit)
temp_model_r2=np.array(temp_r2)

def RMSE(pred, true):
    mask = ~np.isnan(pred) & ~np.isnan(true)
    pred_no_nan=pred[mask]
    true_no_nan=true[mask]
    diff_square=(pred_no_nan-true_no_nan)**2
    return np.sqrt(np.mean(diff_square))

temp_rmse=[[] for _ in range(12)]
for i in range(12):
    for j in range(27):
        temp_rmse_unit=RMSE(preds_out[i,j,:,:],trues_out[i,j,:,:])
        temp_rmse[i].append(temp_rmse_unit)

# salt_rmse=[[] for _ in range(12)]
# for i in range(12):
#     for j in range(20,40):
#         salt_rmse_unit=RMSE(preds_out[i,j,:,:],trues_out[i,j,:,:])
#         salt_rmse[i].append(salt_rmse_unit)

temp_model_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)
print(temp_model_rmse.shape)


"""
Salt
"""

preds_path=r'preds.npy'
trues_path=r'trues.npy'

import numpy as np
import matplotlib.pyplot as plt
preds_data=np.load(preds_path,'r')
trues_data=np.load(trues_path,'r')
print(preds_data.shape,trues_data.shape)
preds=np.squeeze(preds_data,axis=2)
trues=np.squeeze(trues_data,axis=2)

salt_label_path=r'F:\label_data\salt_arr_2000m.npy'
temp_label_path=r'F:\label_data\temp_arr_2000m.npy'

salt_data=np.load(salt_label_path,'r')
temp_data=np.load(temp_label_path,'r')

salt_tar=salt_data[168:180]
temp_tar=temp_data[168:180]
salt_min=np.nanmin(salt_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
salt_max=np.nanmax(salt_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
salt_diff=salt_max-salt_min
temp_min=np.nanmin(temp_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
temp_max=np.nanmax(temp_tar,axis=(2,3))[:, :, np.newaxis, np.newaxis]
temp_diff=temp_max-temp_min

total_diff=np.concatenate((temp_diff,salt_diff),axis=1)[:,27:]

total_min=np.concatenate((temp_min,salt_min),axis=1)[:,27:]

total_tar=np.concatenate((temp_tar,salt_tar),axis=1)

map_temp_path=r'F:\label_data\map_temp_nan.npy'
map_salt_path=r'F:\label_data\map_salt_nan.npy'
map_temp=np.load(map_temp_path,'r')
map_salt=np.load(map_salt_path,'r')
map_temp_tar=map_temp[168:180]
map_salt_tar=map_salt[168:180]
map_tar=np.concatenate((map_temp_tar,map_salt_tar),axis=1)[:,27:]

preds_nan=preds*map_tar
trues_nan=trues*map_tar

preds_out_salt=preds_nan*total_diff+total_min
trues_out_salt=trues_nan*total_diff+total_min

differs_out_salt=trues_out_salt-preds_out_salt

def R2(pred,true):
    from sklearn.metrics import r2_score
    mask = ~np.isnan(pred) & ~np.isnan(true)
    pred_no_nan=pred[mask]
    true_no_nan=true[mask]
    return r2_score(true_no_nan,pred_no_nan)
temp_r2=[[] for _ in range(12)]
for i in range(12):
    for j in range(27):
        temp_r2_unit=R2(preds_out_salt[i,j,:,:],trues_out_salt[i,j,:,:])
        temp_r2[i].append(temp_r2_unit)
temp_model_r2=np.array(temp_r2)

def RMSE(pred, true):
    mask = ~np.isnan(pred) & ~np.isnan(true)
    pred_no_nan=pred[mask]
    true_no_nan=true[mask]
    diff_square=(pred_no_nan-true_no_nan)**2
    return np.sqrt(np.mean(diff_square))

temp_rmse=[[] for _ in range(12)]
for i in range(12):
    for j in range(27):
        temp_rmse_unit=RMSE(preds_out_salt[i,j,:,:],trues_out_salt[i,j,:,:])
        temp_rmse[i].append(temp_rmse_unit)

# salt_rmse=[[] for _ in range(12)]
# for i in range(12):
#     for j in range(20,40):
#         salt_rmse_unit=RMSE(preds_out[i,j,:,:],trues_out[i,j,:,:])
#         salt_rmse[i].append(salt_rmse_unit)

salt_model_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)
print(salt_model_rmse.shape)

#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author thesky

#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author thesky
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'#将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度线方向设置向内
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 temp_rmse 和 salt_rmse 已经存在

# 创建 DataFrame
depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
          600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df_temp = pd.DataFrame(temp_rmse, columns=months, index=depths)
df_salt = pd.DataFrame(salt_rmse, columns=months, index=depths)

# 创建画布和子图（左右排列）
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  # 1行2列
sns.set(font_scale=1.2)

# ---------- 绘制温度热力图 ----------
heatmap=sns.heatmap(df_temp, ax=axes[0], cmap='YlGnBu', annot=False, fmt='.2f',
            cbar_kws={'extend': 'both','label':'RMSE(\u00B0C)'})
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(direction='in')
axes[0].set_xlabel('Month', fontsize=15)
axes[0].set_ylabel('Depth (m)', fontsize=15)
axes[0].tick_params(axis='x', direction='in')
axes[0].tick_params(axis='y', direction='in')
axes[0].set_title('Temperature', fontsize=18)
axes[0].text(-0.05, 1.05, '(a)', transform=axes[0].transAxes,
             fontsize=20, fontweight='bold', va='top', ha='right')
# ---------- 绘制盐度热力图 ----------
heatmap=sns.heatmap(df_salt, ax=axes[1], cmap='YlOrRd', annot=False, fmt='.2f',
            cbar_kws={'extend': 'both','label':'RMSE (psu)'})
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(direction='in')
axes[1].set_xlabel('Month', fontsize=15)
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', direction='in')
axes[1].tick_params(axis='y', direction='in')
axes[1].set_title('Salinity', fontsize=18)
axes[1].text(-0.05, 1.05, '(b)', transform=axes[1].transAxes,
             fontsize=20, fontweight='bold', va='top', ha='right')
# plt.rcParams['font.family'] = 'SimHei'
plt.tight_layout()

# plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\t_s_temporal_rmse.png',dpi=600,bbox_inches='tight')
plt.show()