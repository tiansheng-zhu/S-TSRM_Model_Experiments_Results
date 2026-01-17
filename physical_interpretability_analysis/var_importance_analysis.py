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
np.mean(temp_model_rmse,axis=0)

"""
Salt
"""
preds_path=r'F:preds.npy'
trues_path=r'F:trues.npy'

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


salt_trues_path=r'trues.npy'
salt_trues_data=np.load(salt_trues_path,'r')
salt_trues=np.squeeze(salt_trues_data,axis=2)
salt_trues_nan=salt_trues*np.concatenate((map_temp_tar,map_salt_tar),axis=1)[:,27:]
salt_trues_out=salt_trues_nan*np.concatenate((temp_diff,salt_diff),axis=1)[:,27:]+np.concatenate((temp_min,salt_min),axis=1)[:,27:]

temp_trues_out=trues_out

import os

temp_preds_prefix = r'E:\pythonproject\Model\data\temp'
salt_preds_prefix = r'E:\pythonproject\Model\data\salt'
var_list = ['adt', 'east_wind', 'north_wind', 'noaa_sst', 'sss', 'lon_lat_cor']
var_rmse_dict = {}
for var in var_list:
    rmse = {}
    rmse['temp'] = []
    rmse['salt'] = []
    for i in range(10):
        temp_path = os.path.join(temp_preds_prefix, var + r'_' + str(i + 1) + r'_12.npy')
        salt_path = os.path.join(salt_preds_prefix, var + r'_' + str(i + 1) + r'_12.npy')

        temp_data = np.load(temp_path)
        salt_data = np.load(salt_path)

        temp_s = np.squeeze(temp_data, axis=2)
        salt_s = np.squeeze(salt_data, axis=2)

        temp_map = temp_s * np.concatenate((map_temp_tar, map_salt_tar), axis=1)[:, :27]
        salt_map = salt_s * np.concatenate((map_temp_tar, map_salt_tar), axis=1)[:, 27:]

        temp_preds = temp_map * np.concatenate((temp_diff, salt_diff), axis=1)[:, :27] + np.concatenate(
            (temp_min, salt_min), axis=1)[:, :27]
        salt_preds = salt_map * np.concatenate((temp_diff, salt_diff), axis=1)[:, 27:] + np.concatenate(
            (temp_min, salt_min), axis=1)[:, 27:]
        temp_rmse = [[] for _ in range(12)]
        salt_rmse = [[] for _ in range(12)]

        for i in range(12):
            for j in range(27):
                temp_rmse_unit = RMSE(temp_preds[i, j, :, :], temp_trues_out[i, j, :, :])
                salt_rmse_unit = RMSE(salt_preds[i, j, :, :], salt_trues_out[i, j, :, :])
                temp_rmse[i].append(temp_rmse_unit)
                salt_rmse[i].append(salt_rmse_unit)
        temp_rmse = np.array(temp_rmse)
        salt_rmse = np.array(salt_rmse)
        rmse['temp'].append(temp_rmse)
        rmse['salt'].append(salt_rmse)
    rmse['temp'] = np.array(rmse['temp'])
    rmse['salt'] = np.array(rmse['salt'])
    var_rmse_dict[var] = rmse

var_rmse_dict

depth = np.array([0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.])
colors = ['red', 'orange', 'green', 'cyan', 'blue', 'purple']
labels = ['ADT', 'USSW', 'VSSW', 'SST', 'SSS', 'LON_LAT_COR']
model_mean=np.mean(temp_model_rmse,axis=0)
model_mean

adt_mean=np.mean(var_rmse_dict['adt']['temp'],axis=(0,1))
uwind_mean=np.mean(var_rmse_dict['east_wind']['temp'],axis=(0,1))
vwind_mean=np.mean(var_rmse_dict['north_wind']['temp'],axis=(0,1))
sst_mean=np.mean(var_rmse_dict['noaa_sst']['temp'],axis=(0,1))
sss_mean=np.mean(var_rmse_dict['sss']['temp'],axis=(0,1))
lon_lat_cor_mean=np.mean(var_rmse_dict['lon_lat_cor']['temp'],axis=(0,1))

adt_mean_sub=adt_mean-model_mean
uwind_mean_sub=uwind_mean-model_mean
vwind_mean_sub=vwind_mean-model_mean
sst_mean_sub=sst_mean-model_mean
sss_mean_sub=sss_mean-model_mean
lon_lat_cor_mean_sub=lon_lat_cor_mean-model_mean

total_sub=adt_mean_sub+uwind_mean_sub+vwind_mean_sub+sst_mean_sub+sss_mean_sub+lon_lat_cor_mean_sub
total_sub

adt_mean_per=adt_mean_sub/total_sub
uwind_mean_per=uwind_mean_sub/total_sub
vwind_mean_per=vwind_mean_sub/total_sub
sst_mean_per=sst_mean_sub/total_sub
sss_mean_per=sss_mean_sub/total_sub
lon_lat_cor_mean_per=lon_lat_cor_mean_sub/total_sub

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内
# 输入数据（保持不变）
depth = [0., 5., 10., 20., 30., 50., 75., 100., 125., 150., 200., 250., 300., 400., 500., 600., 700., 800.,
         900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.]

# 转换为numpy数组
depth = np.array(depth)
variables = np.array(
    [adt_mean_per * 100, uwind_mean_per * 100, vwind_mean_per * 100, sst_mean_per * 100, sss_mean_per * 100,
     lon_lat_cor_mean_per * 100])

# 验证数据完整性
assert np.allclose(variables.sum(axis=0), 100, atol=1e-6), "变量之和必须为100%"

# 定义颜色和标签
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']
labels = ['ADT', 'USSW', 'VSSW', 'SST', 'SSS', 'LON_LAT_COR']

plt.figure(figsize=(15, 10))

# 绘制每个深度区间
for i in range(len(depth)):
    # 确定纵向范围
    if i < len(depth) - 1:
        y_start = depth[i]
        y_end = depth[i + 1]
    else:  # 处理最后一个深度
        y_start = depth[i]
        y_end = depth[i] + np.mean(np.diff(depth))

    # 当前深度的变量值
    values = variables[:, i]

    # 计算累积宽度
    cum_width = np.cumsum([0] + values.tolist())

    # 绘制每个变量块
    for j in range(6):
        plt.broken_barh(
            [(cum_width[j], cum_width[j + 1] - cum_width[j])],  # X范围
            (y_start, y_end - y_start),  # Y范围
            facecolors=colors[j],
            edgecolor='white',
            linewidth=0.3
        )

# 坐标轴设置
ax = plt.gca()
ax.invert_yaxis()
# ax.xaxis.set_ticks_position('top')  # X轴刻度在上方
# ax.xaxis.set_label_position('top')  # X轴标签在上方
plt.xlim(0, 100)
plt.ylim(depth[-1] + np.mean(np.diff(depth)), 0)  # 显示完整范围
plt.xlabel('Relative Importance(%)', labelpad=2, fontsize=14)
plt.ylabel('Depth(m)', fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

# 设置y轴间隔
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(50))

# 创建图例（底部水平排列）
legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(6)]
legend = ax.legend(
    handles=legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=6,
    frameon=False,  # 去掉边框
    handletextpad=0.5,
    columnspacing=1.5,
    fontsize=12
)

# 调整布局
plt.subplots_adjust(bottom=0.2)  # 为底部图例留出空间

# 设置图表标题
plt.title('Temperature', fontsize=18)
plt.tight_layout()
# plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\temp_var_sensitivity_pic_v2.png',dpi=600,bbox_inches='tight')
plt.show()

salt_adt_mean=np.mean(var_rmse_dict['adt']['salt'],axis=(0,1))
salt_uwind_mean=np.mean(var_rmse_dict['east_wind']['salt'],axis=(0,1))
salt_vwind_mean=np.mean(var_rmse_dict['north_wind']['salt'],axis=(0,1))
salt_sst_mean=np.mean(var_rmse_dict['noaa_sst']['salt'],axis=(0,1))
salt_sss_mean=np.mean(var_rmse_dict['sss']['salt'],axis=(0,1))
salt_lon_lat_cor_mean=np.mean(var_rmse_dict['lon_lat_cor']['salt'],axis=(0,1))

salt_model_mean=np.mean(salt_model_rmse,axis=0)
salt_model_mean

salt_adt_mean_sub=salt_adt_mean-salt_model_mean
salt_uwind_mean_sub=salt_uwind_mean-salt_model_mean
salt_vwind_mean_sub=salt_vwind_mean-salt_model_mean
salt_sst_mean_sub=salt_sst_mean-salt_model_mean
salt_sss_mean_sub=salt_sss_mean-salt_model_mean
salt_lon_lat_cor_mean_sub=salt_lon_lat_cor_mean-salt_model_mean

salt_total_sub=salt_adt_mean_sub+salt_uwind_mean_sub+salt_vwind_mean_sub+salt_sst_mean_sub+salt_sss_mean_sub+salt_lon_lat_cor_mean_sub
salt_total_sub

salt_adt_mean_per=salt_adt_mean_sub/salt_total_sub
salt_uwind_mean_per=salt_uwind_mean_sub/salt_total_sub
salt_vwind_mean_per=salt_vwind_mean_sub/salt_total_sub
salt_sst_mean_per=salt_sst_mean_sub/salt_total_sub
salt_sss_mean_per=salt_sss_mean_sub/salt_total_sub
salt_lon_lat_cor_mean_per=salt_lon_lat_cor_mean_sub/salt_total_sub

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# 输入数据
depth = [0., 5., 10., 20., 30., 50., 75., 100., 125., 150., 200., 250., 300., 400., 500., 600., 700., 800.,
         900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.]
# 转换为numpy数组并验证
depth = np.array(depth)
variables = np.array([salt_adt_mean_per * 100, salt_uwind_mean_per * 100, salt_vwind_mean_per * 100,
                      salt_sst_mean_per * 100, salt_sss_mean_per * 100, salt_lon_lat_cor_mean_per * 100])
assert np.allclose(variables.sum(axis=0), 100, atol=1e-6), "变量之和必须为100%"

# 可视化设置
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']
labels = ['ADT', 'USSW', 'VSSW', 'SST', 'SSS', 'LON_LAT_COR']

plt.figure(figsize=(15, 10))
ax = plt.gca()

# 绘制每个深度区间
for i in range(len(depth)):
    # 确定纵向范围
    y_start = depth[i]
    y_end = depth[i + 1] if i < len(depth) - 1 else depth[i] + np.mean(np.diff(depth))

    # 当前深度的变量值
    values = variables[:, i]
    cum_width = np.cumsum([0] + values.tolist())

    # 绘制每个变量块
    for j in range(6):
        plt.broken_barh(
            [(cum_width[j], cum_width[j + 1] - cum_width[j])],
            (y_start, y_end - y_start),
            facecolors=colors[j],
            edgecolor='white',
            linewidth=0.3
        )

# 坐标轴设置
ax = plt.gca()
ax.invert_yaxis()
# ax.xaxis.set_ticks_position('top')  # X轴刻度在上方
# ax.xaxis.set_label_position('top')  # X轴标签在上方
plt.xlim(0, 100)
plt.ylim(depth[-1] + np.mean(np.diff(depth)), 0)  # 显示完整范围
plt.xlabel('Relative Importance(%)', labelpad=2, fontsize=14)
plt.ylabel('Depth(m)', fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

# 设置y轴间隔
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(50))

# 创建图例（底部水平排列）
legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(6)]
legend = ax.legend(
    handles=legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=6,
    frameon=False,  # 去掉边框
    handletextpad=0.5,
    columnspacing=1.5,
    fontsize=12
)

# 调整布局
plt.subplots_adjust(bottom=0.2)  # 为底部图例留出空间

# 设置图表标题
plt.title('Salinity', fontsize=18)
plt.tight_layout()
# plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\salt_var_sensitivity_pic_v2.png',dpi=600,bbox_inches='tight')
plt.show()