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
np.mean(temp_model_rmse)

"""
Salt
"""

preds_path=r'F:\label_data\exp_for_sss\salt_6input_sss_last_2_3_3_10_50_early_stop_200\saved\preds.npy'
trues_path=r'F:\label_data\exp_for_sss\salt_6input_sss_last_2_3_3_10_50_early_stop_200\saved\trues.npy'

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

salt_1_model_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)
np.mean(salt_1_model_rmse)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内
# 假设您已经有了预测和真实的海表温度数据
# 这里我们创建一些随机数据作为示例
preds_value = preds_out[:, :, :, 200][0, :, 27:144]
trues_value = trues_out[:, :, :, 200][0, :, 27:144]
differs_value = trues_value - preds_value

preds_value_salt = preds_out_salt[:, :, :, 200][0, :, 27:144]
trues_value_salt = trues_out_salt[:, :, :, 200][0, :, 27:144]
differs_value_salt = trues_value_salt - preds_value_salt

# 创建经纬度网格
lat_range = np.where(~np.isnan(preds_out[:, :, :, 200][0, 0]))[0] - 90
dep_range = np.array([0., 5., 10., 20., 30., 50., 75., 100., 125.,
                      150., 200., 250., 300., 400., 500., 600., 700., 800.,
                      900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.])
lat2d, dep2d = np.meshgrid(lat_range, dep_range)

# lat=np.arange(-63,54,10)
lat = np.array([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
dep = np.arange(0, 2001, 100)

# 创建一个新的图形和子图数组
fig, axs = plt.subplots(2, 3, figsize=(24, 10))

# 绘制预测的海表温度数据
axs[0, 0].set_title('Argo', fontsize=15, color='k', weight='bold')
mesh1 = axs[0, 0].pcolormesh(lat2d, dep2d, trues_value, cmap='rainbow')
# axs[0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)
axs[0, 0].set_xticks(lat)  # 添加经纬度
axs[0, 0].set_xticklabels(lat)
axs[0, 0].invert_yaxis()
axs[0, 0].set_yticks(dep)
axs[0, 0].set_ylabel('Depth(m)')
axs[0, 0].xaxis.set_major_formatter(LatitudeFormatter())
axs[0, 0].text(-70, -100, '(a)  Temperature', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
axs[0, 1].set_title('S-TSRM', fontsize=15, color='k', weight='bold')
mesh2 = axs[0, 1].pcolormesh(lat2d, dep2d, preds_value, cmap='rainbow')
# axs[1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 1].set_xticks(lat)  # 添加经纬度
axs[0, 1].set_xticklabels(lat)
axs[0, 1].invert_yaxis()
axs[0, 1].set_yticks(dep)
axs[0, 1].set_ylabel('Depth(m)')
# axs[1].set_ylabel('Depth/m')
axs[0, 1].xaxis.set_major_formatter(LatitudeFormatter())

axs[0, 2].set_title('Argo - S-TSRM', fontsize=15, color='k', weight='bold')
mesh3 = axs[0, 2].pcolormesh(lat2d, dep2d, differs_value, cmap='rainbow')
# axs[1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 2].set_xticks(lat)  # 添加经纬度
axs[0, 2].set_xticklabels(lat)
axs[0, 2].invert_yaxis()
axs[0, 2].set_ylabel('Depth(m)')
axs[0, 2].set_yticks(dep)
# axs[1].set_ylabel('Depth/m')
axs[0, 2].xaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[0, 0], extend='both', orientation='vertical')
cbar1.set_label('(℃)')  # 设置颜色条标签
contour1 = axs[0, 0].contour(lat2d, dep2d, trues_value, levels=15, colors='black', linewidths=1)
plt.clabel(contour1, inline=True, fontsize=9)
# plt.colorbar(contour,ax=axs[0])

cbar2 = plt.colorbar(mesh2, ax=axs[0, 1], extend='both', orientation='vertical')
cbar2.set_label('(℃)')  # 设置颜色条标签
contour2 = axs[0, 1].contour(lat2d, dep2d, preds_value, levels=15, colors='black', linewidths=1)
plt.clabel(contour2, inline=True, fontsize=9)

cbar3 = plt.colorbar(mesh3, ax=axs[0, 2], extend='both', orientation='vertical')
cbar3.set_label('(℃)')  # 设置颜色条标签
contour3 = axs[0, 2].contour(lat2d, dep2d, differs_value, colors='black', linewidths=1)
plt.clabel(contour3, inline=True, fontsize=9)

# axs[0,0].set_title('True',fontsize=15, color='k',weight='bold')
mesh1 = axs[1, 0].pcolormesh(lat2d, dep2d, trues_value_salt, cmap='coolwarm')
# axs[0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)
axs[1, 0].set_xticks(lat)  # 添加经纬度
axs[1, 0].set_xticklabels(lat)
axs[1, 0].invert_yaxis()
axs[1, 0].set_yticks(dep)
axs[1, 0].set_ylabel('Depth(m)')
axs[1, 0].xaxis.set_major_formatter(LatitudeFormatter())
axs[1, 0].text(-70, -100, '(b)  Salinity', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[1, 1].pcolormesh(lat2d, dep2d, preds_value_salt, cmap='coolwarm')
# axs[1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 1].set_xticks(lat)  # 添加经纬度
axs[1, 1].set_xticklabels(lat)
axs[1, 1].invert_yaxis()
axs[1, 1].set_yticks(dep)
axs[1, 1].set_ylabel('Depth(m)')
# axs[1].set_ylabel('Depth/m')
axs[1, 1].xaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[1, 2].pcolormesh(lat2d, dep2d, differs_value_salt, cmap='coolwarm')
# axs[1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 2].set_xticks(lat)  # 添加经纬度
axs[1, 2].set_xticklabels(lat)
axs[1, 2].invert_yaxis()
axs[1, 2].set_ylabel('Depth(m)')
axs[1, 2].set_yticks(dep)
# axs[1].set_ylabel('Depth/m')
axs[1, 2].xaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[1, 0], extend='both', orientation='vertical')
cbar1.set_label('(psu)')  # 设置颜色条标签
contour1 = axs[1, 0].contour(lat2d, dep2d, trues_value_salt, levels=15, colors='black', linewidths=1)
plt.clabel(contour1, inline=True, fontsize=9)
# plt.colorbar(contour,ax=axs[0])

cbar2 = plt.colorbar(mesh2, ax=axs[1, 1], extend='both', orientation='vertical')
cbar2.set_label('(psu)')  # 设置颜色条标签
contour2 = axs[1, 1].contour(lat2d, dep2d, preds_value_salt, levels=15, colors='black', linewidths=1)
plt.clabel(contour2, inline=True, fontsize=9)

cbar3 = plt.colorbar(mesh3, ax=axs[1, 2], extend='both', orientation='vertical')
cbar3.set_label('(psu)')  # 设置颜色条标签
contour3 = axs[1, 2].contour(lat2d, dep2d, differs_value_salt, colors='black', linewidths=1)
plt.clabel(contour3, inline=True, fontsize=9)
# 调整子图间的间距
# fig.text(0.5, 0.98, 'Temperature at 200°W in Jan in 2019', ha='center', fontsize=12)
plt.tight_layout()
# plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\temp_salt_vertical_v3.png',dpi=600,bbox_inches='tight')
plt.show()