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

depth = np.array([0., 5., 10., 20., 30., 50., 75., 100., 125.,
           150., 200., 250., 300., 400., 500., 600., 700., 800.,
           900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.])

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内
# 假设您已经有了预测和真实的海表温度数据
# 这里我们创建一些随机数据作为示例
predicted_data_50 = preds_out[0, np.where(depth == 50)[0][0], :, :]
actual_data_50 = trues_out[0, np.where(depth == 50)[0][0], :, :]
diff_data_50 = differs_out[0, np.where(depth == 50)[0][0], :, :]

predicted_data_100 = preds_out[0, np.where(depth == 100)[0][0], :, :]
actual_data_100 = trues_out[0, np.where(depth == 100)[0][0], :, :]
diff_data_100 = differs_out[0, np.where(depth == 100)[0][0], :, :]

predicted_data_300 = preds_out[0, np.where(depth == 300)[0][0], :, :]
actual_data_300 = trues_out[0, np.where(depth == 300)[0][0], :, :]
diff_data_300 = differs_out[0, np.where(depth == 300)[0][0], :, :]

predicted_data_600 = preds_out[0, np.where(depth == 600)[0][0], :, :]
actual_data_600 = trues_out[0, np.where(depth == 600)[0][0], :, :]
diff_data_600 = differs_out[0, np.where(depth == 600)[0][0], :, :]

predicted_data_1000 = preds_out[0, np.where(depth == 1000)[0][0], :, :]
actual_data_1000 = trues_out[0, np.where(depth == 1000)[0][0], :, :]
diff_data_1000 = differs_out[0, np.where(depth == 1000)[0][0], :, :]

# 创建经纬度网格
lon_range = np.arange(-179.5, 180, 1)
lat_range = np.arange(-89.5, 90, 1)
lon2d, lat2d = np.meshgrid(lon_range, lat_range)

lon = np.arange(-179.5, 180, 19.5)
lat = np.arange(-89.5, 90, 19.5)

lon = np.array([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
lat = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])
# 创建一个新的图形和子图数组
fig, axs = plt.subplots(5, 3, figsize=(20, 18), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# 绘制预测的海表温度数据
axs[0, 0].set_title('Argo', fontsize=15, color='k', weight='bold')
mesh1 = axs[0, 0].pcolormesh(lon2d, lat2d, actual_data_50, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[0, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[0, 0].set_xticklabels(lon, fontsize=10)
axs[0, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[0, 0].set_yticklabels(lat, fontsize=10)
axs[0, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[0, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[0, 0].text(-210, 100, '(a)  50m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
axs[0, 1].set_title('S-TSRM', fontsize=15, color='k', weight='bold')
mesh2 = axs[0, 1].pcolormesh(lon2d, lat2d, predicted_data_50, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[0, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[0, 1].set_xticklabels(lon, fontsize=10)
axs[0, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[0, 1].set_yticklabels(lat, fontsize=10)
axs[0, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[0, 1].yaxis.set_major_formatter(LatitudeFormatter())

axs[0, 2].set_title('Argo - S-TSRM', fontsize=15, color='k', weight='bold')
mesh3 = axs[0, 2].pcolormesh(lon2d, lat2d, diff_data_50, cmap='jet', transform=ccrs.PlateCarree(central_longitude=180))
axs[0, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[0, 2].set_xticklabels(lon, fontsize=10)
axs[0, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[0, 2].set_yticklabels(lat, fontsize=10)
axs[0, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[0, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[0, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(℃)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[0, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(℃)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[0, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(℃)')  # 设置颜色条标签
# 调整子图间的间距
# fig.text(0.5, 0.63, 'Temperature at 100m in Jan in 2019', ha='center', fontsize=12)

# axs[0,0].set_title('True',fontsize=15, color='k',weight='bold')
mesh1 = axs[1, 0].pcolormesh(lon2d, lat2d, actual_data_100, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[1, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[1, 0].set_xticklabels(lon, fontsize=10)
axs[1, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[1, 0].set_yticklabels(lat, fontsize=10)
axs[1, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[1, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[1, 0].text(-210, 100, '(b)  100m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[1, 1].pcolormesh(lon2d, lat2d, predicted_data_100, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[1, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[1, 1].set_xticklabels(lon, fontsize=10)
axs[1, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[1, 1].set_yticklabels(lat, fontsize=10)
axs[1, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[1, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[1, 2].pcolormesh(lon2d, lat2d, diff_data_100, cmap='jet', transform=ccrs.PlateCarree(central_longitude=180))
axs[1, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[1, 2].set_xticklabels(lon, fontsize=10)
axs[1, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[1, 2].set_yticklabels(lat, fontsize=10)
axs[1, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[1, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[1, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(℃)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[1, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(℃)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[1, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(℃)')  # 设置颜色条标签

mesh1 = axs[2, 0].pcolormesh(lon2d, lat2d, actual_data_300, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[2, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[2, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[2, 0].set_xticklabels(lon, fontsize=10)
axs[2, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[2, 0].set_yticklabels(lat, fontsize=10)
axs[2, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[2, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[2, 0].text(-210, 100, '(c)  300m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[2, 1].pcolormesh(lon2d, lat2d, predicted_data_300, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[2, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[2, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[2, 1].set_xticklabels(lon, fontsize=10)
axs[2, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[2, 1].set_yticklabels(lat, fontsize=10)
axs[2, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[2, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[2, 2].pcolormesh(lon2d, lat2d, diff_data_300, cmap='jet', transform=ccrs.PlateCarree(central_longitude=180))
axs[2, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[2, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[2, 2].set_xticklabels(lon, fontsize=10)
axs[2, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[2, 2].set_yticklabels(lat, fontsize=10)
axs[2, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[2, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[2, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(℃)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[2, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(℃)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[2, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(℃)')  # 设置颜色条标签

mesh1 = axs[3, 0].pcolormesh(lon2d, lat2d, actual_data_600, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[3, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[3, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[3, 0].set_xticklabels(lon, fontsize=10)
axs[3, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[3, 0].set_yticklabels(lat, fontsize=10)
axs[3, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[3, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[3, 0].text(-210, 100, '(d)  600m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[3, 1].pcolormesh(lon2d, lat2d, predicted_data_600, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[3, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[3, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[3, 1].set_xticklabels(lon, fontsize=10)
axs[3, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[3, 1].set_yticklabels(lat, fontsize=10)
axs[3, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[3, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[3, 2].pcolormesh(lon2d, lat2d, diff_data_600, cmap='jet', transform=ccrs.PlateCarree(central_longitude=180))
axs[3, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[3, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[3, 2].set_xticklabels(lon, fontsize=10)
axs[3, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[3, 2].set_yticklabels(lat, fontsize=10)
axs[3, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[3, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[3, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(℃)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[3, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(℃)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[3, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(℃)')  # 设置颜色条标签

mesh1 = axs[4, 0].pcolormesh(lon2d, lat2d, actual_data_1000, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[4, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[4, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[4, 0].set_xticklabels(lon, fontsize=10)
axs[4, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[4, 0].set_yticklabels(lat, fontsize=10)
axs[4, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[4, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[4, 0].text(-210, 100, '(e)  1000m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[4, 1].pcolormesh(lon2d, lat2d, predicted_data_1000, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[4, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[4, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[4, 1].set_xticklabels(lon, fontsize=10)
axs[4, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[4, 1].set_yticklabels(lat, fontsize=10)
axs[4, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[4, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[4, 2].pcolormesh(lon2d, lat2d, diff_data_1000, cmap='jet',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[4, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[4, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[4, 2].set_xticklabels(lon, fontsize=10)
axs[4, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[4, 2].set_yticklabels(lat, fontsize=10)
axs[4, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[4, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[4, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(℃)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[4, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(℃)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[4, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(℃)')  # 设置颜色条标签
plt.suptitle('Temperature', fontsize=20)
plt.tight_layout()
# plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\temp_differ_depth_v3.png',dpi=600,bbox_inches='tight')
plt.show()


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

salt_model_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)
np.mean(salt_model_rmse)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度线方向设置向内
# 假设您已经有了预测和真实的海表温度数据
# 这里我们创建一些随机数据作为示例
predicted_data_50 = preds_out[0, np.where(depth == 50)[0][0], :, :]
actual_data_50 = trues_out[0, np.where(depth == 50)[0][0], :, :]
diff_data_50 = differs_out[0, np.where(depth == 50)[0][0], :, :]

predicted_data_100 = preds_out[0, np.where(depth == 100)[0][0], :, :]
actual_data_100 = trues_out[0, np.where(depth == 100)[0][0], :, :]
diff_data_100 = differs_out[0, np.where(depth == 100)[0][0], :, :]

predicted_data_300 = preds_out[0, np.where(depth == 300)[0][0], :, :]
actual_data_300 = trues_out[0, np.where(depth == 300)[0][0], :, :]
diff_data_300 = differs_out[0, np.where(depth == 300)[0][0], :, :]

predicted_data_600 = preds_out[0, np.where(depth == 600)[0][0], :, :]
actual_data_600 = trues_out[0, np.where(depth == 600)[0][0], :, :]
diff_data_600 = differs_out[0, np.where(depth == 600)[0][0], :, :]

predicted_data_1000 = preds_out[0, np.where(depth == 1000)[0][0], :, :]
actual_data_1000 = trues_out[0, np.where(depth == 1000)[0][0], :, :]
diff_data_1000 = differs_out[0, np.where(depth == 1000)[0][0], :, :]

# 创建经纬度网格
lon_range = np.arange(-179.5, 180, 1)
lat_range = np.arange(-89.5, 90, 1)
lon2d, lat2d = np.meshgrid(lon_range, lat_range)

lon = np.arange(-179.5, 180, 19.5)
lat = np.arange(-89.5, 90, 19.5)

lon = np.array([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
lat = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])
# 创建一个新的图形和子图数组
fig, axs = plt.subplots(5, 3, figsize=(20, 18), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# 绘制预测的海表温度数据
axs[0, 0].set_title('Argo', fontsize=15, color='k', weight='bold')
mesh1 = axs[0, 0].pcolormesh(lon2d, lat2d, actual_data_50, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[0, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[0, 0].set_xticklabels(lon, fontsize=10)
axs[0, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[0, 0].set_yticklabels(lat, fontsize=10)
axs[0, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[0, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[0, 0].text(-210, 100, '(a)  50m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
axs[0, 1].set_title('S-TSRM', fontsize=15, color='k', weight='bold')
mesh2 = axs[0, 1].pcolormesh(lon2d, lat2d, predicted_data_50, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[0, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[0, 1].set_xticklabels(lon, fontsize=10)
axs[0, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[0, 1].set_yticklabels(lat, fontsize=10)
axs[0, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[0, 1].yaxis.set_major_formatter(LatitudeFormatter())

axs[0, 2].set_title('Argo - S-TSRM', fontsize=15, color='k', weight='bold')
mesh3 = axs[0, 2].pcolormesh(lon2d, lat2d, diff_data_50, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[0, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[0, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[0, 2].set_xticklabels(lon, fontsize=10)
axs[0, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[0, 2].set_yticklabels(lat, fontsize=10)
axs[0, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[0, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[0, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(psu)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[0, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(psu)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[0, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(psu)')  # 设置颜色条标签
# 调整子图间的间距
# fig.text(0.5, 0.63, 'Temperature at 100m in Jan in 2019', ha='center', fontsize=12)

# axs[0,0].set_title('True',fontsize=15, color='k',weight='bold')
mesh1 = axs[1, 0].pcolormesh(lon2d, lat2d, actual_data_100, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[1, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[1, 0].set_xticklabels(lon, fontsize=10)
axs[1, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[1, 0].set_yticklabels(lat, fontsize=10)
axs[1, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[1, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[1, 0].text(-210, 100, '(b)  100m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[1, 1].pcolormesh(lon2d, lat2d, predicted_data_100, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[1, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[1, 1].set_xticklabels(lon, fontsize=10)
axs[1, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[1, 1].set_yticklabels(lat, fontsize=10)
axs[1, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[1, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[1, 2].pcolormesh(lon2d, lat2d, diff_data_100, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[1, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[1, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[1, 2].set_xticklabels(lon, fontsize=10)
axs[1, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[1, 2].set_yticklabels(lat, fontsize=10)
axs[1, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[1, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[1, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(psu)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[1, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(psu)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[1, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(psu)')  # 设置颜色条标签

mesh1 = axs[2, 0].pcolormesh(lon2d, lat2d, actual_data_300, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[2, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[2, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[2, 0].set_xticklabels(lon, fontsize=10)
axs[2, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[2, 0].set_yticklabels(lat, fontsize=10)
axs[2, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[2, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[2, 0].text(-210, 100, '(c)  300m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[2, 1].pcolormesh(lon2d, lat2d, predicted_data_300, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[2, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[2, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[2, 1].set_xticklabels(lon, fontsize=10)
axs[2, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[2, 1].set_yticklabels(lat, fontsize=10)
axs[2, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[2, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[2, 2].pcolormesh(lon2d, lat2d, diff_data_300, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[2, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[2, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[2, 2].set_xticklabels(lon, fontsize=10)
axs[2, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[2, 2].set_yticklabels(lat, fontsize=10)
axs[2, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[2, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[2, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(psu)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[2, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(psu)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[2, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(psu)')  # 设置颜色条标签

mesh1 = axs[3, 0].pcolormesh(lon2d, lat2d, actual_data_600, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[3, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[3, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[3, 0].set_xticklabels(lon, fontsize=10)
axs[3, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[3, 0].set_yticklabels(lat, fontsize=10)
axs[3, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[3, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[3, 0].text(-210, 100, '(d)  600m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[3, 1].pcolormesh(lon2d, lat2d, predicted_data_600, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[3, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[3, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[3, 1].set_xticklabels(lon, fontsize=10)
axs[3, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[3, 1].set_yticklabels(lat, fontsize=10)
axs[3, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[3, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[3, 2].pcolormesh(lon2d, lat2d, diff_data_600, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[3, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[3, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[3, 2].set_xticklabels(lon, fontsize=10)
axs[3, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[3, 2].set_yticklabels(lat, fontsize=10)
axs[3, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[3, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[3, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(psu)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[3, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(psu)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[3, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(psu)')  # 设置颜色条标签

mesh1 = axs[4, 0].pcolormesh(lon2d, lat2d, actual_data_1000, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[4, 0].coastlines()
# axs[0].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[4, 0].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[4, 0].set_xticklabels(lon, fontsize=10)
axs[4, 0].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[4, 0].set_yticklabels(lat, fontsize=10)
axs[4, 0].xaxis.set_major_formatter(LongitudeFormatter())
axs[4, 0].yaxis.set_major_formatter(LatitudeFormatter())
axs[4, 0].text(-210, 100, '(e)  1000m', fontsize=20, color='k', weight='bold')

# 绘制真实的海表温度数据
# axs[0,1].set_title('Pred',fontsize=15, color='k',weight='bold')
mesh2 = axs[4, 1].pcolormesh(lon2d, lat2d, predicted_data_1000, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[4, 1].coastlines()
# axs[1].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[4, 1].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[4, 1].set_xticklabels(lon, fontsize=10)
axs[4, 1].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[4, 1].set_yticklabels(lat, fontsize=10)
axs[4, 1].xaxis.set_major_formatter(LongitudeFormatter())
axs[4, 1].yaxis.set_major_formatter(LatitudeFormatter())

# axs[0,2].set_title('Diff',fontsize=15, color='k',weight='bold')
mesh3 = axs[4, 2].pcolormesh(lon2d, lat2d, diff_data_1000, cmap='coolwarm',
                             transform=ccrs.PlateCarree(central_longitude=180))
axs[4, 2].coastlines()
# axs[2].gridlines(crs=ccrs.PlateCarree(central_longitude=180),draw_labels=False,xlocs=lon,ylocs=lat,linewidth=0.25, linestyle='--', color='k', alpha=0.8)

axs[4, 2].set_xticks(lon, crs=ccrs.PlateCarree(central_longitude=180))  # 添加经纬度
axs[4, 2].set_xticklabels(lon, fontsize=10)
axs[4, 2].set_yticks(lat, crs=ccrs.PlateCarree(central_longitude=180))
axs[4, 2].set_yticklabels(lat, fontsize=10)
axs[4, 2].xaxis.set_major_formatter(LongitudeFormatter())
axs[4, 2].yaxis.set_major_formatter(LatitudeFormatter())

cbar1 = plt.colorbar(mesh1, ax=axs[4, 0], extend='both', orientation='vertical', shrink=0.7)
cbar1.set_label('(psu)')  # 设置颜色条标签

cbar2 = plt.colorbar(mesh2, ax=axs[4, 1], extend='both', orientation='vertical', shrink=0.7)
cbar2.set_label('(psu)')  # 设置颜色条标签

cbar3 = plt.colorbar(mesh3, ax=axs[4, 2], extend='both', orientation='vertical', shrink=0.7)
cbar3.set_label('(psu)')  # 设置颜色条标签
plt.suptitle('Salinity', fontsize=20)
plt.tight_layout()
plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\salt_differ_depth_v3.png', dpi=600, bbox_inches='tight')
# fig.suptitle("Global Title for All Subplots", fontsize=16)
# 显示图形
plt.show()
