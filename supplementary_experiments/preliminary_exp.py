"""
Temp
"""
"""
SwinLSTM
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
temp_swinLSTM_r2=np.array(temp_r2)

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

temp_swinLSTM_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)


"""
Attention U-net
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
temp_u_net_r2=np.array(temp_r2)

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

temp_u_net_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)


"""
Salt
"""

"""
SwinLSTM
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
salt_swinLSTM_r2=np.array(temp_r2)

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

salt_swinLSTM_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)


"""
Attention U-net
"""
preds_path=r'F:\label_data\model_mlp_gsta_u-net_swintrans_low_parameter\thermohaline_incepu_salt\saved\preds.npy'
trues_path=r'F:\label_data\model_mlp_gsta_u-net_swintrans_low_parameter\thermohaline_incepu_salt\saved\trues.npy'

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
salt_u_net_r2=np.array(temp_r2)

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

salt_u_net_rmse=np.array(temp_rmse)
# salt_rmse=np.array(salt_rmse)

temp_rmse_swinLSTM_sel_avg=np.mean(temp_swinLSTM_rmse,axis=0)
temp_rmse_u_net_sel_avg=np.mean(temp_u_net_rmse,axis=0)
salt_rmse_swinLSTM_sel_avg=np.mean(salt_swinLSTM_rmse,axis=0)
salt_rmse_u_net_sel_avg=np.mean(salt_u_net_rmse,axis=0)
from matplotlib.ticker import MultipleLocator
plt.rcParams['font.sans-serif']=['Times New Roman']
depth=np.array([   0.,    5.,   10.,   20.,   30.,   50.,   75.,  100.,  125.,
        150.,  200.,  250.,  300.,  400.,  500.,  600.,  700.,  800.,
        900., 1000., 1100., 1200., 1300., 1400., 1500., 1750., 2000.])
# salt_rmse_avg=np.mean(salt_rmse,axis=0)
fig, axs = plt.subplots(1,1,figsize=(15,4))

axs.plot(depth, temp_rmse_swinLSTM_sel_avg,marker='o', linestyle='-', markersize=5, label='SwinLSTM-temp',lw=1)
axs.plot(depth, temp_rmse_u_net_sel_avg, marker='o', linestyle='-', markersize=5, label='Attention U-net-temp',lw=1)

# ax.plot(depth, temp_rmse_u_net_sel_avg, marker='o', linestyle=':', markersize=3.5, label='U-net')
# ax.plot(temp_total_rmse_avg, depth, marker='o', linestyle=':', markersize=3.5, label='t-temp-avg', color='green')
# ax.plot(salt_rmse_avg, depth, marker='o', linestyle=':', markersize=3.5, label='salt-avg')
# ax.plot(salt_total_rmse_avg, depth, marker='o', linestyle=':', markersize=3.5, label='t-salt-avg', color='black')
axs.set_xlim(0, max(depth) * 1.01)
axs.set_ylim(0,max(temp_rmse_swinLSTM_sel_avg.max(),temp_rmse_u_net_sel_avg.max()) + 0.1)
axs.xaxis.set_major_locator(MultipleLocator(50))
axs.tick_params(axis='y', labelsize=10)
axs.tick_params(axis='x', labelsize=10)
axs.set_xlabel('Depth(m)',fontsize=14)
axs.set_ylabel('RMSE',fontsize=14)
# ax.invert_yaxis()

# ax1=axs.twinx()
axs.plot(depth, salt_rmse_swinLSTM_sel_avg,marker='s', linestyle='-', markersize=5, label='SwinLSTM-salt',lw=1)
axs.plot(depth, salt_rmse_u_net_sel_avg, marker='s', linestyle='-', markersize=5, label='Attention U-net-salt',lw=1)
# ax.plot(tempotal_rmse_avg, depth, marker='o', linestyle=':', markersize=3.5, label='t-temp-avg', color='green')
# ax.plot(salt_rmse_avg, depth, marker='o', linestyle=':', markersize=3.5, label='salt-avg')
# ax.plot(salt_total_rmse_avg, depth, marker='o', linestyle=':', markersize=3.5, label='t-salt-avg', color='black')
# ax.set_xlim(-5, max(depth1)+5)
# axs.set_ylim(0.8,1.01)
# # ax.xaxis.set_major_locator(MultipleLocator(25))
# axs.tick_params(axis='y', labelsize=12)
# axs.tick_params(axis='x', labelsize=12)
# axs.set_xlabel('Depth/m',fontsize=14)
# # ax.set_xticklabels(depth1)
# axs.set_ylabel('R²',fontsize=14)

axs.legend(loc='upper right',fontsize=12)
# axs.set_title('AVG_RMSE', fontsize=16)
# plt.savefig(r'E:\3D_thermohaline\high_resolution_exp_picture\temp_salt_swinLSTM_u_net_v2.png',dpi=600,bbox_inches='tight')
plt.tight_layout()