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

temp_preds_out=preds_nan*total_diff+total_min
temp_trues_out=trues_nan*total_diff+total_min

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

salt_preds_out=preds_nan*total_diff+total_min
salt_trues_out=trues_nan*total_diff+total_min


temp_preds_out_r=temp_preds_out.transpose(0,2,3,1)
temp_trues_out_r=temp_trues_out.transpose(0,2,3,1)
salt_preds_out_r=salt_preds_out.transpose(0,2,3,1)
salt_trues_out_r=salt_trues_out.transpose(0,2,3,1)

import numpy as np
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import warnings
from scipy.integrate import cumtrapz

# 忽略警告
warnings.filterwarnings('ignore')


def compute_basin_moc(ds, basin='atlantic', reference_depth=2000):
    # 地球物理常数
    R = 6371000  # 地球半径 (m)
    omega = 7.2921e-5  # 地转角速度
    g = 9.81
    rho0 = 1025  # 参考密度 (kg/m³)

    lon = ds.longitude.values
    lat = ds.latitude.values
    depth = ds.depth.values
    time = ds.time.values

    # 构造网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    f = 2 * omega * np.sin(np.deg2rad(lat_grid))  # Coriolis参数 [lat, lon]
    dlon = np.deg2rad(np.diff(lon)[0])  # 经度间隔 (弧度)
    dx = dlon * R * np.cos(np.deg2rad(lat_grid))  # 东西距离 [m]

    moc = np.zeros((len(time), len(lat), len(depth)))

    # 添加f值阈值避免赤道附近除零问题
    # f_threshold = 1e-9  # 约对应0.1°纬度
    # f_safe = np.where(np.abs(f) < f_threshold, np.sign(f)*f_threshold, f)
    f_safe = f

    for t in tqdm(range(len(time)), desc=f'计算{basin.upper()}MOC每月流函数'):
        temp = ds.temperature[t].values  # [lat, lon, depth]
        salt = ds.salinity[t].values
        dens = np.full_like(temp, np.nan)

        # 计算密度场
        for k, z in enumerate(depth):
            p = gsw.p_from_z(-z, lat_grid)
            SA = gsw.SA_from_SP(salt[:, :, k], p, lon_grid, lat_grid)
            CT = gsw.CT_from_t(SA, temp[:, :, k], p)
            dens[:, :, k] = gsw.rho(SA, CT, p)

        # 计算东西密度梯度 ∂ρ/∂x [kg/m⁴]
        drho_dx = np.gradient(dens, axis=1) / dx[:, :, None]  # axis1=经度方向

        # 计算地转速度垂直梯度 ∂v/∂z [s⁻¹]
        dv_dz = (g / (rho0 * f[:, :, None])) * drho_dx

        # 从参考层向上积分
        ref_idx = np.abs(depth - reference_depth).argmin()
        v_geo = np.zeros_like(dv_dz)

        for i in range(len(lat)):
            for j in range(len(lon)):
                # 从参考层积分到表层
                if ref_idx > 0:
                    v_geo[i, j, :ref_idx + 1] = cumtrapz(
                        dv_dz[i, j, :ref_idx + 1][::-1],
                        -depth[:ref_idx + 1][::-1],
                        initial=0
                    )[::-1]

        # 定义海盆掩膜
        if basin.lower() == 'atlantic':
            # 大西洋掩膜 (经度: 280°–360° & 0°–20°, 纬度: 20°N–70°N)
            basin_mask = (
                    ((lon_grid >= 280) | (lon_grid <= 20)) &
                    (lat_grid >= 20) &
                    (lat_grid <= 70)
            )
            moc_name = 'AMOC'
        elif basin.lower() == 'pacific':
            # 太平洋掩膜 (经度: 120°–290°, 纬度: 60°S–60°N)
            basin_mask = (
                    (lon_grid >= 120) &
                    (lon_grid <= 290) &
                    (lat_grid >= -60) &
                    (lat_grid <= 60)
            )
            moc_name = 'PMOC'
        elif basin.lower() == 'indian':
            # 印度洋掩膜 (经度: 20°–120°, 纬度: 60°S–30°N)
            basin_mask = (
                    (lon_grid >= 20) &
                    (lon_grid <= 120) &
                    (lat_grid >= -60) &
                    (lat_grid <= 30)
            )
            moc_name = 'IMOC'
        elif basin.lower() == 'southern':
            # 南大洋掩膜 (纬度: 78°S–34°S)
            # 南大洋没有经度边界，环绕整个南极大陆
            basin_mask = (lat_grid <= -34) & (lat_grid >= -78)
            moc_name = 'SMOC'
        else:
            raise ValueError(f"未知海盆类型: {basin}. 支持: 'atlantic', 'pacific', 'indian'")

        # 计算经向输送 ∫v dx [m²/s]
        trans = np.nansum(np.where(basin_mask[:, :, None], v_geo * dx[:, :, None], 0), axis=1)

        # 深度积分得到流函数 [Sv]
        for i in range(len(lat)):
            # 从海底到表层积分
            moc[t, i, :] = cumtrapz(trans[i, ::-1], -depth[::-1], initial=0)[::-1] / 1e6

    return xr.DataArray(
        moc,
        coords={'time': time, 'latitude': lat, 'depth': depth},
        dims=['time', 'latitude', 'depth'],
        name=moc_name
    )


print("创建模拟数据集...")
lon = np.arange(0.5, 360, 1.0)  # 0.5, 1.5, ..., 359.5
lat = np.arange(-89.5, 90, 1.0)  # -89.5, -88.5, ..., 89.5
depth = np.array([0., 5., 10., 20., 30., 50., 75., 100., 125., 150.,
                  200., 250., 300., 400., 500., 600., 700., 800., 900., 1000.,
                  1100., 1200., 1300., 1400., 1500., 1750., 2000.])

# 创建时间序列 (2004-2019年每月)
time = np.arange('2019-01', '2020-01', dtype='datetime64[M]')

# 创建随机温盐数据 (实际应用中应从文件中加载)
np.random.seed(42)

# 创建xarray数据集
ds_preds = xr.Dataset(
    {
        'temperature': (['time', 'latitude', 'longitude', 'depth'], temp_preds_out_r),
        'salinity': (['time', 'latitude', 'longitude', 'depth'], salt_preds_out_r),
    },
    coords={
        'time': time,
        'latitude': lat,
        'longitude': lon,
        'depth': depth
    }
)

print("数据集创建完成:")
print(ds_preds)
# 计算全球经圈翻转环流 - 为了演示只计算第一个时间步
print("\n计算大西洋经圈翻转环流...")
amoc = compute_basin_moc(ds_preds, basin='atlantic')

print("\n计算太平洋经圈翻转环流...")
pmoc = compute_basin_moc(ds_preds, basin='pacific')

print("\n计算印度洋经圈翻转环流...")
imoc = compute_basin_moc(ds_preds, basin='indian')

print("\n计算南大洋经圈翻转环流...")
smoc = compute_basin_moc(ds_preds, basin='southern')



print("创建模拟数据集...")
lon = np.arange(0.5, 360, 1.0)  # 0.5, 1.5, ..., 359.5
lat = np.arange(-89.5, 90, 1.0)  # -89.5, -88.5, ..., 89.5
depth = np.array([0., 5., 10., 20., 30., 50., 75., 100., 125., 150.,
                 200., 250., 300., 400., 500., 600., 700., 800., 900., 1000.,
                 1100., 1200., 1300., 1400., 1500., 1750., 2000.])

# 创建时间序列 (2004-2019年每月)
time = np.arange('2019-01', '2020-01', dtype='datetime64[M]')

# 创建随机温盐数据 (实际应用中应从文件中加载)
np.random.seed(42)


# 创建xarray数据集
ds_trues = xr.Dataset(
    {
        'temperature': (['time', 'latitude', 'longitude', 'depth'], temp_trues_out_r),
        'salinity': (['time', 'latitude', 'longitude', 'depth'], salt_trues_out_r),
    },
    coords={
        'time': time,
        'latitude': lat,
        'longitude': lon,
        'depth': depth
    }
)

print("数据集创建完成:")
print(ds_trues)
# 计算全球经圈翻转环流 - 为了演示只计算第一个时间步
print("\n计算大西洋经圈翻转环流...")
amoc_t = compute_basin_moc(ds_trues, basin='atlantic')

print("\n计算太平洋经圈翻转环流...")
pmoc_t = compute_basin_moc(ds_trues, basin='pacific')

print("\n计算印度洋经圈翻转环流...")
imoc_t = compute_basin_moc(ds_trues, basin='indian')

print("\n计算南大洋经圈翻转环流...")
smoc_t = compute_basin_moc(ds_trues, basin='southern')

amoc_differ=amoc_t-amoc
pmoc_differ=pmoc_t-pmoc
imoc_differ=imoc_t-imoc
smoc_differ=smoc_t-smoc

R = 6371000  # 地球半径 (m)
omega = 7.2921e-5  # 地转角速度
g = 9.81
rho0 = 1025  # 参考密度 (kg/m³)

lon = ds_preds.longitude.values
lat = ds_preds.latitude.values
depth = ds_preds.depth.values
time = ds_preds.time.values

# 构造网格
lon_grid, lat_grid = np.meshgrid(lon, lat)
f = 2 * omega * np.sin(np.deg2rad(lat_grid))  # Coriolis参数 [lat, lon]

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from matplotlib import gridspec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker

# 假设已经计算得到以下数据:
# amoc_argo: 大西洋区域Argo数据的MOC [time, latitude, depth]
# amoc_recon: 大西洋区域重构数据的MOC [time, latitude, depth]
# smoc_argo: 南大洋区域Argo数据的MOC [time, latitude, depth]
# smoc_recon: 南大洋区域重构数据的MOC [time, latitude, depth]

# 选择特定年份 (这里以2019年为例)
year = 2019

# 提取特定月份 (1月, 4月, 7月, 10月)
months = [1, 4, 7, 10]
month_names = ['January', 'April', 'July', 'October']

# 准备绘图数据
# 大西洋数据
amoc_argo_months = [amoc_t.sel(time=f'{year}-{m:02d}') for m in months]
# print(amoc_argo_months)
amoc_recon_months = [amoc.sel(time=f'{year}-{m:02d}') for m in months]
amoc_diff_months = [a - r for a, r in zip(amoc_argo_months, amoc_recon_months)]

# 南大洋数据
smoc_argo_months = [smoc_t.sel(time=f'{year}-{m:02d}') for m in months]
smoc_recon_months = [smoc.sel(time=f'{year}-{m:02d}') for m in months]
smoc_diff_months = [a - r for a, r in zip(smoc_argo_months, smoc_recon_months)]

# 创建图形
fig = plt.figure(figsize=(24, 24), dpi=600)
gs = gridspec.GridSpec(6, 4, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.4, wspace=0.3)

# 设置统一的色标范围
# 大西洋MOC范围 (根据实际数据调整)
# amoc_min, amoc_max = -20, 20
# 南大洋MOC范围 (通常更大)
# smoc_min, smoc_max = -30, 30
# 差值范围 (根据实际数据调整)
# diff_min, diff_max = -5, 5

# 创建自定义色标
amoc_cmap = plt.cm.RdBu_r
smoc_cmap = plt.cm.RdBu_r
diff_cmap = plt.cm.coolwarm

# amoc_norm = mcolors.Normalize(vmin=amoc_min, vmax=amoc_max)
# smoc_norm = mcolors.Normalize(vmin=smoc_min, vmax=smoc_max)
# diff_norm = mcolors.Normalize(vmin=diff_min, vmax=diff_max)

# 绘制大西洋区域MOC (上半部分)
# 第一行: Argo数据
for i, month in enumerate(months):
    ax = plt.subplot(gs[0, i])
    data = amoc_argo_months[i][0]
    # print(data.shape)
    # 绘制填充等值线
    amoc_min, amoc_max = data.min(), data.max()
    contour = ax.contourf(data.latitude, data.depth, data.T,
                          levels=30,
                          cmap=amoc_cmap, extend='both')

    # 绘制等值线
    # cs = ax.contour(data.latitude, data.depth, data.T,
    #                levels=10,
    #                colors='k', linewidths=0.5)
    # ax.clabel(cs, fmt='%d',fontsize=12)

    # 设置坐标轴
    ax.set_ylim(0, 2000)
    dep = np.arange(0, 2001, 400)
    ax.set_yticks(dep)
    ax.invert_yaxis()
    ax.set_xlim(20, 70)
    ticks = np.arange(20, 71, 10)  # 20,30,40,50,60,70
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t)}°N' for t in ticks])
    # ax.xaxis.set_major_locator(mticker.FixedLocator([20, 30, 40, 50, 60, 70]))
    # ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{int(x)}°N'))
    # 添加标题 (仅第一行)
    if i == 0:
        ax.set_ylabel('Depth(m)', fontsize=14)
        ax.text(-0.2, 0.5, 'Argo', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='center', ha='right')

    ax.set_title(f'{month_names[i]}', fontweight='bold', fontsize=16)

    # 添加色标 (仅最后一列)
    if i == len(months) - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('AMOC(Sv)', fontsize=14)

# 第二行: 重构数据
for i, month in enumerate(months):
    ax = plt.subplot(gs[1, i])
    data = amoc_recon_months[i][0]

    contour = ax.contourf(data.latitude, data.depth, data.T,
                          levels=30,
                          cmap=amoc_cmap, extend='both')

    # cs = ax.contour(data.latitude, data.depth, data.T,
    #                levels=10,
    #                colors='k', linewidths=0.5)
    # ax.clabel(cs, fmt='%d',fontsize=12)

    ax.set_ylim(0, 2000)
    dep = np.arange(0, 2001, 400)
    ax.set_yticks(dep)
    ax.invert_yaxis()
    ax.set_xlim(20, 70)
    ticks = np.arange(20, 71, 10)  # 20,30,40,50,60,70
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t)}°N' for t in ticks])
    if i == len(months) - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('AMOC(Sv)', fontsize=14)
    if i == 0:
        ax.set_ylabel('Depth(m)', fontsize=14)
        ax.text(-0.2, 0.5, 'S-TSRM', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='right')

# 第三行: 差值
for i, month in enumerate(months):
    ax = plt.subplot(gs[2, i])
    data = amoc_diff_months[i][0]

    contour = ax.contourf(data.latitude, data.depth, data.T,
                          levels=30,
                          cmap=diff_cmap, extend='both')

    # # 绘制零值线
    # cs = ax.contour(data.latitude, data.depth, data.T,
    #                levels=5, colors='k', linewidths=0.5)
    # ax.clabel(cs, fmt='%d',fontsize=12)

    ax.set_ylim(0, 2000)
    dep = np.arange(0, 2001, 400)
    ax.set_yticks(dep)
    ax.invert_yaxis()
    ax.set_xlim(20, 70)
    # ax.set_xlabel('Latitude')
    ticks = np.arange(20, 71, 10)  # 20,30,40,50,60,70
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t)}°N' for t in ticks])
    if i == 0:
        ax.set_ylabel('Depth(m)', fontsize=14)
        ax.text(-0.16, 0.5, 'Argo - S-TSRM', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='right')

    # # 添加色标 (仅最后一列)
    if i == len(months) - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('ΔMOC(Sv)', fontsize=14)

# 绘制南大洋区域MOC (下半部分)
# 第四行: Argo数据
for i, month in enumerate(months):
    ax = plt.subplot(gs[3, i])
    data = smoc_argo_months[i][0]

    contour = ax.contourf(data.latitude, data.depth, data.T,
                          levels=30,
                          cmap=smoc_cmap, extend='both')

    # 绘制等值线
    # cs = ax.contour(data.latitude, data.depth, data.T,
    #                levels=10,
    #                colors='k', linewidths=0.5)
    # ax.clabel(cs, fmt='%d',fontsize=12)

    # 设置坐标轴
    ax.set_ylim(0, 2000)
    dep = np.arange(0, 2001, 400)
    ax.set_yticks(dep)
    ax.invert_yaxis()
    ax.set_xlim(-78, -34)
    ticks = np.arange(-70, -29, 10)  # -70, -60, -50, -40, -30 (注意-34以上，所以-30就够了)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(abs(t))}°S' for t in ticks])

    if i == 0:
        ax.set_ylabel('Depth(m)', fontsize=14)
        ax.text(-0.2, 0.5, 'Argo', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='right')

    # 添加色标 (仅最后一列)
    if i == len(months) - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('SMOC(Sv)', fontsize=14)

# 第五行: 重构数据
for i, month in enumerate(months):
    ax = plt.subplot(gs[4, i])
    data = smoc_recon_months[i][0]

    contour = ax.contourf(data.latitude, data.depth, data.T,
                          levels=30,
                          cmap=smoc_cmap, extend='both')

    # cs = ax.contour(data.latitude, data.depth, data.T,
    #                levels=10,
    #                colors='k', linewidths=0.5)
    # ax.clabel(cs, fmt='%d',fontsize=12)

    ax.set_ylim(0, 2000)
    dep = np.arange(0, 2001, 400)
    ax.set_yticks(dep)
    ax.invert_yaxis()
    ax.set_xlim(-78, -34)
    ticks = np.arange(-70, -29, 10)  # -70, -60, -50, -40, -30 (注意-34以上，所以-30就够了)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(abs(t))}°S' for t in ticks])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.1)
    # cbar = plt.colorbar(contour, cax=cax)
    # cbar.set_label('SMOC(Sv)', fontsize=14)
    if i == 0:
        ax.set_ylabel('Depth(m)', fontsize=14)
        ax.text(-0.2, 0.5, 'S-TSRM', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='right')
    if i == len(months) - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('SMOC(Sv)', fontsize=14)

# 第六行: 差值
for i, month in enumerate(months):
    ax = plt.subplot(gs[5, i])
    data = smoc_diff_months[i][0]

    contour = ax.contourf(data.latitude, data.depth, data.T,
                          levels=30,
                          cmap=diff_cmap, extend='both')

    # 绘制零值线
    # cs = ax.contour(data.latitude, data.depth, data.T,
    #                levels=5, colors='k', linewidths=0.5)
    # ax.clabel(cs, fmt='%d',fontsize=12)

    ax.set_ylim(0, 2000)
    dep = np.arange(0, 2001, 400)
    ax.set_yticks(dep)
    ax.invert_yaxis()
    ax.set_xlim(-78, -34)
    # ax.set_xlabel('Latitude/°')
    ticks = np.arange(-70, -29, 10)  # -70, -60, -50, -40, -30 (注意-34以上，所以-30就够了)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(abs(t))}°S' for t in ticks])
    if i == 0:
        ax.set_ylabel('Depth(m)', fontsize=14)
        ax.text(-0.16, 0.5, 'Argo - S-TSRM', transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='right')

    # 添加色标 (仅最后一列)
    if i == len(months) - 1:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('ΔMOC(Sv)', fontsize=14)

# 添加整体标题
# plt.suptitle(f'Atlantic and Southern Ocean MOC Comparison ({year})',
#              fontsize=20, y=0.93)

# 添加区域标签
fig.text(0.12, 0.9, '(a) Atlantic Ocean', fontsize=20, fontweight='bold')
fig.text(0.12, 0.49, '(b) Southern Ocean', fontsize=20, fontweight='bold')

# 调整布局并保存
plt.tight_layout()  # 为整体标题留出空间
# plt.savefig(f'E:/3D_thermohaline/high_resolution_exp_picture/MOC_comparison_{year}_v4.png', bbox_inches='tight', dpi=600)
plt.show()