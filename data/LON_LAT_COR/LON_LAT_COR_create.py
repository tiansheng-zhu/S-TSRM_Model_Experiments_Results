import numpy as np
import xarray as xr


path=r'F:\IPRC_argo_2005-2020\argo_2005-2020_grd.nc'
data=xr.open_dataset(path,decode_times=False)

lon=data.LONGITUDE
lon
lat=data.LATITUDE
lat

time=data.TIME
time

d2r=np.pi/180
def latlon2xyz(lat, lon):
    x = -np.cos(lat)*np.cos(lon)
    y = -np.cos(lat)*np.sin(lon)
    z = np.sin(lat)
    return x, y, z

lon_lat_cor_arr=[]
_lon,_lat=np.meshgrid((lon-180)*d2r,lat*d2r)
print(_lon)
print(_lat)
for i in range(len(time)):
    x,y,z=latlon2xyz(_lon,_lat)
    print(x.shape,y.shape,z.shape)
    cor_plus=x+y+z
    lon_lat_cor_arr.append(cor_plus)
    _lon=_lon+2*np.pi/12
    _lat=_lat+2*np.pi/12
    break

lon_lat_cor=np.array(lon_lat_cor_arr)
lon_lat_cor.shape

map_to_temp=np.load(r'G:\map_data_for_IPRC_ARGO\map_temp_0.npy')

map_to_temp_no_nan=np.nan_to_num(map_to_temp,np.nan,0)
map_to_temp_no_nan

map_to_temp_cycle=map_to_temp[:12,0,:,:]
map_to_temp_cycle.shape

map_lon_lat_cor=[]
for i in range(lon_lat_cor.shape[0]):
    lon_lat_cor_temp=lon_lat_cor[i]*map_to_temp_cycle[i%12]
    map_lon_lat_cor.append(lon_lat_cor_temp)

map_lon_lat_cor=np.array(map_lon_lat_cor)
map_lon_lat_cor.shape

var_data=map_lon_lat_cor
var_data_min = np.nanmin(var_data, axis=(1, 2))[:, np.newaxis, np.newaxis]
var_data_max = np.nanmax(var_data, axis=(1, 2))[:, np.newaxis, np.newaxis]
var_data_diff = var_data_max - var_data_min
limited_var = (var_data - var_data_min) / var_data_diff
pro_map_lon_lat_cor = np.nan_to_num(limited_var, nan=0.0)

np.save(r'pro_map_lon_lat_cor.npy',pro_map_lon_lat_cor)



