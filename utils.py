import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

lat_min = 0
lat_max = 40
lon_min = 60
lon_max = 100
lon_size = int((lon_max - lon_min) * 10)
lat_size = int((lat_max - lat_min) * 10)

def local_stats_manual(array, window_size=5):
    lat_size, lon_size = array.shape
    offset = window_size // 2
    
    mean_array = np.zeros_like(array)
    std_array = np.zeros_like(array)
    
    for i in range(lat_size):
        for j in range(lon_size):
            i_start, i_end = max(0, i-offset), min(lat_size, i+offset+1)
            j_start, j_end = max(0, j-offset), min(lon_size, j+offset+1)
            
            window = array[i_start:i_end, j_start:j_end]
            local_mean = np.nanmean(window)
            local_std = np.nanstd(window)
            
            mean_array[i, j] = local_mean
            std_array[i, j] = local_std
    
    return mean_array, std_array

def extract_rainfall(h5_file_path, lat_index_range=(2400, 2800), lon_index_range=(900, 1300)):
    with h5py.File(h5_file_path, 'r') as f:
        hourlyPrecipRate = f['Grid/hourlyPrecipRate'][lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
        lats = f['Grid/Latitude'][lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]
        lons = f['Grid/Longitude'][lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]]

    return hourlyPrecipRate, lats, lons

def read_insat_file(filename, PAR):
    data_grid=None
    try:
        h5f = h5py.File(filename, 'r')
        if PAR not in h5f:
            print(f"Error: Parameter '{PAR}' not found in file {filename}")
            return None
        lat = h5f['Latitude'][()] * 0.01
        lon = h5f['Longitude'][()] * 0.01
        data = np.squeeze(h5f[PAR][()])

        lat_filtered = lat[(lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)]
        lon_filtered = lon[(lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)]
        data_filtered = data[(lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)]

        data_grid = np.zeros((lat_size, lon_size))
        cnt = np.zeros((lat_size, lon_size))

        lat_ind = np.int32((lat_filtered - lat_min) * 10)
        lon_ind = np.int32((lon_filtered - lon_min) * 10)

        np.add.at(data_grid, (lat_ind, lon_ind), data_filtered)
        np.add.at(cnt, (lat_ind, lon_ind), 1)

        data_grid[cnt > 0] = data_grid[cnt > 0] / cnt[cnt > 0]
        data_grid[cnt == 0] = np.nan
    except OSError as e:
        print(f"Error reading file {filename}, possibly bad file: {e}")
    return data_grid

def combine_data(rainfall_data, rainfall_lats, rainfall_lons, olr_data, output_csv):
    pixel_data = []

    mean_rainfall, std_rainfall = local_stats_manual(rainfall_data, window_size=5)
    mean_olr, std_olr = local_stats_manual(olr_data, window_size=5)

    for i in range(rainfall_data.shape[0]):
        for j in range(rainfall_data.shape[1]):
            pixel_data.append({
                'Latitude': rainfall_lats[i, j],
                'Longitude': rainfall_lons[i, j],
                'Rainfall': rainfall_data[i, j],
                'OLR': olr_data[i, j],
                'Rainfall_5x5_mean': mean_rainfall[i, j],
                'Rainfall_5x5_std': std_rainfall[i, j],
                'OLR_5x5_mean': mean_olr[i, j],
                'OLR_5x5_std': std_olr[i, j],
            })

    df = pd.DataFrame(pixel_data)
    df.to_csv(output_csv, index=False, float_format='%.2f')

    print(f"Data saved to {output_csv}")
    print(f"Total pixels: {len(pixel_data)}")

def get_elev_map():

    elev_file = "GMTED2010_15n015_00625deg.nc"
    h5f2 = h5py.File(elev_file,'r')
    elev = np.squeeze(h5f2['elevation'][()])
    lat_elev = h5f2['latitude'][()]
    lon_elev = h5f2['longitude'][()]
   
    f_elev = RegularGridInterpolator((lat_elev, lon_elev), elev, method='linear', bounds_error=False, fill_value=None)

    lat1 = np.arange(0.05, 40.05, 0.1)
    lon1 = np.arange(60.05, 100.05, 0.1)
    
    lon1_mesh, lat1_mesh = np.meshgrid(lon1, lat1)
    points = np.column_stack((lat1_mesh.ravel(), lon1_mesh.ravel()))
    elev1 = f_elev(points).reshape(lat1_mesh.shape)

    return elev1


def derive_elev_correction(elev_10km,olr_grid,cloud_mask):
    x1 = []
    y1 = []

    for el in [0,2000,4000,5000]:
        x1.append(np.mean(elev_10km[(elev_10km>=el)&(elev_10km<el+1000)]))
        y1.append(np.mean(olr_grid[(elev_10km>=el)&(elev_10km<el+1000)]))
   
    y1[1:]=y1[1:]-y1[0]        
    y1[0]=0
    corr_p = np.polyfit(x1,y1,2)
    corrs = corr_p[0]*(elev_10km**2) + corr_p[1]*elev_10km+corr_p[2]
    olr_corr = np.copy(olr_grid)
    olr_corr[elev_10km>=1000] = olr_grid[elev_10km>=1000]-corrs[elev_10km>=1000]
    return olr_corr
