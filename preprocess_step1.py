import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import os
from glob import glob
import re
from adaptive_binarize import *
from utils import *
from datetime import datetime, timedelta

lat_min = 0
lat_max = 40
lon_min = 60
lon_max = 100
lon_size = int((lon_max - lon_min) * 10)
lat_size = int((lat_max - lat_min) * 10)

def local_stats_convolution(array, window_size=5):
    local_mean = uniform_filter(array, size=window_size)
    squared_array = array ** 2
    local_mean_squared = uniform_filter(squared_array, size=window_size)
    local_variance = local_mean_squared - local_mean ** 2
    local_std = np.sqrt(local_variance)
    
    return local_mean, local_std

def extract_rainfall(h5_file_path, lon_index_range=(2400, 2800), lat_index_range=(900, 1300)):
    with h5py.File(h5_file_path, 'r') as f:
        hourlyPrecipRate = f['Grid/hourlyPrecipRate'][lon_index_range[0]:lon_index_range[1],lat_index_range[0]:lat_index_range[1]].T
        lats = f['Grid/Latitude'][lon_index_range[0]:lon_index_range[1],lat_index_range[0]:lat_index_range[1]].T
        lons = f['Grid/Longitude'][lon_index_range[0]:lon_index_range[1],lat_index_range[0]:lat_index_range[1]].T

    return hourlyPrecipRate, lats, lons

def combine_data(rainfall_data, rainfall_lats, rainfall_lons, olr_data, output_csv, iteration):
    pixel_data = []

    mean_olr, std_olr = local_stats_convolution(olr_data, window_size=5)

    pixel_data = {
        'Latitude': rainfall_lats.flatten(),
        'Longitude': rainfall_lons.flatten(),
        'Rainfall': rainfall_data.flatten(),
        'OLR': olr_data.flatten(),
        'OLR_5x5_mean': mean_olr.flatten(),
        'OLR_5x5_std': std_olr.flatten(),
    }

    df = pd.DataFrame(pixel_data)
    df.to_csv(output_csv, index=False, float_format='%.2f')

    print(f"Data saved to {output_csv}")
    print(f"Total pixels: {len(pixel_data)}")

def extract_timestamp(filename):
    month_abbrs = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_pattern = '|'.join(month_abbrs)

    if filename.startswith('3DIMG'):
        match = re.search(rf'(\d{{2}}({month_pattern})\d{{4}}_\d{{4}})', filename)
        if match:
            return match.group(1)

    elif filename.startswith('GPMMRG_MAP'):
        match = re.search(r'_(\d{10})_', filename)
        if match:
            timestamp = match.group(1)
            year = '20' + timestamp[0:2]
            month_num = timestamp[2:4]
            day = timestamp[4:6]
            hour = timestamp[6:10]
            month_map = {
                '01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR', '05': 'MAY',
                '06': 'JUN', '07': 'JUL', '08': 'AUG', '09': 'SEP', '10': 'OCT',
                '11': 'NOV', '12': 'DEC'
            }
            if month_num in month_map:
                return f"{day}{month_map[month_num]}{year}_{hour}"
    return None

def is_hourly_file(timestamp):
    return timestamp and timestamp.endswith('00')

start = datetime(2018, 6, 1)
# start= datetime(2019, 3, 1)
end = datetime(2019, 6, 1)
month_years = []
current = start
while current < end:
    month_years.append(current.strftime('%b%Y').capitalize()) 
    next_month = current.replace(day=28) + timedelta(days=4)
    current = next_month.replace(day=1)

rainfall_folders = [f'path_to_GSMAP_data/GSMAP_{m}' for m in month_years]
olr_folders = [f'"path_to_monthwise_OLR_data"/{m}' for m in month_years]
output_folder = "path_to_output_directory"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")

rainfall_files = []
for folder in rainfall_folders:
    rainfall_files.extend(sorted(glob(os.path.join(folder, '*.h5'))))

all_olr_files = []
for folder in olr_folders:
    all_olr_files.extend(sorted(glob(os.path.join(folder, '*.h5'))))

def is_6th_day(filename):
    ts = extract_timestamp(os.path.basename(filename))
    if ts:
        try:
            day = int(ts[:2])
            return day % 6 == 0
        except:
            return False
    return False

rainfall_files = [f for f in rainfall_files if is_6th_day(f)]
all_olr_files = [f for f in all_olr_files if is_6th_day(f)]


valid_months = [m.upper() for m in month_years]

hourly_olr_files = [
    file for file in all_olr_files
    if any(month in file for month in valid_months) and is_hourly_file(extract_timestamp(os.path.basename(file)))
]

rain_index = 0
hem_index = 0
processed_count = 0
print(len(hourly_olr_files), len(rainfall_files))

for iteration, olr_file in enumerate(hourly_olr_files):

    olr_timestamp = extract_timestamp(os.path.basename(olr_file))

    matching_rainfall_files = [file for file in rainfall_files if extract_timestamp(os.path.basename(file)) == olr_timestamp]

    if len(matching_rainfall_files) == 1:
        rainfall_file = matching_rainfall_files[0]
        olr_data = read_insat_file(olr_file, 'OLR')
        if olr_data is None:
            print(f"Skipping OLR file due to read error: {olr_file}")
            continue
        olr_data[olr_data <= 0] = 300
        if np.isnan(olr_data).sum() > 0:
            print(f"Skipping OLR file due to NaN values: {olr_file}")
            continue
        output_csv = f'combined_data_{olr_timestamp}.csv'
        output_path = os.path.join(output_folder, output_csv)
        rainfall_data, rainfall_lats, rainfall_lons = extract_rainfall(rainfall_file)
        binarized_data = adaptive_binarize(olr_data, 200, 200, 0.1)

        elev_10km = get_elev_map()
        mask = (binarized_data > 0)
        olr_max = 300 
        olr_data = np.where(mask, olr_data, olr_max)
        combine_data(rainfall_data, rainfall_lats, rainfall_lons, olr_data, output_path, iteration)

        print(f"Processed and saved: {output_path}")
        processed_count += 1
    else:
        print(f"Error: Matching files not found or multiple files found for OLR timestamp {olr_timestamp}")
    
print(f"Total files processed: {processed_count}")
print(f"Unprocessed OLR files: {len(hourly_olr_files) - processed_count}")