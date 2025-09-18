import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime

base_dir = '/path/to/your/data'

pred_dirs = glob.glob(os.path.join(base_dir, 'pred_rr_*'))

os.makedirs('npz_files_without_height', exist_ok=True)

for pred_dir in pred_dirs:
    month_name = os.path.basename(pred_dir).replace('pred_rr_', '')
    
    if '_' in month_name:
        month_parts = month_name.split('_')
        if len(month_parts) == 2 and len(month_parts[1]) == 2:
            month_name = month_parts[0] + '20' + month_parts[1]
    csv_files = sorted(glob.glob(os.path.join(pred_dir, '*.csv')))
    
    if not csv_files:
        print(f"No CSV files found in {pred_dir}, skipping...")
        continue
    
    print(f"Processing {len(csv_files)} files for {month_name}...")
 
    olr_data = []
    rainfall_data = []
    time_data = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        timestamp = filename.replace('.csv', '')
        time_data.append(timestamp)
        
        try:
            df = pd.read_csv(csv_file)            
            olr = df['OLR'].values.reshape(400, 400)
            rainfall = df['Rainfall'].values.reshape(400, 400)
            
            olr_data.append(olr)
            rainfall_data.append(rainfall)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not olr_data:
        print(f"No valid data processed for {month_name}, skipping...")
        continue
    
    olr_array = np.array(olr_data) 
    rainfall_array = np.array(rainfall_data)  
    time_array = np.array(time_data)
    
    output_file = "path to save npz file"

    np.savez(output_file, Time=time_array, OLR=olr_array, Rainfall=rainfall_array)
    
    print(f"Saved {output_file} with {len(time_data)} timestamps")

print("Processing complete!")
