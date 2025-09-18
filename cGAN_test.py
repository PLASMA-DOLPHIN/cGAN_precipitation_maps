import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import csv
from scipy.stats import gamma
import pandas as pd
from datetime import datetime, timedelta

class Pix2PixTestDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        
        self.input_data = data['OLR'] / 200.0 - 1 
        self.target_data = data['Rainfall'] / 100.0 - 1 


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_image = self.input_data[idx]
        target_image = self.target_data[idx]
        
        input_image = torch.from_numpy(input_image).float().unsqueeze(0)
        target_image = torch.from_numpy(target_image).float().unsqueeze(0)
        
        input_image = F.pad(input_image, (0, 512 - input_image.shape[2], 0, 512 - input_image.shape[1]))
        target_image = F.pad(target_image, (0, 512 - target_image.shape[2], 0, 512 - target_image.shape[1]))
        
        return input_image, target_image

class RainfallCalibrator:
    def __init__(self, method='exponential', alpha=1.0):
 
        self.method = method
        self.alpha = alpha
        self.threshold = 0.0  
        
    def calibrate(self, predictions, olr_values):

        predictions = np.array(predictions)
        olr_values=np.array(olr_values)
        mask = (predictions > self.threshold) & (olr_values < 150)
        
        if self.method == 'exponential':
            scaled = predictions.copy()
            scaled[mask] = predictions[mask] * np.exp((predictions[mask] - self.threshold) / 50.0 * self.alpha)
            return scaled
            
        elif self.method == 'power':
            scaled = predictions.copy()
            scaled[mask] = predictions[mask] * (1 + (predictions[mask] - self.threshold) / 50.0) ** self.alpha
            return scaled
            
        elif self.method == 'gamma':
            scaled = predictions.copy()
            shape = 2.0
            scale = self.alpha
            scaled[mask] = predictions[mask] * (1 + gamma.pdf(predictions[mask]/20, shape, scale=scale))
            return scaled
            
        return predictions

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ImprovedGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=9):
        super(ImprovedGenerator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def compute_metrics(true, pred):
    rmse = np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))
    corr = np.corrcoef(true.flatten(), pred.flatten())[0, 1]
    return rmse, corr

def compute_threat_score(true, pred, threshold):
    true_binary = true > threshold
    pred_binary = pred > threshold
    
    true_positive = np.sum(np.logical_and(true_binary, pred_binary))
    false_positive = np.sum(np.logical_and(np.logical_not(true_binary), pred_binary))
    false_negative = np.sum(np.logical_and(true_binary, np.logical_not(pred_binary)))
    
    threat_score = true_positive / (true_positive + false_positive + false_negative + 1e-10)
    return threat_score

def plot_comparison(true, pred, index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(true, cmap='viridis',vmax=30)
    ax1.set_title(f'True Rainfall (Image {index+1})')
    plt.colorbar(im1, ax=ax1, label='mm/day')

    im2 = ax2.imshow(pred, cmap='viridis',vmax=30)
    ax2.set_title(f'Generated Rainfall (Image {index+1})')
    plt.colorbar(im2, ax=ax2, label='mm/day')

    plt.tight_layout()
    plt.show()

def test_model(model, test_loader, device, data, calibrator, csv_folder):
    model.eval()
    filenames = data['Time']
    
    if not os.path.exists(csv_folder):
        print(f"Folder {csv_folder} does not exist, skipping.")
        return
    
    with torch.no_grad():
        for i, (input_image, target_image) in enumerate(test_loader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)
            
            generated_image = model(input_image)
            
            generated_image = (generated_image + 1) * 100.0
            target_image = (target_image + 1) * 100.0
            
            generated_image = generated_image[:, :, :400, :400].reshape(-1, 160000)
            target_image = target_image[:, :, :400, :400]
            
            generated_np = generated_image.cpu().numpy().squeeze()
            target_np = target_image.cpu().numpy().squeeze()
            
            generated_np = calibrator.calibrate(generated_np, data['OLR'][i,:,:].flatten())
            
            csv_filename = f"{filenames[i]}.csv"
            csv_path = os.path.join(csv_folder, csv_filename)
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                if len(generated_np) == len(df):
                    df['Scaled_Generated_without_height_jun18_may19'] = np.round(generated_np, 1)
                    df.to_csv(csv_path, index=False)
                    print(f"Updated {csv_path} with generated data")
                else:
                    print(f"Skipping {csv_path}: Index mismatch (Generated: {len(generated_np)}, CSV: {len(df)})")
                    if 'Generated' in df.columns:
                        df['Scaled_Generated_without_height_jun18_may19'] = df['Generated']
                
                df.to_csv(csv_path, index=False)
            else:
                print(f"CSV file {csv_filename} not found in the specified folder")
    
    print(f"Test completed for {csv_folder}. Generated data added to existing CSV files.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImprovedGenerator().to(device)
    model.load_state_dict(torch.load('your_saved_model.pth', map_location=device))
    

    start = datetime(2019, 6, 1)
    end = datetime(2022, 1, 1)
    month_years = []
    current = start
    while current < end:
        month = current.strftime('%b') 
        year = current.strftime('%Y')   
        month_years.append((month, year))
        next_month = current.replace(day=28) + timedelta(days=4)
        current = next_month.replace(day=1)

    calibrator = RainfallCalibrator(method='power', alpha=2.0)
    
    for month, year in month_years:
        test_npz_file = f'npz_files_without_height/pix2pix_{month}{year}_data.npz'
        
        if not os.path.exists(test_npz_file):
            print(f"NPZ file {test_npz_file} not found, skipping {month} {year}.")
            continue
        
        print(f"Processing {month} {year}...")
        
        test_dataset = Pix2PixTestDataset(test_npz_file)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        data = np.load(test_npz_file)
        
        csv_folder = f'/Predictions_without_height/pred_rr_{month}_{year[-2:]}'
        os.makedirs(csv_folder, exist_ok=True)
        
        test_model(model, test_loader, device, data, calibrator, csv_folder)
        
        print(f"Completed processing {month} {year}.")