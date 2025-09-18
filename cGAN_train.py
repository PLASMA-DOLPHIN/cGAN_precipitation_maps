import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np

class Pix2PixDataset(Dataset):
    def __init__(self, npz_file, num_samples=150):
        data = np.load(npz_file)
        self.input_data = data['OLR'] / 200.0 - 1 
        self.target_data = data['Rainfall'] / 100.0 - 1  

        print(f"Input data shape: {self.input_data.shape}")
        print(f"Target data shape: {self.target_data.shape}")
        print(f"Input data min: {self.input_data.min()}, max: {self.input_data.max()}")
        print(f"Target data min: {self.target_data.min()}, max: {self.target_data.max()}")
            
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

class Discriminator(nn.Module):
    def __init__(self, in_channels=4): 
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        generated = generated.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return F.mse_loss(gen_features, target_features)

def train_improved_pix2pix(generator, discriminator, train_loader, num_epochs, device):
    criterion_gan = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=num_epochs)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=num_epochs)

    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(train_loader):
            real_A, real_B = real_A.to(device), real_B.to(device)

            optimizer_d.zero_grad()

            fake_B = generator(real_A)
            pred_fake = discriminator(real_A, fake_B.detach().repeat(1, 3, 1, 1))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            pred_real = discriminator(real_A, real_B.repeat(1, 3, 1, 1))
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            fake_B = generator(real_A)
            pred_fake = discriminator(real_A, fake_B.repeat(1, 3, 1, 1))
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            loss_g_pixel = criterion_pixelwise(fake_B, real_B) * 100
            loss_g_perceptual = criterion_perceptual(fake_B, real_B) * 10

            loss_g = loss_g_gan + loss_g_pixel + loss_g_perceptual
            loss_g.backward()
            optimizer_g.step()

        scheduler_g.step()
        scheduler_d.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")

    return generator

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    npz_file = 'your_training_data.npz'
   
    dataset = Pix2PixDataset(npz_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    generator = ImprovedGenerator().to(device)
    discriminator = Discriminator().to(device)
    
    num_epochs = 10 
    trained_generator = train_improved_pix2pix(generator, discriminator, dataloader, num_epochs, device)
    torch.save(trained_generator.state_dict(), 'your_saved-model.pth')
    print("Model saved successfully!")
