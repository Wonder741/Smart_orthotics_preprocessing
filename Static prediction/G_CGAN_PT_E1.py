import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image

# Define paths
data_root = r'/home/uqcwan/CGAN/Data'
pdd_path = os.path.join(data_root, 'PDD')
scan_path = os.path.join(data_root, 'SCAN')
debug_files = os.path.join(data_root, 'DEBUG')
# Create the destination directory if it doesn't exist
if not os.path.exists(debug_files):
    os.makedirs(debug_files)


# Hyperparameters
epochs = 1000
batch_size = 1024
learning_rate = 0.001
image_size = 64

# Custom Dataset
class PairedDataset(Dataset):
    def __init__(self, pdd_path, scan_path, transform=None):
        self.pdd_path = pdd_path
        self.scan_path = scan_path
        self.transform = transform
        self.image_names = os.listdir(pdd_path)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        pdd_image = Image.open(os.path.join(self.pdd_path, self.image_names[idx])).convert('L')
        scan_image = Image.open(os.path.join(self.scan_path, self.image_names[idx])).convert('L')
        
        if self.transform:
            pdd_image = self.transform(pdd_image)
            scan_image = self.transform(scan_image)
        
        return {'PDD': pdd_image, 'SCAN': scan_image}

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Dataset and DataLoader
dataset = PairedDataset(pdd_path, scan_path, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Define layers here
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

# Initialize models
G = Generator().cuda()
D = Discriminator().cuda()

# Loss functions
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for i, batch in enumerate(dataloader):
        real_scan = batch['SCAN'].cuda()
        real_pdd = batch['PDD'].cuda()
        
        # Adjust valid and fake tensor sizes to match discriminator output
        valid = torch.ones((real_scan.size(0), 1, 4, 4), requires_grad=False).cuda()
        fake = torch.zeros((real_scan.size(0), 1, 4, 4), requires_grad=False).cuda()

        # Train Generator
        optimizer_G.zero_grad()
        fake_pdd = G(real_scan)
        loss_GAN = criterion_GAN(D(fake_pdd, real_scan), valid)
        loss_L1 = criterion_L1(fake_pdd, real_pdd)
        loss_G = loss_GAN + 100 * loss_L1
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        loss_real = criterion_GAN(D(real_pdd, real_scan), valid)
        loss_fake = criterion_GAN(D(fake_pdd.detach(), real_scan), fake)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Loss D: {loss_D.item()}, loss G: {loss_G.item()}")

    if epoch % 100 == 0:
        debug_files = os.path.join(data_root, 'DEBUG', f"output_{epoch}.png")
        save_image(fake_pdd.data[:25], debug_files, nrow=5, normalize=True)

# Save models
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
