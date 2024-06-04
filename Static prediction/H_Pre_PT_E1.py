import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = self.downsample(1, 64, apply_batchnorm=False)  # 64x64 -> 32x32
        self.down2 = self.downsample(64, 128)                      # 32x32 -> 16x16
        self.down3 = self.downsample(128, 256)                     # 16x16 -> 8x8
        self.down4 = self.downsample(256, 512)                     # 8x8 -> 4x4
        self.down5 = self.downsample(512, 512)                     # 4x4 -> 2x2
        self.down6 = self.downsample(512, 512)                     # 2x2 -> 1x1

        self.up1 = self.upsample(512, 512, apply_dropout=True)     # 1x1 -> 2x2
        self.up2 = self.upsample(1024, 512, apply_dropout=True)    # 2x2 -> 4x4
        self.up3 = self.upsample(1024, 512, apply_dropout=True)    # 4x4 -> 8x8
        self.up4 = self.upsample(1024, 256)                        # 8x8 -> 16x16
        self.up5 = self.upsample(512, 128)                         # 16x16 -> 32x32
        self.up6 = self.upsample(256, 64)                          # 32x32 -> 64x64
        self.last = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)  # 64x64 -> 128x128

    def downsample(self, in_channels, out_channels, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def upsample(self, in_channels, out_channels, apply_dropout=False):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.BatchNorm2d(out_channels))
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        down1 = self.down1(x)                                      # 64x64x1 -> 32x32x64
        down2 = self.down2(down1)                                  # 32x32x64 -> 16x16x128
        down3 = self.down3(down2)                                  # 16x16x128 -> 8x8x256
        down4 = self.down4(down3)                                  # 8x8x256 -> 4x4x512
        down5 = self.down5(down4)                                  # 4x4x512 -> 2x2x512
        down6 = self.down6(down5)                                  # 2x2x512 -> 1x1x512

        up1 = self.up1(down6)                                      # 1x1x512 -> 2x2x512
        up1 = torch.cat([up1, down5], dim=1)                       # 2x2x512 -> 2x2x1024
        up2 = self.up2(up1)                                        # 2x2x1024 -> 4x4x512
        up2 = torch.cat([up2, down4], dim=1)                       # 4x4x512 -> 4x4x1024
        up3 = self.up3(up2)                                        # 4x4x1024 -> 8x8x512
        up3 = torch.cat([up3, down3], dim=1)                       # 8x8x512 -> 8x8x1024
        up4 = self.up4(up3)                                        # 8x8x1024 -> 16x16x256
        up4 = torch.cat([up4, down2], dim=1)                       # 16x16x256 -> 16x16x512
        up5 = self.up5(up4)                                        # 16x16x512 -> 32x32x128
        up5 = torch.cat([up5, down1], dim=1)                       # 32x32x128 -> 32x32x256
        up6 = self.up6(up5)                                        # 32x32x256 -> 64x64x64
        return self.last(up6)                                      # 64x64x64 -> 128x128x1

# Load the trained generator model
generator = Generator().cuda()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # Set the model to evaluation mode

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.cuda()

data_root = r'\\wsl.localhost\Ubuntu\home\uqcwan\CGAN\Data'
file_name = 'left_01wangchongguang_1.png'
# Load and preprocess the input image
input_image_path = os.path.join(data_root, 'PDD', file_name)
input_image = load_image(input_image_path)

# Generate prediction
with torch.no_grad():
    generated_image = generator(input_image)

# Post-process the generated image
generated_image = generated_image.squeeze(0)  # Remove batch dimension
generated_image = generated_image.cpu().detach()
generated_image = (generated_image + 1) / 2  # De-normalize to [0, 1]

# Visualize the output
plt.imshow(generated_image.squeeze(0), cmap='gray')
plt.axis('off')
plt.show()

# Save the output image
output_image = transforms.ToPILImage()(generated_image)
output_image.save(file_name)
