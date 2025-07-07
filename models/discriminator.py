import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.blocks = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, 1)  # MATCHES YOUR COLAB TRAINING

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits
