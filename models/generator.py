import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    """
    Pixel-wise feature vector normalization.
    Scales latent vectors to unit length for StyleGAN stability.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, eps=1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)

class AdaptiveInstanceNorm(nn.Module):
    """
    AdaIN: Modulates feature maps using learned style vectors.
    """
    def __init__(self, channels, style_dim):
        super().__init__()
        self.fc_scale = nn.Linear(style_dim, channels)
        self.fc_bias = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        b, c, h, w = x.shape
        scale = self.fc_scale(style).view(b, c, 1, 1)
        bias = self.fc_bias(style).view(b, c, 1, 1)

        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8

        normalized = (x - mean) / std
        return normalized * scale + bias

class StyledConvBlock(nn.Module):
    """
    Core convolutional block of StyleGAN2.
    """
    def __init__(self, in_channels, out_channels, style_dim, upsample):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        if self.upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.activation(x)
        return x

class Generator(nn.Module):
    """
    StyleGAN2 Generator for 64x64 resolution.
    """
    def __init__(self, latent_dim=100, style_dim=512):
        super().__init__()
        self.pixel_norm = PixelNorm()

        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )

        self.constant_input = nn.Parameter(torch.randn(1, 512, 4, 4))

        self.blocks = nn.ModuleList([
            StyledConvBlock(512, 256, style_dim, upsample=True),  # 8x8
            StyledConvBlock(256, 128, style_dim, upsample=True),  # 16x16
            StyledConvBlock(128, 64, style_dim, upsample=True),   # 32x32
            StyledConvBlock(64, 32, style_dim, upsample=True),    # 64x64
        ])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()  # map outputs to [-1, 1] for normalized images
        )

    def forward(self, z):
        """
        z: (batch_size, latent_dim) random noise vector
        """
        style = self.mapping(self.pixel_norm(z))
        batch_size = z.shape[0]
        x = self.constant_input.repeat(batch_size, 1, 1, 1)
        for block in self.blocks:
            x = block(x, style)
        return self.to_rgb(x)
