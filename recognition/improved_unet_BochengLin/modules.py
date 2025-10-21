import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution: Conv3d + InstanceNorm + LeakyReLU, repeated twice."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DownBlockImproved(nn.Module):
    """Downsampling with strided convolution instead of max-pooling."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample_conv(x)


class UpBlockImproved(nn.Module):
    """Upsampling with ConvTranspose3d and skip connection concatenation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # Concatenation doubles channels: out_channels * 2
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad to match skip connection size (handles odd dimensions)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate with skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1x1 convolution for final output."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D_Improved(nn.Module):
    """
    3D U-Net architecture for semantic segmentation.
    
    Encoder: 1 -> 32 -> 64 -> 128 -> 256 -> 320 channels (downsampling)
    Decoder: 320 -> 256 -> 128 -> 64 -> 32 channels (upsampling with skip connections)
    Output: num_classes channels
    
    Key features:
    - Strided convolutions for downsampling (learnable)
    - Instance normalization (better for small batch sizes)
    - LeakyReLU activations
    - Skip connections to preserve fine details
    """
    def __init__(self, in_channels, num_classes):
        super(UNet3D_Improved, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.inc = ConvBlock(in_channels, 32)
        self.down1 = DownBlockImproved(32, 64)
        self.down2 = DownBlockImproved(64, 128)
        self.down3 = DownBlockImproved(128, 256)
        self.down4 = DownBlockImproved(256, 320)

        # Decoder
        self.up1 = UpBlockImproved(320, 256)
        self.up2 = UpBlockImproved(256, 128)
        self.up3 = UpBlockImproved(128, 64)
        self.up4 = UpBlockImproved(64, 32)
        
        # Output
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits