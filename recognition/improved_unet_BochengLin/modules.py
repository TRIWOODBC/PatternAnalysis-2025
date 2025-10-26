import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3d(nn.Module):
    """
    Channel attention mechanism for 3D features.
    
    Uses both average and max pooling to capture channel statistics,
    then applies a lightweight MLP to learn channel importance weights.
    This allows the network to adaptively recalibrate feature maps.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Lightweight MLP with bottleneck
        mid_channels = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get channel attention weights from both pooling strategies
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply sigmoid to get weights in [0, 1]
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ConvBlock(nn.Module):
    """
    Standard 3D convolution block with two conv layers.
    
    Each layer includes: Conv3d -> InstanceNorm -> LeakyReLU
    Instance norm works well with small batch sizes in medical imaging.
    """
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
    """
    Encoding block with learned downsampling.
    
    Uses strided convolution instead of max-pooling, which allows
    the network to learn the best way to downsample for this task.
    """
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
    """
    Decoding block with channel attention.
    
    Combines upsampling, skip connections, and adaptive channel attention
    to refocus on the most informative features during reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation with skip connection, we have double the channels
        self.conv = ConvBlock(out_channels * 2, out_channels)
        # Learn which channels matter most at this level
        self.attention = ChannelAttention3d(out_channels)

    def forward(self, x1, x2):
        # Upsample the deeper feature map
        x1 = self.up(x1)
        
        # Pad to match skip connection size (handles odd-sized volumes)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Merge upsampled features with skip connection
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        # Adaptively weight channels using attention
        x = self.attention(x)
        
        return x


class OutConv(nn.Module):
    """Final output layer: reduces features to class predictions."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D_Improved(nn.Module):
    """
    3D U-Net with channel attention for medical image segmentation.
    
    Encoder-decoder with skip connections and attention at each decoder stage.
    Designed for 3D medical imaging with small batch sizes.
    """
    def __init__(self, in_channels, num_classes):
        super(UNet3D_Improved, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Downsampling path (encoder)
        self.inc = ConvBlock(in_channels, 32)
        self.down1 = DownBlockImproved(32, 64)
        self.down2 = DownBlockImproved(64, 128)
        self.down3 = DownBlockImproved(128, 256)
        self.down4 = DownBlockImproved(256, 320)

        # Upsampling path (decoder) with attention
        self.up1 = UpBlockImproved(320, 256)
        self.up2 = UpBlockImproved(256, 128)
        self.up3 = UpBlockImproved(128, 64)
        self.up4 = UpBlockImproved(64, 32)
        
        # Final output
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        # Encoder: extract multi-scale features
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder: reconstruct with skip connections and attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output: class predictions
        logits = self.outc(x)
        return logits