import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """
    3D convolution block with dilated convolutions for receptive field expansion.
    
    Uses dilation to increase receptive field without reducing spatial resolution.
    Stacks two dilated convs with instance norm and LeakyReLU activation.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = dilation  # Maintain spatial dimensions
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class ContextAggregationModule(nn.Module):
    """
    Multi-scale context aggregation using progressive dilations.
    
    Applies parallel dilated convolutions (dilations 1,2,4,8) to capture
    multi-scale contextual information at full resolution.
    Concatenates all branches and projects back to output channels.
    """
    def __init__(self, channels, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.branches = nn.ModuleList()
        for dil in dilations:
            self.branches.append(DilatedConvBlock(channels, channels, dilation=dil))
        
        # Projection: concatenates all dilations (channels * len(dilations) -> channels)
        self.projection = nn.Sequential(
            nn.Conv3d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        # Apply all dilated convolutions in parallel
        branch_outs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        concat = torch.cat(branch_outs, dim=1)
        
        # Project back to original channel size
        out = self.projection(concat)
        
        # Residual connection
        return out + x


class CAN3D(nn.Module):
    """
    Context Aggregation Network 3D for medical image segmentation (memory-efficient).
    
    Full-resolution dense prediction using multi-scale dilated convolutions.
    Minimizes downsampling (1 level) to preserve spatial detail while reducing memory.
    Uses progressive dilations (1,2,4,8) for receptive field expansion.
    """
    def __init__(self, in_channels, num_classes):
        super(CAN3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initial feature extraction (no downsampling)
        self.init_conv = DilatedConvBlock(in_channels, 32, dilation=1)
        
        # Context aggregation at full resolution (2 stages instead of 4)
        self.context1 = ContextAggregationModule(32, dilations=[1, 2, 4])
        self.context2 = ContextAggregationModule(32, dilations=[1, 2, 4])
        
        # Single downsampling for computational efficiency
        self.downsample = nn.Conv3d(32, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_ds = nn.InstanceNorm3d(48)
        self.relu_ds = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Context at reduced resolution (lighter: dilations 1,2 only)
        self.context_low = ContextAggregationModule(48, dilations=[1, 2])
        
        # Upsample back to full resolution
        self.upsample = nn.ConvTranspose3d(48, 32, kernel_size=2, stride=2)
        
        # Output: 1x1x1 convolution for class predictions
        self.out_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Initial feature extraction
        x = self.init_conv(x)  # (B, 32, H, W, D)
        
        # Context aggregation at full resolution
        x = self.context1(x)
        x = self.context2(x)
        
        # Downsampling path
        x_ds = self.relu_ds(self.norm_ds(self.downsample(x)))  # (B, 48, H/2, W/2, D/2)
        
        # Context at lower resolution
        x_ds = self.context_low(x_ds)
        
        # Upsample back to full resolution
        x_up = self.upsample(x_ds)
        
        # Pad/crop to match original spatial dimensions (handles odd sizes)
        if x_up.shape[2:] != x.shape[2:]:
            diffZ = x.size()[2] - x_up.size()[2]
            diffY = x.size()[3] - x_up.size()[3]
            diffX = x.size()[4] - x_up.size()[4]
            x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2,
                                diffZ // 2, diffZ - diffZ // 2])
        
        # Merge upsampled features with original
        x = x + x_up  # Residual connection
        
        # Output: class predictions at full resolution
        logits = self.out_conv(x)
        
        return logits


# Backwards compatibility alias
UNet3D_Improved = CAN3D