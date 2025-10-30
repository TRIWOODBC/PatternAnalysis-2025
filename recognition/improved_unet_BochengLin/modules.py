import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualContextBlock(nn.Module):
    """
    Residual 3D convolutional block inspired by the BraTS 2017 U-Net variant.

    This block performs two 3×3×3 convolutions with Instance Normalization
    and LeakyReLU activations, plus a residual skip connection.

    Dropout is used between the two convolutions to improve generalization
    and robustness against overfitting. This block preserves the spatial
    resolution of its input.

    Reference:
        Isensee et al., "Brain Tumor Segmentation and Radiomics Survival Prediction:
        Contribution to the BRATS 2017 Challenge", arXiv:1802.10508
    """
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout3d(p=dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        # Adjust residual path when input and output channels differ
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)
        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling block using a stride-2 convolution.

    The original BraTS 2017 U-Net replaced max pooling with strided convolutions
    to improve feature continuity and allow learnable downsampling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpsampleBlock(nn.Module):
    """
    Upsampling block using trilinear interpolation followed by 3×3×3 convolution.

    The paper avoids transposed convolutions (deconvolutions) to prevent
    checkerboard artifacts and uses nearest or trilinear interpolation instead.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        # Upsample by a factor of 2 using trilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = self.act(self.norm(self.conv(x)))
        return x


class UNet3D_Improved(nn.Module):
    """
    Improved 3D U-Net based on the BraTS 2017 Challenge architecture.

    This implementation follows the design principles from:
    - Residual convolutional blocks with dropout (0.3)
    - Strided convolution for downsampling (no pooling)
    - Trilinear interpolation for upsampling (no transposed convolution)
    - Instance normalization and LeakyReLU activations throughout
    - 3D convolutions for full volumetric context

    The model is lightweight enough for 128³ input volumes with a batch size of 2–4
    on a 16GB GPU, while maintaining high segmentation accuracy.

    Reference:
        Isensee et al., arXiv:1802.10508
    """
    def __init__(self, in_channels=1, num_classes=6, base_filters=32):
        super().__init__()

        # ---------- Encoder path ----------
        self.enc1 = ResidualContextBlock(in_channels, base_filters)
        self.down1 = DownsampleBlock(base_filters, base_filters * 2)

        self.enc2 = ResidualContextBlock(base_filters * 2, base_filters * 2)
        self.down2 = DownsampleBlock(base_filters * 2, base_filters * 4)

        self.enc3 = ResidualContextBlock(base_filters * 4, base_filters * 4)
        self.down3 = DownsampleBlock(base_filters * 4, base_filters * 8)

        # Bottleneck (deepest layer)
        self.bottleneck = ResidualContextBlock(base_filters * 8, base_filters * 8)

        # ---------- Decoder path ----------
        self.up3 = UpsampleBlock(base_filters * 8, base_filters * 4)
        self.dec3 = ResidualContextBlock(base_filters * 8, base_filters * 4)

        self.up2 = UpsampleBlock(base_filters * 4, base_filters * 2)
        self.dec2 = ResidualContextBlock(base_filters * 4, base_filters * 2)

        self.up1 = UpsampleBlock(base_filters * 2, base_filters)
        self.dec1 = ResidualContextBlock(base_filters * 2, base_filters)

        # Final output layer (1×1×1 conv)
        self.out_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # ---------- Encoder ----------
        e1 = self.enc1(x)                 # Level 1
        e2 = self.enc2(self.down1(e1))    # Level 2
        e3 = self.enc3(self.down2(e2))    # Level 3
        b = self.bottleneck(self.down3(e3))  # Bottleneck

        # ---------- Decoder ----------
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # ---------- Output ----------
        logits = self.out_conv(d1)
        return logits