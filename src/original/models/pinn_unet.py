import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A simple block of (Conv -> ReLU -> Conv -> ReLU),
    optionally with batch normalization, etc.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    """
    Example encoder that downsamples input from 32x32 -> 8x8 or 4x4.
    Each step: ConvBlock -> maxpool (stride=2).
    """

    def __init__(self, in_channels=1, base_filters=32):
        super(Encoder, self).__init__()
        # You can adjust the depth as you like
        self.block1 = ConvBlock(in_channels, base_filters)
        self.block2 = ConvBlock(base_filters, base_filters * 2)
        self.block3 = ConvBlock(base_filters * 2, base_filters * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B, in_channels, 32, 32]
        x = self.block1(x)  # [B, base_filters, 32, 32]
        x = self.pool(x)  # [B, base_filters, 16, 16]
        x = self.block2(x)  # [B, base_filters*2, 16, 16]
        x = self.pool(x)  # [B, base_filters*2, 8, 8]
        x = self.block3(x)  # [B, base_filters*4, 8, 8]
        # optionally another pool -> [B, base_filters*4, 4, 4]
        return x


class Decoder(nn.Module):
    """
    Example decoder that upsamples from 8x8 -> 128x128.
    Each step: (deconv/upsample) -> ConvBlock.
    """

    def __init__(self, out_channels=1, base_filters=32):
        super(Decoder, self).__init__()
        # Reverse of encoder
        self.up1 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2,
                                      kernel_size=2, stride=2)
        self.block1 = ConvBlock(base_filters * 2, base_filters * 2)

        self.up2 = nn.ConvTranspose2d(base_filters * 2, base_filters,
                                      kernel_size=2, stride=2)
        self.block2 = ConvBlock(base_filters, base_filters)

        # Another upsampling to go from 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(base_filters, base_filters // 2,
                                      kernel_size=2, stride=2)
        self.block3 = ConvBlock(base_filters // 2, base_filters // 2)

        # Another upsampling from 64x64 -> 128x128
        self.up4 = nn.ConvTranspose2d(base_filters // 2, base_filters // 4,
                                      kernel_size=2, stride=2)
        # final block
        self.block4 = nn.Sequential(
            nn.Conv2d(base_filters // 4, out_channels, kernel_size=3, padding=1)
            # no ReLU if we want raw predicted T (could do one final conv)
        )

    def forward(self, x):
        # x: [B, base_filters*4, 8, 8]
        x = self.up1(x)  # [B, base_filters*2, 16, 16]
        x = self.block1(x)  # [B, base_filters*2, 16, 16]

        x = self.up2(x)  # [B, base_filters, 32, 32]
        x = self.block2(x)  # [B, base_filters, 32, 32]

        x = self.up3(x)  # [B, base_filters//2, 64, 64]
        x = self.block3(x)  # [B, base_filters//2, 64, 64]

        x = self.up4(x)  # [B, base_filters//4, 128, 128]
        x = self.block4(x)  # [B, out_channels, 128, 128]
        return x


class MultiSourceTOFModel(nn.Module):
    """
    End-to-end model:
      - Takes a batch of (B, 1, 32, 32) images (the "TOF matrix" or conditioning info)
      - Encodes -> Decodes -> Produces [B, n_src, 128, 128] output
    """

    def __init__(self, in_channels=1, n_src=8, base_filters=32):
        super(MultiSourceTOFModel, self).__init__()
        self.encoder = Encoder(in_channels, base_filters)
        self.decoder = Decoder(out_channels=n_src, base_filters=base_filters)

    def forward(self, x):
        # x: [B, in_channels, 32, 32]
        enc = self.encoder(x)  # e.g. [B, base_filters*4, 8, 8]
        out = self.decoder(enc)  # e.g. [B, n_src, 128, 128]
        return out

