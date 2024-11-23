import torch
import torch.nn as nn

class EikonalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        # Decoder
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(64, 1, final=True)

    def conv_block(self, in_channels, out_channels, final=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if not final:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        # Decoder
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)
        return dec1