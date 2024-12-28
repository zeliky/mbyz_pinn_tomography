import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders import get_decoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class CombinedSosTofModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_encoder(
            name='resnet34',
            in_channels=1,  # single-channel input
            weights=None
        )
        encoder_channels = self.encoder.out_channels

        self.decoder_sos = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=True,  # UNet center block (VGG style)
        )

        self.decoder_tof = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=True,
        )

        # 1) For SoS
        self.upsample_sos = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # or no activation if it doesn't fit your data
        )
        # 2) For ToF
        self.upsample_tof = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            #nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            #nn.Sigmoid(),
        )



    def forward(self, x):
        """
        x shape: [B, 1, 32, 32]
        We want final outputs:
          - sos_out: [B, 1,   128, 128]
          - tof_out: [B, 32, 128, 128]
        """
        # --- 1) Extract encoder features ---
        features = self.encoder(x)

        # --- 2) Decode the features for SoS ---
        sos_decoded = self.decoder_sos(*features)  # [B, 16, 32, 32] default output since input is [32,32]
        sos_out = self.upsample_sos(sos_decoded) # [B, 1, 128, 128] #reqired SOS image with 128x128 c(x,y) values

        # --- 3) Decode the features for ToF ---
        tof_decoded = self.decoder_tof(*features)  # [B, 16, 32, 32] by default
        tof_out = self.upsample_tof(tof_decoded) # [B, 1, 128, 128] #reqired TOF per source tensor with 32 and 128x128 T(x,y) values

        return sos_out, tof_out
