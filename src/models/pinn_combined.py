import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class CombinedSosTofModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- 1) Shared Encoder -----
        self.encoder = get_encoder(
            name='resnet34',
            in_channels=1,  # single-channel input
            weights=None
        )
        # SMP encoders store a list of channel dimensions for each downsample stage:
        encoder_channels = self.encoder.out_channels  # e.g. [64, 64, 128, 256, 512] for ResNet34

        # ----- 2) Decoder for SOS -----
        self.decoder_sos = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 32),
            n_blocks=5,
            use_batchnorm=True,
            center=True,  # typically True for VGG-like encoder, can be True/False for ResNet
        )


        # ----- 3) Decoder for TOF -----
        self.decoder_tof = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 32),
            n_blocks=5,
            use_batchnorm=True,
            center=True,
        )

        # Final conv for 1-channel SOS output
        self.head_sos = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(),
        )
        # Final conv for 32-channel TOF output
        self.head_tof = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU()

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
        sos_decoder_output  = self.decoder_sos(*features)  # [B, 16, 32, 32] default output since input is [32,32]
        sos_out = self.head_sos(sos_decoder_output ) # [B, 1, 128, 128] #reqired SOS image with 128x128 c(x,y) values


        # --- 3) Decode the features for ToF ---
        tof_decoder_output  = self.decoder_tof(*features)  # [B, 16, 32, 32] by default
        tof_out = self.head_tof(tof_decoder_output ) # [B, 1, 128, 128] #reqired TOF per source tensor with 32 and 128x128 T(x,y) values
        return sos_out, tof_out
