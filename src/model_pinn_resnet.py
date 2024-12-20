import torch.nn as nn
import segmentation_models_pytorch as smp


class TofToSosUNetModel(nn.Module):
    def __init__(self):
        super(TofToSosUNetModel, self).__init__()
        self.unet = smp.Unet(
            encoder_name='resnet34',      # Choose an encoder
            encoder_weights=None,         # No pre-training on ImageNet
            in_channels=1,                # TOF matrix has 1 channel
            classes=1,                    # SOS map has 1 channel
        )

    def forward(self, x):
        return self.unet(x)