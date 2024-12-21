import torch
import torch.nn as nn
import torch.nn.functional as F

class ToFPredictor(nn.Module):
    def __init__(self, num_sources=32, w=128, h=128):
        super(ToFPredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        self.num_sources = num_sources
        self.W = w
        self.H = h

    def forward(self, x):
        outputs = []
        for i in range(self.num_sources):

            layer = x[:, i, :, :].unsqueeze(1)  # predict tof on each layer (1 source all receivers)
            encoded = self.encoder(layer)
            decoded = self.decoder(encoded)
            outputs.append(decoded)
        return torch.cat(outputs, dim=1)  # Concatenate all layers

