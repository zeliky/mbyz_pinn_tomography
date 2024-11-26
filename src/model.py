import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeedOfSoundCNN(nn.Module):
    def __init__(self ):
        super(SpeedOfSoundCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Output: [64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: [64, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: [128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: [128, 8, 8]
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Output: [64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Output: [32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # Output: [16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # Output: [1, 128, 128]

            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Output shape: [batch_size, 1, 128, 128]


class TravelTimeFCNN(nn.Module):
    def __init__(self):
        super(TravelTimeFCNN, self).__init__()
        self.fcnn = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x_r, x_s):
        input_features = torch.cat([x_r, x_s], dim=1)  # Shape: [batch_size, 4]
        T = self.fcnn(input_features)
        return T  # Predicted travel time in seconds ??


class PINNModel(nn.Module):
    def __init__(self, c_min=1400.0, c_max=1600.0):
        super(PINNModel, self).__init__()
        self.cnn = SpeedOfSoundCNN()
        self.fcnn = TravelTimeFCNN()

    def forward(self, tof_input, x_r, x_s, x_coords):
        c = self.cnn(tof_input)  # Estimated SoS map
        T_pred = self.fcnn(x_r, x_s)  # Predicted travel time at receivers
        T_grid = self.fcnn(x_coords, x_s)  # Predicted T(x) at grid points for physics loss
        return c, T_pred, T_grid
