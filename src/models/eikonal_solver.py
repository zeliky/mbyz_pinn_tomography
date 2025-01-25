import torch
import torch.nn as nn
import torch.nn.functional as F

class EikonalSolverMultiLayer(nn.Module):
    def __init__(self, num_layers, speed_of_sound, domain_size, grid_resolution, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            for _ in range(num_layers)
        ])

        # Initialize weights for slowness (1/c)
        self.delta_s = domain_size / grid_resolution
        #print(delta_s)
        #print(delta_s / speed_of_sound)
        for conv in self.convs:
            conv.weight.data.fill_(self.delta_s / speed_of_sound)

    def forward(self, T_init, sos_pred):
        T = T_init.clone()

        for conv in self.convs:
            # Compute tentative updates from neighbors
            slowness = 1.0 / sos_pred
            #slowness = self.delta_s / sos_pred.clamp(min=1e-8)  # Add batch and channel dims

            T_neighbors = conv(T) * slowness
            # Update travel time by propagating the minimum travel time
            T = torch.min(T, T_neighbors)

        return T

