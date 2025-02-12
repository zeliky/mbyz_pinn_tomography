import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class TOFtoSOSPINNLinerModel(nn.Module):

    def __init__(self, num_sources):
        self.num_sources =num_sources
        super(TOFtoSOSPINNLinerModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + num_sources, 128),  # (x, y, ToF values per source)
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positivity of SoS
        )

    def forward(self, x, y, tof_inputs):
        inputs = torch.cat([x, y, tof_inputs], dim=1)
        return self.net(inputs)

