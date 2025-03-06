import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv


class DualHeadGATModel(nn.Module):
    def __init__(self,  in_channels=2, hidden_channels=64, out_channels=2,
                 num_layers=5, heads=8):
        """
        A multi-layer GAT model:
          - num_layers GATConv layers
          - heads attention heads per hidden layer
          - ReLU in between
        Args:
            in_channels: Input feature dimension (e.g., 2 for [T, c]).
            hidden_channels: Hidden dimension per head.
            out_channels: Output dimension (2 for [T, c]).
            num_layers: How many GATConv layers total.
            heads: Number of attention heads in the hidden layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.skip = nn.Linear(in_channels, out_channels)  # Skip connection

        # 1) First GAT layer: from in_channels to hidden_channels * heads
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        # 2) Middle GAT layers
        for _ in range(num_layers - 2):
            # Input has hidden_channels*heads; output also hidden_channels * heads
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))

        # 3) Final GAT layer: to out_channels, with a single head (concat=False)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
        else:
            # In case num_layers=1, we go directly from in_channels -> out_channels
            self.convs[0] = GATConv(in_channels, out_channels, heads=1, concat=False)

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, fixed_tof_mask):

        x_residual = self.skip(x)  # Preserve input features
        # x = self.update_tof_in_fmm_order(x, edge_index, fixed_tof_mask)
        for i, conv in enumerate(self.convs):
            x = self.relu(conv(x, edge_index))
        return x #+ self.relu(x_residual)

    def update_tof_in_fmm_order(self, x, edge_index, fixed_tof_mask):
        """
        Ensure TOF updates in FMM-style wavefront propagation order.
        """
        known_tof = x[:, 0].clone()
        known_tof[~fixed_tof_mask] = float("inf")  # Mark unknown TOF as high

        # Step 2: Iteratively update nearest unknowns first
        for _ in range(2):  # Control update depth
            for i, j in edge_index.T:
                if i < j and fixed_tof_mask[j]:  # If neighbor j has a fixed TOF value
                    known_tof[i] = min(known_tof[i], known_tof[j] + 1)

        x[:, 0] = known_tof  # Assign TOF back to tensor

        return x
