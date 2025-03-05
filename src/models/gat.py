import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv


class DualHeadGAT(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2,
                 num_layers=3, heads=4):
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

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = self.relu(conv(x, edge_index))

        return x
