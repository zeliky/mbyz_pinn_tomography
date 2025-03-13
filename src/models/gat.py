import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn, MessagePassing
from torch_geometric.nn import GATConv


class FMMMessagePassing(MessagePassing):
    def __init__(self):
        # We set aggr='min' so that the aggregator takes the minimum
        # over all neighbor messages, mimicking a fast-marching style update.
        super().__init__(aggr='min')

    def forward(self, tof, edge_index, pos, speed):
        """
        Args:
            tof (Tensor): Per-node time of flight, shape [num_nodes].
            edge_index (LongTensor): Graph edges of shape [2, num_edges].
            pos (Tensor): Node positions [num_nodes, 2] (for distance calculation).
            speed (Tensor): Per-node wave speed [num_nodes].

        Returns:
            updated_tof (Tensor): Updated time of flight per node.
        """
        # We call propagate(...), passing all relevant data.
        # Inside .propagate, the framework will call message(...), aggregate(...), and update(...).
        return self.propagate(edge_index, tof=tof, pos=pos, speed=speed)

    def message(self, tof_j, pos_i, pos_j, speed_j):
        """
        Here we define how a neighbor j's feature (tof_j) contributes to node i.
        `tof_j`, `pos_j`, `speed_j` come from the neighbor, while `pos_i` is the center node's position.
        The suffix '_j' indicates we are dealing with the neighbor's data.
        """
        # Compute Euclidean distance between node i and j
        dist_ij = (pos_i - pos_j).norm(dim=-1)  # shape [num_edges]

        #  “fast marching” style update:
        # T_i potential from neighbor j = T_j + distance / speed_j
        return tof_j + dist_ij / (speed_j + 1e-8)

    def update(self, aggr_out):
        """
        The aggregated output from neighbors is passed here.
        With aggr='min', aggr_out is the minimum of (tof_j + dist/speed_j) across neighbors j.
        """
        # For FMM, we can directly return the aggregated (min) value
        return aggr_out



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

        self.fmm_layer = FMMMessagePassing()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, in_channels, heads=1, concat=False)

    def forward(self, x, edge_index, pos):
        T_in = x[:, 0]
        c_in = x[:, 1]

        # 1) FMM pass to update TOF
        updated_T = self.fmm_layer(T_in, edge_index, pos, c_in)

        # 2) Recombine updated TOF with c_in to form new features
        x_fmm = torch.stack([updated_T, c_in], dim=-1)  # shape [num_nodes, 2]

        # 3) GAT layers to refine/learn from the updated features
        x_gat = self.gat1(x_fmm, edge_index).relu()
        x = self.gat2(x_gat, edge_index).softplus()

        # x_gat is shape [num_nodes, 2]: new [TOF, c] predictions
        return x
