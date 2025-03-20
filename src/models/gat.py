import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv, MessagePassing
from scipy.interpolate import griddata
import numpy as np





class FMMMessagePassing(MessagePassing):
    def __init__(self, sos_values):
        # We set aggr='min' so that the aggregator takes the minimum
        # over all neighbor messages, mimicking a fast-marching style update.
        super().__init__(aggr='min', node_dim=0)

        # Store speed of sound as shape [num_nodes]
        self.sos_values = nn.Parameter(sos_values)
        self.tof_values = None
        self.min_sos = 1.2
        self.max_sos = 2.5

    def init_tof_values(self, tof):
        self.tof_values = tof
    def forward(self, edge_index, pos):
        """
        Args:
            edge_index (LongTensor): Graph edges of shape [2, num_edges].
            pos (Tensor): Node positions [num_nodes, 2] (for distance calculation).
            speed (Tensor): Per-node wave speed [num_nodes].

        Returns:
            updated_tof (Tensor): Updated time of flight per node.
        """

        # We call propagate(...), passing all relevant data.
        # Inside .propagate, the framework will call message(...), aggregate(...), and update(...).

        speed = self.min_sos  + (self.max_sos - self.min_sos) * torch.sigmoid(self.sos_values)
        return self.propagate(edge_index, tof=self.tof_values, pos=pos, speed=speed)

    def message(self, tof_j, pos_i, pos_j, speed_j):
        """
        Here we define how a neighbor j's feature (tof_j) contributes to node i.
        `tof_j`, `pos_j`, `speed_j` come from the neighbor, while `pos_i` is the center node's position.
        The suffix '_j' indicates we are dealing with the neighbor's data.
        """
        # Compute Euclidean distance between node i and j
        dist_ij = (pos_j - pos_i).norm(dim=-1).float()  # shape [num_edges]
        #  “fast marching” style update:
        # T_i potential from neighbor j = T_j + distance / speed_j
        update = tof_j + dist_ij / (speed_j + 1e-8)
        return update

    def update(self, aggr_out):
        """
        The aggregated output from neighbors is passed here.
        With aggr='min', aggr_out is the minimum of (tof_j + dist/speed_j) across neighbors j.
        """
        # For FMM, we can directly return the aggregated (min) value
        return aggr_out



class DualHeadGATModel(nn.Module):
    def __init__(self,  in_channels=2, hidden_channels=64, out_channels=2,
                  heads=8):
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
        # Store wave speed as node-wise trainable parameter
        # Initialize to 0.15 cm/us for all nodes, for example

        # Simple GAT stack
        #  - first GAT: from 2 -> hidden_channels * heads
        #  - second GAT: from hidden_channels * heads -> 2 (i.e., [T, c])
        self.gat1 = GATConv(in_channels=in_channels, out_channels=hidden_channels,
                            heads=heads, concat=True)
        self.gat2 = GATConv(in_channels=hidden_channels * heads,
                            out_channels=out_channels, heads=1, concat=False)

        self.relu = nn.ReLU()



    def forward(self, x, edge_index, pos):

        # 1) Separate T_in from x, and override c_in with our trainable parameter
        T_in = x[:, 0]  # shape (num_nodes,)
        c_in = self.c_values.to(x.device)  # shape (num_nodes,)

        receiver_indices = torch.arange(32, 64, device=x.device)

        for _ in range(self.fmm_iterations):
            self.message_passing(edge_index, pos)
            updated_T = self.message_passing.tof_values
            print(f"pred: {updated_T[receiver_indices]}")
            print(f"true: {T_in[receiver_indices]}")




        # 3) Combine updated T with the current learned c for GAT input
        x_fmm = torch.stack([updated_T, c_in], dim=-1)  # shape (num_nodes, 2)

        # 4) Pass through GAT layers
        #    Here, we do 2 layers (customize num_gat_layers if you like)
        x_gat = self.gat1(x_fmm, edge_index)
        x_gat = self.relu(x_gat)
        x_gat = self.gat2(x_gat, edge_index)
        x_gat[:, 1] = F.softplus(x_gat[:, 1])  # ensure positivity for c

        return x_gat  # shape (num_nodes, 2) => [T, c]


class SosEstimator:
    def __init__(self, num_nodes, init_value=0.15, fmm_iterations=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.init_value = init_value
        self.fmm_iterations = fmm_iterations
        self.c_values = None
        self.message_passing = None
        self.optimizer = None
        self.positions = None
        self.criterion = nn.MSELoss()
        self.lr = 1

    def reset(self, device):
        initial_c_values = torch.full((self.num_nodes,), self.init_value, requires_grad=True).to(device)
        self.message_passing = FMMMessagePassing(initial_c_values)


    def estimate(self, x, edge_index, pos,transmitters_indices, receiver_indices):
        self.positions = pos
        T_in = x[:, 0]  # shape (num_nodes,)
        self.message_passing.init_tof_values(T_in)
        self.optimizer = optim.Adam(self.message_passing.parameters(), lr=self.lr)
        for _ in range(self.fmm_iterations):
            self.optimizer.zero_grad()
            updated_T = self.message_passing(edge_index, pos)
            #print(f"pred: {updated_T}")
            #print(f"true: {T_in[receiver_indices]}")
            loss = self.criterion(updated_T[receiver_indices], T_in[receiver_indices]) + torch.sum(updated_T[transmitters_indices])
            #loss =  torch.abs(torch.sum(updated_T[transmitters_indices]))
            #print(updated_T[transmitters_indices])
            loss.backward()
            self.optimizer.step()

        T = updated_T
        c = self.message_passing.sos_values
        return T, c

    def get_sos(self, nx, ny):
        """
        Returns a 2D array (ny, nx) containing the SoS values on a discrete grid.

        1) Places SoS at integer-rounded node positions (if they fit in [0..nx-1, 0..ny-1]).
        2) Interpolates missing cells using linear interpolation from the scattered node positions.
        """

        # 1) Create an empty grid (ny rows, nx cols), fill with NaNs
        sos_grid = np.full((ny, nx), np.nan, dtype=float)

        c_values = self.message_passing.sos_values
        # 2) Place known SoS values at integer-rounded positions
        for i, (x_i, y_i) in enumerate(self.positions):
            gx = int(torch.round(x_i))
            gy = int(torch.round(y_i))
            # Ensure it's within grid bounds
            if 0 <= gx < nx and 0 <= gy < ny:
                sos_grid[gy, gx] = c_values[i]
        #return sos_grid



        # 3) Interpolate missing cells
        #    (A) Build a list of known points (grid_x, grid_y) and their SoS
        known_mask = ~np.isnan(sos_grid)
        known_y, known_x = np.where(known_mask)
        known_sos = sos_grid[known_mask]

        #    (B) Build a coordinate mesh for the entire grid
        grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))

        #    (C) Use griddata on the integer grid
        points = np.column_stack([known_x, known_y])  # shape (K, 2)
        sos_filled = griddata(points, known_sos,
                              (grid_x, grid_y),
                              method='linear')

        # 4) Fill any remaining NaNs via nearest-neighbor
        #    (linear interpolation can leave NaNs at boundaries)
        nan_mask = np.isnan(sos_filled)
        if np.any(nan_mask):
            sos_filled[nan_mask] = griddata(points, known_sos,
                                            (grid_x[nan_mask], grid_y[nan_mask]),
                                            method='nearest')

        return sos_filled


