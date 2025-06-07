import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions import Categorical, Independent
from torch_geometric.data import Data


class GNNPolicy(nn.Module):
    """
    GNN-based policy that uses GAT layers to process the graph.
    """

    def __init__(
            self, num_sensor_nodes,
            in_channels: int = 4,
            hidden_channels: int = 16,
            num_heads: int = 8,
            max_change_per_step: float = 0.5,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensor_nodes=num_sensor_nodes

        total_hidden = hidden_channels * num_heads
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat4 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)

        self.out = GATConv(total_hidden, out_channels=1, heads=1, concat=False, dropout=dropout)

        self.max_change_per_step = max_change_per_step

    # ------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))
        x = F.relu(self.gat4(x, edge_index))

        delta_c = self.out(x, edge_index).squeeze(-1)  # [N]
        # Scale to the physical range (–max_change … +max_change)
        delta_c = torch.tanh(delta_c) * self.max_change_per_step
        return delta_c

    # -------------------------------------------------------------
    # Helper to map sampled indices → actual c values  (optional)
    # -------------------------------------------------------------
    @staticmethod
    def indices_to_c(indices: torch.Tensor) -> torch.Tensor:
        """Convert sampled bin indices → actual speed values."""
        return GNNPolicy.C_BINS.to(indices.device)[indices]

