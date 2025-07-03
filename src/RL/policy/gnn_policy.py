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
            in_channels: int = 6,
            hidden_channels: int = 16,
            num_heads: int = 8,

            dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensor_nodes=num_sensor_nodes

        total_hidden = hidden_channels * num_heads
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        #self.gat3 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        #self.gat4 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)

        self.out = GATConv(total_hidden, out_channels=1, heads=1, concat=False, dropout=dropout)

    # ------------------------------------------------------------------
    def forward(self, data):
        c_base = data.x[:, 0].clone()

        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
       
        #x = F.relu(self.gat3(x, edge_index))
        #x = F.relu(self.gat4(x, edge_index))

        delta_c = self.out(x, edge_index).squeeze()
             
        # delta_c = torch.tanh(delta_c)      # [N]
        c_pred = torch.softplus(c_base + delta_c)  # make sure that after change C will be positive
        return c_pred-c_base


    # -------------------------------------------------------------
    # Helper to map sampled indices → actual c values  (optional)
    # -------------------------------------------------------------
    @staticmethod
    def indices_to_c(indices: torch.Tensor) -> torch.Tensor:
        """Convert sampled bin indices → actual speed values."""
        return GNNPolicy.C_BINS.to(indices.device)[indices]

