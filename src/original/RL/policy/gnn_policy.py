import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions import Normal, Independent
from torch_geometric.data import Data


class GNNPolicy(nn.Module):
    """
    GNN-based policy that uses GAT layers to process the graph.
    Returns (action_distribution, value) for PPO.
    """

    def __init__(
            self, num_sensor_nodes,
            in_channels: int = 6,
            hidden_channels: int = 16,
            num_heads: int = 8,
            dropout: float = 0.1,
            action_std_init: float = 0.1,
    ):
        super().__init__()
        self.num_sensor_nodes = num_sensor_nodes

        total_hidden = hidden_channels * num_heads
        
        # Shared GAT layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        
        # Separate heads for policy and value
        self.action_head = GATConv(total_hidden, out_channels=1, heads=1, concat=False, dropout=dropout)
        self.value_head = GATConv(total_hidden, out_channels=1, heads=1, concat=False, dropout=dropout)
        
        # Learnable action standard deviation
        self.action_log_std = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(action_std_init)))

    # ------------------------------------------------------------------
    def forward(self, data):
        """
        Forward pass that returns (action_distribution, value) for PPO.
        
        Args:
            data: PyG Data object with node features and edge indices
            
        Returns:
            action_dist: Normal distribution over actions
            value: State value estimate (scalar)
        """
        x, edge_index = data.x, data.edge_index
        
        # Shared feature extraction through GAT layers
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        
        # Action head - compute action means for each node
        action_mean = self.action_head(x, edge_index).squeeze(-1)  # [N]
        
        # Only apply actions to mesh nodes (skip sensor nodes)
        mesh_start = self.num_sensor_nodes
        mesh_action_mean = action_mean[mesh_start:]  # Only mesh nodes get actions
        
        # Action standard deviation (learnable parameter)
        action_std = torch.exp(self.action_log_std).expand_as(mesh_action_mean)
        
        # Create action distribution (Normal distribution for continuous actions)
        action_dist = Normal(mesh_action_mean, action_std)
        
        # Value head - global state value estimate
        value_features = self.value_head(x, edge_index).squeeze(-1)  # [N]
        value = global_mean_pool(value_features, batch=None).squeeze()  # Global average -> scalar
        
        return action_dist, value
