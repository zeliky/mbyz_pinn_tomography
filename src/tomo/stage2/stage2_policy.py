"""Stage 2 GNN policy: outputs small bounded SoS corrections only on ROI nodes."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class Stage2Policy(nn.Module):
    """GNN policy that outputs delta_update only for ROI nodes.

    Action head outputs scalar per ROI node, tanh-scaled to [-max_delta, +max_delta].
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_delta: float = 0.02,
        action_std_init: float = 0.1,
    ):
        super().__init__()
        self.max_delta = max_delta
        total_hidden = hidden_channels * num_heads

        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)

        self.action_head = GATConv(total_hidden, 1, heads=1, concat=False, dropout=dropout)
        self.value_head = GATConv(total_hidden, 1, heads=1, concat=False, dropout=dropout)

        self.action_log_std = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(action_std_init)))

    def forward(self, data: Data):
        """Forward pass. Returns (action_dist, value).

        action_dist: Normal over ROI node actions only.
        value: scalar state value.
        """
        x = data.x
        edge_index = data.edge_index

        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))

        action_mean = self.action_head(x, edge_index).squeeze(-1)

        roi_start = data.roi_node_start
        roi_end = data.roi_node_end
        roi_action_mean = action_mean[roi_start:roi_end]

        roi_action_mean = torch.tanh(roi_action_mean) * self.max_delta

        action_std = torch.exp(self.action_log_std).expand_as(roi_action_mean)
        action_dist = Normal(roi_action_mean, action_std)

        value_features = self.value_head(x, edge_index).squeeze(-1)
        value = global_mean_pool(value_features.unsqueeze(-1), batch=None).squeeze()

        return action_dist, value
