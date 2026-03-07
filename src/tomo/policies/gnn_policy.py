"""GNN policy: learned delta_c for mesh nodes. Port from original RL GNN policy."""

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from tomo.policies.base import Policy
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class GNNPolicy(Policy):
    """GAT-based policy that outputs delta_c for mesh nodes. Sensor nodes get zero delta."""

    def __init__(
        self,
        num_sensor_nodes: int,
        in_channels: int = 6,
        hidden_channels: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        delta_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_sensor_nodes = num_sensor_nodes
        self.delta_scale = delta_scale
        total_hidden = hidden_channels * num_heads
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(total_hidden, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.delta_head = GATConv(total_hidden, 1, heads=1, concat=False, dropout=dropout)

    def _node_features(self, state: SoSState, observation: Observation, tof_error: torch.Tensor, graph: Data) -> torch.Tensor:
        """Build node feature matrix: [c, tof_error, ...] and existing graph.x if any."""
        x = graph.x
        if x is None or x.shape[1] < 2:
            num_nodes = state.c_values.shape[0]
            device = state.c_values.device
            c = state.c_values
            err = tof_error.to(device)
            if err.numel() != num_nodes:
                err = err.expand(num_nodes) if err.numel() == 1 else torch.zeros(num_nodes, device=device)
            x = torch.stack([c, err], dim=-1)
            if num_nodes > 2:
                pad = torch.zeros(num_nodes, 4, device=device)
                x = torch.cat([x, pad], dim=-1)
        return x

    def __call__(
        self,
        state: SoSState,
        observation: Observation,
        tof_error: torch.Tensor,
        observation_graph: Data,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        x = self._node_features(state, observation, tof_error, observation_graph)
        if x.shape[1] < 6:
            pad = torch.zeros(x.shape[0], 6 - x.shape[1], device=x.device)
            x = torch.cat([x, pad], dim=-1)
        edge_index = observation_graph.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        delta_all = self.delta_head(x, edge_index).squeeze(-1)
        # Zero delta for sensor nodes; only mesh nodes get updates
        delta = torch.zeros_like(state.c_values, device=state.c_values.device)
        mesh_start = self.num_sensor_nodes
        if mesh_start < delta.shape[0]:
            delta[mesh_start:] = delta_all[mesh_start:] * self.delta_scale
        return delta
