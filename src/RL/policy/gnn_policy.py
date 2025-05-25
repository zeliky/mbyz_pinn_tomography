import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions import Categorical, Independent
from torch_geometric.data import Data


class GNNPolicy(nn.Module):
    """
    GNN‑based policy network that outputs, **for every mesh node**, a categorical
    distribution over a discrete set of candidate c(x,y) values:
        {0.1, 0.5, 1.2, 1.5, 2.5}

    *Forward* returns *(action_dist, value)* so it is compatible with the PPO
    agent in **RLAgent**.
    """

    # Discrete candidate SoS values (cm / µs or your chosen units)
    C_BINS = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]) # shape (5,)

    def __init__(
        self,
        num_sensor_nodes: int,  # S + R  (non‑trainable nodes)
        input_dim: int = 2,     # (T, c)
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_sensor_nodes = num_sensor_nodes

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Two‑layer GAT encoder
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

        # Policy head → logits per node (5 bins)
        self.policy_head = nn.Linear(hidden_dim, len(self.C_BINS))
        # Value head → global state‑value
        self.value_head = nn.Linear(hidden_dim, 1)

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------
    def forward(self, data: Data):
        """Return (action_distribution, value_estimate)."""
        x, edge_index = data.x, data.edge_index

        # Normalize input features
        x = self.input_norm(x)

        # GAT layers with gradient clipping
        h = F.relu(self.gat1(x, edge_index))
        h = torch.clamp(h, min=-10, max=10)  # Prevent extreme values
        h = F.relu(self.gat2(h, edge_index))
        h = torch.clamp(h, min=-10, max=10)  # Prevent extreme values

        # ----------------- Value -----------------
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)  # single graph
        graph_emb = global_mean_pool(h, batch)
        value = self.value_head(graph_emb).squeeze(-1)  # shape []

        # ----------------- Policy ----------------
        logits = self.policy_head(h)  # (N, 5)
        
        # Ensure logits are finite
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Mask out sensor nodes (sources + receivers) so the agent only changes mesh nodes
        if self.num_sensor_nodes > 0:
            sensor_mask = torch.arange(h.size(0), device=h.device) < self.num_sensor_nodes
            logits[sensor_mask] = -1e9  # effectively zero probability for non‑mesh nodes

        # Create an Independent Categorical over mesh nodes (action dim = num_mesh_nodes)
        dist = Independent(Categorical(logits=logits), 1)  # event_shape = (N,)

        return dist, value

    # -------------------------------------------------------------
    # Helper to map sampled indices → actual c values  (optional)
    # -------------------------------------------------------------
    @staticmethod
    def indices_to_c(indices: torch.Tensor) -> torch.Tensor:
        """Convert sampled bin indices → actual speed values."""
        return GNNPolicy.C_BINS.to(indices.device)[indices]

