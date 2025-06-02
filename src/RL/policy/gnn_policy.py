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

    # Discrete candidate SoS values (cm / µs or your chosen units)
    C_BINS = torch.tensor([0.1, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])

    def __init__(
        self,
        num_sensor_nodes: int,
        in_channels: int = 2,  # ToF and distance
        hidden_channels: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        self.num_sensor_nodes = num_sensor_nodes
        self.use_skip_connections = use_skip_connections
        
        # Input projection to ensure consistent dimensions
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers with consistent hidden dimensions
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_channels if i == 0 else hidden_channels * num_heads,
                out_channels=hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True if i < num_layers - 1 else False  # Only last layer doesn't concatenate heads
            )
            for i in range(num_layers)
        ])
        
        # Layer normalization for each GAT layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels * num_heads if i < num_layers - 1 else hidden_channels)
            for i in range(num_layers)
        ])
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, len(self.C_BINS))  # Output logits for each SoS bin
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # Output value estimate
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use orthogonal initialization with smaller gain
            nn.init.orthogonal_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, in_channels]
                - edge_index: Graph connectivity [2, num_edges]
                - pos: Node positions [num_nodes, 2]
        
        Returns:
            action_dist: Distribution over actions for all mesh nodes
            value: State value estimate
        """
        # Project input features to hidden dimension
        h = self.input_proj(data.x)        
        
        skip_connections = []
        
        # Process through GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            # Store skip connection before GAT
            if self.use_skip_connections and i > 0:
                skip_connections.append(h)
            
            # Apply GAT
            h_new = gat(h, data.edge_index)                        
            h_new = F.relu(h_new)
            
            # Add skip connection if available
            if self.use_skip_connections and i > 0:
                # Ensure dimensions match by projecting skip connection if needed
                if h_new.size(-1) != skip_connections[i-1].size(-1):
                    skip_connections[i-1] = nn.Linear(
                        skip_connections[i-1].size(-1), 
                        h_new.size(-1)
                    ).to(h_new.device)(skip_connections[i-1])
                h_new = h_new + skip_connections[i-1]
            
            # Clip values to prevent explosion
            h_new = torch.clamp(h_new, min=-10, max=10)
            h = h_new
        
        # Get value estimate using global mean pooling
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)  # single graph
        graph_emb = global_mean_pool(h, batch)
        
        value = self.value_head(graph_emb).squeeze(-1)
        # Get action logits for all nodes
        logits = self.policy_head(h)  # (N, num_bins)
        
        # Mask out sensor nodes (sources + receivers) so the agent only changes mesh nodes
        if self.num_sensor_nodes > 0:
            sensor_mask = torch.arange(h.size(0), device=h.device) < self.num_sensor_nodes
            logits[sensor_mask] = -1e9  # effectively zero probability for non-mesh nodes
        
        # Create an Independent Categorical over mesh nodes
        dist = Independent(Categorical(logits=logits), 1)  # event_shape = (N,)
        
        return dist, value

    # -------------------------------------------------------------
    # Helper to map sampled indices → actual c values  (optional)
    # -------------------------------------------------------------
    @staticmethod
    def indices_to_c(indices: torch.Tensor) -> torch.Tensor:
        """Convert sampled bin indices → actual speed values."""
        return GNNPolicy.C_BINS.to(indices.device)[indices]

