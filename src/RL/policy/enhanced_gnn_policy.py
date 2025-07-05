import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions import Normal, Independent
from torch_geometric.data import Data


class EnhancedGNNPolicy(nn.Module):
    """
    Enhanced GNN-based policy with deeper architecture for better wave propagation modeling.
    
    Features:
    - Deeper network (4-6 layers) for multi-hop message passing
    - More attention heads for complex wave patterns
    - Residual connections for training stability
    - Layer normalization
    - Enhanced feature processing
    """

    def __init__(
            self, 
            num_sensor_nodes,
            in_channels: int = 8,  # Enhanced features
            hidden_channels: int = 24,  # Increased capacity
            num_heads: int = 12,  # More attention heads
            num_layers: int = 4,  # Deeper network
            dropout: float = 0.1,
            action_std_init: float = 0.1,
            use_residual: bool = True,
            use_layer_norm: bool = True,
    ):
        super().__init__()
        self.num_sensor_nodes = num_sensor_nodes
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        total_hidden = hidden_channels * num_heads
        
        # Multi-layer GAT architecture
        self.gat_layers = nn.ModuleList()
        
        # First layer: input -> hidden
        self.gat_layers.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, 
                   concat=True, dropout=dropout)
        )
        
        # Intermediate layers: hidden -> hidden
        for i in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(total_hidden, hidden_channels, heads=num_heads,
                       concat=True, dropout=dropout)
            )
        
        # Layer normalization for training stability
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(total_hidden) for _ in range(num_layers)
            ])
        
        # Enhanced action head with intermediate layer
        self.action_head = nn.Sequential(
            nn.Linear(total_hidden, total_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_hidden // 2, 1)
        )
        
        # Enhanced value head with intermediate layer
        self.value_head = nn.Sequential(
            nn.Linear(total_hidden, total_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_hidden // 2, 1)
        )
        
        # Learnable action standard deviation
        self.action_log_std = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(action_std_init)))

    def forward(self, data):
        """
        Enhanced forward pass with residual connections and layer normalization.
        
        Args:
            data: PyG Data object with node features and edge indices
            
        Returns:
            action_dist: Normal distribution over actions
            value: State value estimate (scalar)
        """
        x, edge_index = data.x, data.edge_index
        
        # Multi-layer feature extraction with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            x_new = gat_layer(x, edge_index)
            
            # Apply activation
            x_new = torch.relu(x_new)
            
            # Residual connection (skip first layer)
            if self.use_residual and i > 0 and x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new
            
            # Layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
        
        # Action head - compute action means for mesh nodes only
        action_logits = self.action_head(x).squeeze(-1)  # [N]
        
        # Only apply actions to mesh nodes (skip sensor nodes)
        mesh_start = self.num_sensor_nodes
        mesh_action_mean = action_logits[mesh_start:]  # Only mesh nodes get actions
        
        # Action standard deviation (learnable parameter)
        action_std = torch.exp(self.action_log_std).expand_as(mesh_action_mean)
        
        # Create action distribution (Normal distribution for continuous actions)
        action_dist = Normal(mesh_action_mean, action_std)
        
        # Value head - global state value estimate
        value_logits = self.value_head(x).squeeze(-1)  # [N]
        value = global_mean_pool(value_logits, batch=None).squeeze()  # Global average -> scalar
        
        return action_dist, value

    def get_model_info(self):
        """Return information about the model architecture and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_layers': self.num_layers,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm,
        }


class LightEnhancedGNNPolicy(EnhancedGNNPolicy):
    """
    Light version of enhanced GNN policy (~100K parameters).
    Good for initial experiments and faster training.
    """
    
    def __init__(self, num_sensor_nodes, **kwargs):
        # Override defaults for light version
        defaults = {
            'in_channels': 8,
            'hidden_channels': 20,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'action_std_init': 0.1,
            'use_residual': True,
            'use_layer_norm': True,
        }
        defaults.update(kwargs)
        super().__init__(num_sensor_nodes, **defaults)


class HeavyEnhancedGNNPolicy(EnhancedGNNPolicy):
    """
    Heavy version of enhanced GNN policy (~1.6M parameters).
    For complex problems requiring maximum model capacity.
    """
    
    def __init__(self, num_sensor_nodes, **kwargs):
        # Override defaults for heavy version
        defaults = {
            'in_channels': 10,
            'hidden_channels': 32,
            'num_heads': 16,
            'num_layers': 6,
            'dropout': 0.1,
            'action_std_init': 0.1,
            'use_residual': True,
            'use_layer_norm': True,
        }
        defaults.update(kwargs)
        super().__init__(num_sensor_nodes, **defaults)


def create_enhanced_gnn_policy(num_sensor_nodes, model_size='medium', **kwargs):
    """
    Factory function to create enhanced GNN policies of different sizes.
    
    Args:
        num_sensor_nodes: Number of sensor nodes in the graph
        model_size: 'light' (~100K params), 'medium' (~400K params), 'heavy' (~1.6M params)
        **kwargs: Additional arguments to override defaults
    
    Returns:
        Enhanced GNN policy instance
    """
    if model_size == 'light':
        return LightEnhancedGNNPolicy(num_sensor_nodes, **kwargs)
    elif model_size == 'medium':
        return EnhancedGNNPolicy(num_sensor_nodes, **kwargs)
    elif model_size == 'heavy':
        return HeavyEnhancedGNNPolicy(num_sensor_nodes, **kwargs)
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from 'light', 'medium', 'heavy'")


# Physics-informed feature enhancement functions
def compute_physics_features(positions, sources_positions, receivers_positions, c_map=None):
    """
    Compute physics-informed features for enhanced node representations.
    
    Args:
        positions: Node positions (N, 2)
        sources_positions: Source positions (S, 2)  
        receivers_positions: Receiver positions (R, 2)
        c_map: Current speed of sound map (optional)
    
    Returns:
        physics_features: Enhanced features tensor (N, num_features)
    """
    device = positions.device
    N = positions.shape[0]
    
    # Convert to tensors if needed
    if not isinstance(sources_positions, torch.Tensor):
        sources_positions = torch.tensor(sources_positions, device=device, dtype=torch.float32)
    if not isinstance(receivers_positions, torch.Tensor):
        receivers_positions = torch.tensor(receivers_positions, device=device, dtype=torch.float32)
    
    features = []
    
    # 1. Distance to nearest source
    dist_to_sources = torch.cdist(positions, sources_positions)  # (N, S)
    min_dist_to_source = dist_to_sources.min(dim=1)[0]  # (N,)
    features.append(min_dist_to_source.unsqueeze(-1))
    
    # 2. Distance to nearest receiver
    dist_to_receivers = torch.cdist(positions, receivers_positions)  # (N, R)
    min_dist_to_receiver = dist_to_receivers.min(dim=1)[0]  # (N,)
    features.append(min_dist_to_receiver.unsqueeze(-1))
    
    # 3. Average distance to all sources (global context)
    avg_dist_to_sources = dist_to_sources.mean(dim=1)  # (N,)
    features.append(avg_dist_to_sources.unsqueeze(-1))
    
    # 4. Average distance to all receivers (global context)
    avg_dist_to_receivers = dist_to_receivers.mean(dim=1)  # (N,)
    features.append(avg_dist_to_receivers.unsqueeze(-1))
    
    # Normalize distances by domain size (assuming 128x128 domain)
    domain_size = 128.0
    for i in range(len(features)):
        features[i] = features[i] / domain_size
    
    # Stack all features
    physics_features = torch.cat(features, dim=-1)  # (N, 4)
    
    return physics_features


if __name__ == "__main__":
    # Test the enhanced policies
    print("Testing Enhanced GNN Policies...")
    
    # Test parameters
    num_sensor_nodes = 64
    num_mesh_nodes = 1000
    total_nodes = num_sensor_nodes + num_mesh_nodes
    
    # Create dummy data
    x = torch.randn(total_nodes, 8)  # 8 input features
    edge_index = torch.randint(0, total_nodes, (2, 5000))  # Random edges
    data = Data(x=x, edge_index=edge_index)
    
    # Test different model sizes
    for size in ['light', 'medium', 'heavy']:
        print(f"\n=== Testing {size.upper()} model ===")
        
        policy = create_enhanced_gnn_policy(num_sensor_nodes, model_size=size)
        info = policy.get_model_info()
        
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Layers: {info['num_layers']}")
        
        # Forward pass
        action_dist, value = policy(data)
        print(f"Action distribution shape: {action_dist.mean.shape}")
        print(f"Value shape: {value.shape}")
        
        # Memory usage estimate
        param_memory = info['total_parameters'] * 4 / (1024**2)  # MB
        print(f"Estimated parameter memory: {param_memory:.1f} MB")
    
    print("\n✅ All enhanced policies tested successfully!")
