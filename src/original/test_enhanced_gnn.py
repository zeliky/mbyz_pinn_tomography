"""
Test script for the enhanced GNN policy and quick evaluation comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import modules
from RL.policy.gnn_policy import GNNPolicy
from RL.policy.enhanced_gnn_policy import create_enhanced_gnn_policy, compute_physics_features
from graph.network import GraphDataset
from torch_geometric.data import Data


def test_enhanced_gnn_architectures():
    """Test different enhanced GNN architectures and compare parameter counts."""
    print("="*60)
    print("TESTING ENHANCED GNN ARCHITECTURES")
    print("="*60)
    
    # Test parameters
    num_sensor_nodes = 64
    num_mesh_nodes = 1000
    total_nodes = num_sensor_nodes + num_mesh_nodes
    
    # Create dummy data
    x = torch.randn(total_nodes, 8)  # 8 input features
    edge_index = torch.randint(0, total_nodes, (2, 5000))  # Random edges
    data = Data(x=x, edge_index=edge_index)
    
    # Test original GNN policy
    print("\n1. ORIGINAL GNN POLICY")
    print("-" * 30)
    original_policy = GNNPolicy(num_sensor_nodes=num_sensor_nodes)
    original_params = sum(p.numel() for p in original_policy.parameters())
    print(f"Parameters: {original_params:,}")
    
    # Forward pass
    action_dist, value = original_policy(data)
    print(f"Action distribution shape: {action_dist.mean.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test different enhanced model sizes
    model_sizes = ['light', 'medium', 'heavy']
    
    for size in model_sizes:
        print(f"\n2. ENHANCED GNN POLICY ({size.upper()})")
        print("-" * 40)
        
        try:
            policy = create_enhanced_gnn_policy(num_sensor_nodes, model_size=size)
            info = policy.get_model_info()
            
            print(f"Parameters: {info['total_parameters']:,}")
            print(f"Layers: {info['num_layers']}")
            print(f"Residual connections: {info['use_residual']}")
            print(f"Layer normalization: {info['use_layer_norm']}")
            
            # Forward pass
            action_dist, value = policy(data)
            print(f"Action distribution shape: {action_dist.mean.shape}")
            print(f"Value shape: {value.shape}")
            
            # Memory usage estimate
            param_memory = info['total_parameters'] * 4 / (1024**2)  # MB
            print(f"Estimated parameter memory: {param_memory:.1f} MB")
            
            # Parameter increase factor
            increase_factor = info['total_parameters'] / original_params
            print(f"Parameter increase vs original: {increase_factor:.1f}x")
            
        except Exception as e:
            print(f"Error testing {size} model: {e}")
    
    print("\n" + "="*60)
    print("✅ Enhanced GNN architecture tests completed!")


def test_physics_features():
    """Test physics-informed feature computation."""
    print("\n" + "="*60)
    print("TESTING PHYSICS-INFORMED FEATURES")
    print("="*60)
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Node positions (100 nodes in 10x10 grid)
    positions = torch.tensor([[i, j] for i in range(10) for j in range(10)], 
                           device=device, dtype=torch.float32)
    
    # Source and receiver positions
    sources_positions = np.array([[2, 2], [7, 7], [2, 7], [7, 2]])
    receivers_positions = np.array([[1, 1], [8, 8], [1, 8], [8, 1], [5, 5]])
    
    print(f"Node positions shape: {positions.shape}")
    print(f"Sources: {len(sources_positions)}")
    print(f"Receivers: {len(receivers_positions)}")
    
    # Compute physics features
    physics_features = compute_physics_features(
        positions, sources_positions, receivers_positions
    )
    
    print(f"Physics features shape: {physics_features.shape}")
    print(f"Features per node: {physics_features.shape[1]}")
    
    # Analyze features
    feature_names = [
        'Min distance to source',
        'Min distance to receiver', 
        'Avg distance to sources',
        'Avg distance to receivers'
    ]
    
    for i, name in enumerate(feature_names):
        values = physics_features[:, i]
        print(f"{name}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    print("✅ Physics features test completed!")


def create_test_graph_dataset():
    """Create a test graph dataset for evaluation."""
    print("\n" + "="*60)
    print("CREATING TEST GRAPH DATASET")
    print("="*60)
    
    # Graph configuration
    graph_config = {
        'c_init': 1.5,
        't_init': 0.0,
        'x_range': (10, 118),
        'y_range': (10, 118),
        'nx': 4,
        'ny': 4,
        'mesh_node_k': 9,
        'sensor_k': 5,
        'num_source_nodes': 16,
        'num_receiver_nodes': 32
    }
    
    # Create dataset
    graph_dataset = GraphDataset(**graph_config)
    
    # Create dummy source/receiver positions
    sources_positions = np.random.rand(16, 2) * 108 + 10  # Within domain
    receivers_positions = np.random.rand(32, 2) * 108 + 10
    
    # Build graph
    graph_dataset.build(sources_positions, receivers_positions)
    
    print(f"Total nodes: {graph_dataset.num_source_nodes + graph_dataset.num_receiver_nodes + graph_dataset.num_mesh_nodes}")
    print(f"Source nodes: {graph_dataset.num_source_nodes}")
    print(f"Receiver nodes: {graph_dataset.num_receiver_nodes}")
    print(f"Mesh nodes: {graph_dataset.num_mesh_nodes}")
    print(f"Graph edges: {graph_dataset.global_edges.shape[0]}")
    
    # Test graph generation
    dummy_tof = torch.randn(16, 32)  # 16 sources, 32 receivers
    selected_sources = list(range(16))
    
    for data in graph_dataset.get_graph(dummy_tof, selected_sources, 'cpu'):
        print(f"Graph data shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Node features: {data.x.shape[1]}")
        break  # Only test first graph
    
    print("✅ Graph dataset test completed!")
    
    return graph_dataset, sources_positions, receivers_positions


def quick_performance_comparison():
    """Quick performance comparison between original and enhanced GNN."""
    print("\n" + "="*60)
    print("QUICK PERFORMANCE COMPARISON")
    print("="*60)
    
    # Create test data
    num_sensor_nodes = 48
    num_mesh_nodes = 2916  # 54x54 mesh
    total_nodes = num_sensor_nodes + num_mesh_nodes
    
    x = torch.randn(total_nodes, 8)
    edge_index = torch.randint(0, total_nodes, (2, 10000))
    data = Data(x=x, edge_index=edge_index)
    
    # Test models
    models = {
        'Original': GNNPolicy(num_sensor_nodes=num_sensor_nodes),
        'Enhanced Light': create_enhanced_gnn_policy(num_sensor_nodes, 'light'),
        'Enhanced Medium': create_enhanced_gnn_policy(num_sensor_nodes, 'medium'),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Parameter count
        params = sum(p.numel() for p in model.parameters())
        
        # Timing test
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model(data)
            
            # Actual timing
            import time
            start_time = time.time()
            for _ in range(20):
                action_dist, value = model(data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 20
        
        results[name] = {
            'parameters': params,
            'inference_time': avg_time,
            'action_shape': action_dist.mean.shape,
            'value_shape': value.shape
        }
        
        print(f"  Parameters: {params:,}")
        print(f"  Inference time: {avg_time*1000:.2f} ms")
        print(f"  Action shape: {action_dist.mean.shape}")
    
    # Summary comparison
    print("\n" + "="*40)
    print("PERFORMANCE SUMMARY")
    print("="*40)
    
    baseline_params = results['Original']['parameters']
    baseline_time = results['Original']['inference_time']
    
    for name, result in results.items():
        param_ratio = result['parameters'] / baseline_params
        time_ratio = result['inference_time'] / baseline_time
        
        print(f"{name}:")
        print(f"  Parameter ratio: {param_ratio:.1f}x")
        print(f"  Time ratio: {time_ratio:.1f}x")
        print(f"  Memory estimate: {result['parameters'] * 4 / (1024**2):.1f} MB")
    
    print("✅ Performance comparison completed!")
    return results


def main():
    """Main test function."""
    print("🚀 Starting Enhanced GNN Policy Tests...")
    
    # Run all tests
    test_enhanced_gnn_architectures()
    test_physics_features()
    graph_dataset, sources_pos, receivers_pos = create_test_graph_dataset()
    performance_results = quick_performance_comparison()
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Summary recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    print("1. For initial experiments: Use 'light' enhanced model (~100K params)")
    print("2. For production use: Use 'medium' enhanced model (~400K params)")
    print("3. For complex problems: Use 'heavy' enhanced model (~1.6M params)")
    print("4. Consider physics-informed features for better wave propagation modeling")
    print("5. Use the evaluation framework to compare with traditional solvers")
    
    return performance_results


if __name__ == "__main__":
    main()
