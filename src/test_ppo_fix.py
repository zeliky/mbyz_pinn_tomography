#!/usr/bin/env python3
"""
Test script to verify the fixed PPO implementation.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from RL.policy.gnn_policy import GNNPolicy
from RL.agent import RLAgent

def create_dummy_graph_data(num_nodes=100, num_sensor_nodes=64):
    """Create a dummy PyG Data object for testing."""
    # Create dummy node features (6 features per node)
    x = torch.randn(num_nodes, 6)
    
    # Create dummy edge indices (fully connected for simplicity)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))
    
    # Create dummy positions
    pos = torch.randn(num_nodes, 2)
    
    return Data(x=x, edge_index=edge_index, pos=pos)

def test_policy_interface():
    """Test that the GNNPolicy returns the correct format."""
    print("Testing GNNPolicy interface...")
    
    num_sensor_nodes = 64
    policy = GNNPolicy(num_sensor_nodes=num_sensor_nodes)
    dummy_data = create_dummy_graph_data(num_nodes=100, num_sensor_nodes=num_sensor_nodes)
    
    # Test forward pass
    action_dist, value = policy(dummy_data)
    
    # Check action distribution
    assert hasattr(action_dist, 'sample'), "action_dist should have sample method"
    assert hasattr(action_dist, 'log_prob'), "action_dist should have log_prob method"
    assert hasattr(action_dist, 'entropy'), "action_dist should have entropy method"
    
    # Check value
    assert value.shape == torch.Size([]), f"Value should be scalar, got shape {value.shape}"
    
    # Test sampling
    action = action_dist.sample()
    logprob = action_dist.log_prob(action)
    entropy = action_dist.entropy()
    
    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Log prob shape: {logprob.shape}")
    print(f"✓ Entropy shape: {entropy.shape}")
    print(f"✓ Value shape: {value.shape}")
    print("✓ Policy interface test passed!")
    return True

def test_agent_buffer():
    """Test that the RLAgent properly stores buffer data."""
    print("\nTesting RLAgent buffer...")
    
    # Create dummy environment class
    class DummyEnv:
        def reset(self):
            return create_dummy_graph_data()
        
        def step(self, action):
            next_obs = create_dummy_graph_data()
            reward = np.random.random()
            done = np.random.random() > 0.8
            info = {'avg_pde_loss': 0.1, 'avg_c_mse_loss': 0.1, 'avg_t_mse_loss': 0.1}
            return next_obs, reward, done, info
    
    # Create agent
    policy = GNNPolicy(num_sensor_nodes=64)
    env = DummyEnv()
    agent = RLAgent(policy, env, device='cpu')
    
    # Test action selection
    obs = create_dummy_graph_data()
    action = agent.select_action(obs)
    
    # Check buffer contents
    assert len(agent.buffer.observations) == 1, "Should have 1 observation"
    assert len(agent.buffer.actions) == 1, "Should have 1 action"
    assert len(agent.buffer.logprobs) == 1, "Should have 1 logprob"
    assert len(agent.buffer.values) == 1, "Should have 1 value"
    
    print(f"✓ Buffer observations: {len(agent.buffer.observations)}")
    print(f"✓ Buffer actions: {len(agent.buffer.actions)}")
    print(f"✓ Buffer logprobs: {len(agent.buffer.logprobs)}")
    print(f"✓ Buffer values: {len(agent.buffer.values)}")
    print("✓ Agent buffer test passed!")
    return True

def test_ppo_update():
    """Test that PPO update runs without errors."""
    print("\nTesting PPO update...")
    
    # Create dummy environment class
    class DummyEnv:
        def reset(self):
            return create_dummy_graph_data()
        
        def step(self, action):
            next_obs = create_dummy_graph_data()
            reward = np.random.random()
            done = np.random.random() > 0.8
            info = {'avg_pde_loss': 0.1, 'avg_c_mse_loss': 0.1, 'avg_t_mse_loss': 0.1}
            return next_obs, reward, done, info
    
    # Create agent
    policy = GNNPolicy(num_sensor_nodes=64)
    env = DummyEnv()
    agent = RLAgent(policy, env, device='cpu', update_epochs=2)
    
    # Collect some transitions
    obs = create_dummy_graph_data()
    for i in range(5):
        action = agent.select_action(obs)
        agent.store_reward(np.random.random(), i == 4)  # Last one is done
    
    # Test PPO update without physics losses
    try:
        agent.update_policy(obs)
        print("✓ PPO update (no physics) completed without errors!")
    except Exception as e:
        print(f"✗ PPO update (no physics) failed: {e}")
        return False
    
    # Test PPO update with physics losses
    try:
        physics_losses = {
            'avg_pde_loss': 0.05,
            'avg_c_mse_loss': 0.02,
            'avg_t_mse_loss': 0.03
        }
        # Collect new transitions
        for i in range(3):
            action = agent.select_action(obs)
            agent.store_reward(np.random.random(), i == 2)
        
        agent.update_policy(obs, physics_losses=physics_losses)
        print("✓ PPO update with physics losses completed without errors!")
        return True
    except Exception as e:
        print(f"✗ PPO update with physics losses failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Fixed PPO Implementation")
    print("=" * 50)
    
    tests = [
        test_policy_interface,
        test_agent_buffer,
        test_ppo_update
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! PPO implementation is working correctly.")
    else:
        print("❌ Some tests failed. Check the implementation.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()
