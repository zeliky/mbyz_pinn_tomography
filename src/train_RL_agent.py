import torch
import numpy as np
from dataset import TofDataset
from torch.utils.data import DataLoader
from graph.network import GraphDataset
from RL.env.acoustic_env import AcousticEnv
from RL.policy.gnn_policy import GNNPolicy
from RL.agent import RLAgent


def build_dummy_config():
    """Creates a tiny config just for a runnable example."""
    # Position sources and receivers around the anatomy perimeter
    # Sources and receivers positioned around a circle at radius ~50 (outside anatomy center)
    angles_src = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    angles_rcv = np.linspace(0, 2 * np.pi, 32, endpoint=False) + np.pi / 32
    
    # Scale and translate to be around the anatomy (center at 64, 64)
    radius = 50
    center = 64
    sources_positions = np.stack([
        center + radius * np.cos(angles_src), 
        center + radius * np.sin(angles_src)
    ], axis=-1)  # (32, 2)
    
    receivers_positions = np.stack([
        center + radius * np.cos(angles_rcv), 
        center + radius * np.sin(angles_rcv)
    ], axis=-1)  # (32, 2)

    # synth ground‑truth ToF matrix (S, R)
    S = len(sources_positions)
    R = len(receivers_positions)
    true_c = 1.5  # homogeneous speed for dummy test
    tof_matrix = np.zeros((S, R), dtype=np.float32)
    for i in range(S):
        for j in range(R):
            dist = np.linalg.norm(sources_positions[i] - receivers_positions[j])
            tof_matrix[i, j] = dist / true_c

    # Create dummy SOS matrix for the full 128x128 grid
    sos_matrix = np.full((128, 128), true_c, dtype=np.float32)

    return {
        "sources_positions": sources_positions,
        "receivers_positions": receivers_positions,
        "tof_matrix": torch.tensor(tof_matrix, dtype=torch.float32),
        "sos_matrix": torch.tensor(sos_matrix, dtype=torch.float32),
        "c_init": 1.2,
        "full_mesh_resolution": (128, 128),
        "device": "cpu",
        "selected_sources": list(range(32)),  # Use all sources
    }


def train(config, total_episodes=200):
    # 1. Build GraphDataset with 64x64 mesh in center (32:96, 32:96)
    graph_dataset = GraphDataset(
        c_init=1.2,  # Default SOS value
        x_range=(32, 96),  # Anatomy center x-range
        y_range=(32, 96),  # Anatomy center y-range
        nx=1,  # Grid spacing (64 nodes over 64 units = 1 unit spacing)
        ny=1,  # Grid spacing
        mesh_node_k=9,  # 9 nearest neighbors for mesh connectivity
        sensor_k=4,  # Not used with new connectivity
        num_source_nodes=len(config["sources_positions"]),
        num_receiver_nodes=len(config["receivers_positions"]),
        device=config.get("device", "cpu"),
    )

    # 2. Create environment with differentiable solver option
    env = AcousticEnv(
        config, 
        graph_dataset,
        use_differentiable_solver=True,  # Enable differentiable wave solver
        solver_optimization_level='standard'  # Use standard optimization level
    )

    # 3. Create GNNPolicy
    policy = GNNPolicy(num_sensor_nodes=env.num_source_nodes + env.num_receiver_nodes)

    # 4. RLAgent
    agent = RLAgent(policy, env, device=config.get("device", "cpu"))

    # 5. Train
    agent.train(max_steps=total_episodes)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TofDataset(['train'])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    cfg = build_dummy_config()
    train(cfg)
