import torch
import numpy as np
from dataset import TofDataset
from torch.utils.data import DataLoader
from graph_dataset import GraphDataset
from RL.env.acoustic_env import AcousticEnv
from gnn_policy import GNNPolicy
from rl_agent import RLAgent


def build_dummy_config():
    """Creates a tiny config just for a runnable example."""
    # toy positions: 32 sources & 32 receivers on a circle of radius 1
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    sources_positions = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * 1.0  # (32, 2)
    receivers_positions = np.stack([np.cos(angles + np.pi / 32), np.sin(angles + np.pi / 32)], axis=-1) * 1.0

    # synth ground‑truth ToF matrix (S, R)
    S = len(sources_positions)
    R = len(receivers_positions)
    true_c = 1.5  # homogeneous speed for dummy test
    tof_matrix = np.zeros((S, R), dtype=np.float32)
    for i in range(S):
        for j in range(R):
            dist = np.linalg.norm(sources_positions[i] - receivers_positions[j])
            tof_matrix[i, j] = dist / true_c

    return {
        "sources_positions": sources_positions,
        "receivers_positions": receivers_positions,
        "tof_matrix": tof_matrix,
        "device": "cpu",
        "selected_sources": [0],  # single‑source example
    }


def train(config, total_episodes=200):
    # 1. Build GraphDataset
    graph_dataset = GraphDataset(
        nx=8,
        ny=8,
        mesh_node_k=10,
        sensor_k=4,
        num_source_nodes=len(config["sources_positions"]),
        num_receiver_nodes=len(config["receivers_positions"]),
        device=config.get("device", "cpu"),
    )

    # 2. Create environment
    env = AcousticEnv(config, graph_dataset)

    # 3. Create GNNPolicy
    policy = GNNPolicy(num_sensor_nodes=env.num_source_nodes + env.num_receiver_nodes)

    # 4. RLAgent
    agent = RLAgent(policy, env, device=config.get("device", "cpu"))

    # 5. Train
    agent.train(total_episodes=total_episodes, max_steps=10)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TofDataset(['train'])
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    cfg = build_dummy_config()
    train(cfg)
