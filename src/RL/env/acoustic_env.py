import torch
import numpy as np
from torch_geometric.data import Data
from logger import visualize_matdata
from .wave_solver import WaveSolver   # assumes the WaveSolver text‑doc is saved as wave_solver.py on disk


class AcousticEnv:
    """Environment wrapper that interacts with RLAgent.

    * action *   – a LongTensor of size **N** (one integer per node) with
                   values in {0,1,2,3,4} that pick a discrete c‑bin for **mesh** nodes.
                   Sensor nodes (sources & receivers) are ignored / clamped.

    The environment keeps a global c‑map (speed of sound per node),
    re‑runs the wave solver, computes reward, and returns a fresh
    list of PyG Data objects (one per selected source).
    """

    C_BINS = torch.tensor([0.1, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])  # candidate SoS values


    # -------------------------------------------------------------
    # Construction / Reset
    # -------------------------------------------------------------
    def __init__(
        self,
        config,
        graph_dataset,
        receiver_accuracy_threshold: float = 0.8,  # 80% of receivers must be accurate
        receiver_tof_threshold: float = 0.2,      # 20% error threshold for each receiver
        source_completion_threshold: float = 0.8,  # 80% of sources must be done
    ):
        self.device = config.get("device", "cpu")
        self.c_init = config['c_init']

        # positions & ground‑truth ToF
        self.sources_positions = np.asarray(config["sources_positions"], dtype=np.float32)
        self.receivers_positions = np.asarray(config["receivers_positions"], dtype=np.float32)
        self.tof_matrix = torch.tensor(config["tof_matrix"], dtype=torch.float32, device=self.device).clone().detach()

        # selected sources per episode (can be 1..S)
        self.selected_sources = list(config.get("selected_sources", range(len(self.sources_positions))))
        self.curr_source_idx = 0  # index into selected_sources list

        # Accuracy thresholds
        self.receiver_accuracy_threshold = receiver_accuracy_threshold
        self.receiver_tof_threshold = receiver_tof_threshold
        self.source_completion_threshold = source_completion_threshold

        # graph
        self.graph_dataset = graph_dataset
        self.graph_dataset.build(self.sources_positions, self.receivers_positions)
        # counts
        self.num_source_nodes = len(self.sources_positions)
        self.num_receiver_nodes = len(self.receivers_positions)
        self.num_mesh_nodes = self.graph_dataset.num_mesh_nodes
        self.total_nodes = self.num_source_nodes + self.num_receiver_nodes + self.num_mesh_nodes

        self.full_mesh = torch.full(config['full_mesh_resolution'], self.c_init, device=self.device)
        
        # speed map – initialise with middle bin (1.2)
        self.c_map = torch.full((self.total_nodes,), self.c_init, device=self.device)
        # keep sensor nodes fixed (optional – could also let them vary)
        self.c_map[: self.num_source_nodes + self.num_receiver_nodes] = self.c_init

        # solver
        self.solver = WaveSolver(num_iterations=20)

    # -------------------------------------------------------------
    def reset(self):
        """Reset environment state & return initial observation (PyG list)."""
        self.c_map[self.num_source_nodes + self.num_receiver_nodes :] = self.c_init  # mesh reset
        self.full_mesh = torch.full(self.full_mesh.shape, self.c_init, device=self.device)
        return self._build_observation()

    def test_cmap(self, c_map, src_id):
        #mesh_start = self.num_source_nodes + self.num_receiver_nodes

        #self.c_map[mesh_start: mesh_start + self.num_mesh_nodes] = torch.from_numpy(mesh_cmap)
        self.c_map = c_map
        return self._run_wave_simulation(src_id)

    # -------------------------------------------------------------
    def _update_full_mesh(self, mesh_actions):
        """Update the full 128x128 mesh with agent actions based on mesh node positions."""
        mesh_start = self.num_source_nodes + self.num_receiver_nodes
        mesh_positions = self.graph_dataset.positions[mesh_start:]
        
        # Reset mesh to base value
        self.full_mesh.fill_(self.c_init)
        
        # Update mesh with agent actions
        for i, (x, y) in enumerate(mesh_positions):
            x_idx = int(x)
            y_idx = int(y)
            self.full_mesh[y_idx, x_idx] = AcousticEnv.C_BINS[mesh_actions[i]]

    def _check_source_done(self, T_receivers, target_tof):
        """
        Check if a source is 'done' based on receiver accuracy.
        Returns True if enough receivers have accurate ToF values.
        """
        # Calculate relative error for each receiver
        relative_errors = (T_receivers - target_tof).abs() / (target_tof + 1e-6)
        
        # Count receivers with error below threshold
        accurate_receivers = (relative_errors < self.receiver_tof_threshold).float().mean()
        
        # Source is done if enough receivers are accurate
        return accurate_receivers >= self.receiver_accuracy_threshold

    def _extract_receiver_values(self, T_grid, receiver_positions):
        """
        Extract time-of-flight values for receiver positions from the full T grid.
        
        Args:
            T_grid: 2D tensor of time-of-flight values for the entire mesh
            receiver_positions: Array of receiver positions (x, y)
            
        Returns:
            T_receivers: Tensor of time-of-flight values for receiver nodes
        """
        T_receivers = []
        for x, y in receiver_positions:
            x_idx = int(x)
            y_idx = int(y)
            T_receivers.append(T_grid[y_idx, x_idx])
        return torch.tensor(T_receivers, dtype=torch.float32, device=T_grid.device)

    def _extract_mesh_values(self, T_grid, mesh_positions):
        """
        Extract time-of-flight values for receiver positions from the full T grid.

        Args:
            T_grid: 2D tensor of time-of-flight values for the entire mesh
            mesh_positions: Array of receiver positions (x, y)

        Returns:
            T_mesh: Tensor of time-of-flight values for mesh nodes
        """
        T_mesh = []
        for x, y in mesh_positions:
            x_idx = int(x)
            y_idx = int(y)
            T_mesh.append(T_grid[y_idx, x_idx])
        return torch.tensor(T_mesh, dtype=torch.float32, device=T_grid.device)
    def _run_wave_simulation(self, src_id):
        # -----------------------------------------------------
        # Wave simulation for the *current* source
        # -----------------------------------------------------

        (_, data) = next(
            self.graph_dataset.get_graph(
                tof_matrix=self.tof_matrix, selected_sources=[src_id], device=self.device
            )
        )

        # Get source position
        source_pos = data.pos[src_id]

        # Simulate T on full mesh
        T_grid = self.solver.simulate_T(self.full_mesh, source_pos)
        
        # Extract T values for receivers
        receiver_start = self.num_source_nodes
        receiver_end = receiver_start + self.num_receiver_nodes
        receiver_positions = data.pos[receiver_start:receiver_end]

        T_receivers = self._extract_receiver_values(T_grid, receiver_positions)
        T_full = self._extract_mesh_values(T_grid,  data.pos)

        # Compute reward using accuracy-based metric
        reward = self.accuracy_reward(T_receivers, self.tof_matrix[src_id])
        
        # Check if this source is done
        source_done = self._check_source_done(T_receivers, self.tof_matrix[src_id])
        data.x[:, 0] = T_full

        #print(f"source ID: {src_id}")
        #print(f"simulated TOF values:")
        #print(T_receivers)
        #print(f"real values:")
        #print(self.tof_matrix[src_id])

        # Calculate MAE for info (keeping it for monitoring)
        mae = (T_receivers - self.tof_matrix[src_id]).abs().mean()
        info = {
            "mae": mae.item(), 
            "source": src_id, 
            "reward": reward.item(),
            "source_done": source_done,
            #"accurate_receivers": (relative_errors < self.receiver_tof_threshold).float().mean().item()
        }
        return data, reward, source_done, info

    def step(self, action):
        """
        action – LongTensor (N,) with discrete indices produced by the agent.
                 Only mesh nodes (last *num_mesh_nodes*) are applied.
        """

        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        mesh_start = self.num_source_nodes + self.num_receiver_nodes
        mesh_actions = action[mesh_start : mesh_start + self.num_mesh_nodes]
        
        # Update full mesh with agent actions
        self._update_full_mesh(mesh_actions)
        # Update c_map for the graph nodes
        self.c_map[mesh_start:] = AcousticEnv.C_BINS[mesh_actions]

        # Run wave simulation for all selected sources
        total_accuracy_rewards = 0.0
        all_observations = []
        all_infos = []
        done_sources = 0

        for src_id in self.selected_sources:
            observation, reward, source_done, info = self._run_wave_simulation(src_id)
            total_accuracy_rewards += reward
            all_observations.append(observation)
            all_infos.append(info)
            if source_done:
                done_sources += 1

        # Compute average reward across all sources
        avg_reward = -total_accuracy_rewards / len(self.selected_sources)
        
        # Check if enough sources are done
        sources_done_ratio = done_sources / len(self.selected_sources)
        mission_done = sources_done_ratio >= self.source_completion_threshold
        
        # Return the last observation (we could also return all observations if needed)
        observation = all_observations[-1]
        info = {
            "sources": self.selected_sources,
            "individual_maes": [info["mae"] for info in all_infos],
            "sources_done_ratio": sources_done_ratio,
            "done_sources": done_sources,
            "total_sources": len(self.selected_sources),
            "avg_reward": avg_reward
        }
        return observation, total_accuracy_rewards, mission_done, info

    def accuracy_reward(self, pred, target, power=30, multiple=1e-30):
        # Compute percent accuracy per cell (assuming target > 0)
        accuracy = 100.0 - (100.0 * torch.abs(pred - target) / target.clamp(min=1e-6))
        # Normalize to [0, 1]

        # Scale using steep power law
        reward = multiple * accuracy.pow(power)

        # Normalize to [0, 1] range (by dividing by max possible value)
        reward = reward / (1.0 ** power)  # max is 1^power
        return reward.mean()  # average over all cells



    # -------------------------------------------------------------
    def _build_observation(self):
        """Helper to build initial observation for the first source."""
        src_id = self.selected_sources[self.curr_source_idx]
        (_, data) = next(
            self.graph_dataset.get_graph(
                tof_matrix=self.tof_matrix, selected_sources=[src_id], device=self.device
            )
        )
        # insert current c map
        data.x[:, 1] = self.c_map
        return data

    # -------------------------------------------------------------
    def render(self, mode="human"):
        pass