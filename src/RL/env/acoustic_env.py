import torch
import numpy as np
import torch.nn as nn
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
        self.tof_matrix = config["tof_matrix"].to(self.device)
        self.sos_matrix = config["sos_matrix"].to(self.device)

        # selected sources per episode (can be 1..S)
        self.selected_sources = list(config.get("selected_sources", range(len(self.sources_positions))))


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

        # solver
        self.solver = WaveSolver(num_iterations=20)

    # -------------------------------------------------------------
    def reset(self):
        """Reset environment state & return initial observation (PyG list)."""
        self.c_map[self.num_source_nodes + self.num_receiver_nodes :] = self.c_init  # mesh reset
        self.full_mesh = torch.full(self.full_mesh.shape, self.c_init, device=self.device)
        self.observation = self._build_observation()
        return self.observation

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
        
        # Update c_map with mesh actions
        self.c_map = self.c_map.clone()
        self.c_map[self.num_source_nodes + self.num_receiver_nodes:] += mesh_actions

        # Create a new full mesh tensor
        new_full_mesh = self.full_mesh.clone()
        
        # Update mesh with agent actions using scatter
        indices = torch.tensor([[int(y), int(x)] for x, y in mesh_positions], device=self.device)
        values = self.c_map[mesh_start:mesh_start + self.num_mesh_nodes]
        
        # Reshape indices and values for scatter
        indices = indices.t()  # Transpose to [2, num_points]
        #values = values.unsqueeze(0)  # Add batch dimension [1, num_points]
        
        # Use scatter_ to update values
        new_full_mesh.scatter_(0, indices, values)
        
        # Update the full mesh
        self.full_mesh = new_full_mesh

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
        source_pos = self.sources_positions[src_id]
        receiver_positions = self.receivers_positions

        # Ensure full_mesh has gradients
        self.full_mesh.requires_grad_(True)

        # Simulate T on full mesh - ensure solver returns tensor with grads
        T_grid = self.solver.simulate_T(self.full_mesh, source_pos)
        if not isinstance(T_grid, torch.Tensor):
            T_grid = torch.tensor(T_grid, device=self.device, requires_grad=True)
        
        T_receivers = self._extract_receiver_values(T_grid, receiver_positions)

        # Compute reward using accuracy-based metric
        reward = self.accuracy_reward(T_receivers, self.tof_matrix[src_id])

        # Check if this source is done
        source_done = self._check_source_done(T_receivers, self.tof_matrix[src_id])

        print(f"source ID: {src_id}")
        print(f"simulated TOF values:")
        print(T_receivers)
        print(f"real values:")
        print(self.tof_matrix[src_id])

        T = T_grid  # Time of flight values
        c = self.full_mesh  # Speed of sound values
        physics_loss = self._compute_physics_loss(T, c)

        criterion = nn.MSELoss()
        boundary_loss = criterion(T_receivers, self.tof_matrix[src_id])

        mse_loss = criterion(c, self.sos_matrix)
        return {
            "source": src_id,
            "boundary_loss": boundary_loss,
            "mse_loss": mse_loss,
            "reward": reward,
            "source_done": source_done,
            "pde_loss": physics_loss
        }


    def step(self, action):
        """
        action – LongTensor (N,) with discrete indices produced by the agent.
                 Only mesh nodes (last *num_mesh_nodes*) are applied.
        """

        action = torch.as_tensor(action, device=self.device)
        mesh_start = self.num_source_nodes + self.num_receiver_nodes
        mesh_actions = action[mesh_start : mesh_start + self.num_mesh_nodes]
        
        # Update full mesh with agent actions
        self._update_full_mesh(mesh_actions)

        # Run wave simulation for all selected sources
        total_accuracy_rewards = 0.0
        total_pde_loss = 0.0
        total_mse_loss = 0.0
        all_observations = []
        all_infos = []
        done_sources = 0

        self.observation.x[:, 0] = self.c_map
        all_observations.append(self.observation)

        checked_sources = 0
        for src_id in self.selected_sources:
            info = self._run_wave_simulation(src_id)
            print(f"reward: {info['reward']}")
            total_accuracy_rewards += info['reward']
            total_pde_loss += info['pde_loss']
            total_mse_loss += info['mse_loss']
            total_accuracy_rewards += info['reward']
            all_infos.append(info)
            checked_sources+=1
            if info['reward'] < 0.5:
                print(f"reward is too low - break")
                break

            if info['source_done']:
                done_sources += 1

        # Compute average reward across all sources
        avg_reward = -total_accuracy_rewards / checked_sources
        avg_pde_loss = total_pde_loss / checked_sources
        avg_mse_loss = total_mse_loss / checked_sources

        # Check if enough sources are done
        sources_done_ratio = done_sources / len(self.selected_sources)
        mission_done = sources_done_ratio >= self.source_completion_threshold
        
        # Return the last observation (we could also return all observations if needed)
        observation = all_observations[-1]
        info = {
            "sources": self.selected_sources,
            "sources_done_ratio": sources_done_ratio,
            "done_sources": done_sources,
            "checked_sources": checked_sources,
            "avg_reward": avg_reward,
            "avg_pde_loss": avg_pde_loss,
            "avg_mse_loss": avg_mse_loss
        }
        return observation, total_accuracy_rewards, mission_done, info

    def accuracy_reward(self, pred, target, power=10, multiple=1e-20):
        # Compute percent accuracy per cell (assuming target > 0)
        accuracy = 100 - (100* torch.abs(pred - target) / target.clamp(min=1e-6))
        # Normalize to [0, 1]

        # Scale using steep power law
        reward = multiple * accuracy.pow(power)

        return reward.mean()  # average over all cells



    # -------------------------------------------------------------
    def _build_observation(self):
        """Helper to build initial observation for the first source."""
        data = next(
            self.graph_dataset.get_graph(
                tof_matrix=self.tof_matrix, selected_sources=self.selected_sources, device=self.device
            )
        )
        # insert current c map
        data.x[:, 0] = self.c_map
        return data

    # -------------------------------------------------------------
    def render(self, mode="human"):
        pass

    def _compute_physics_loss(self, T, c):
        """
        Compute physics-based losses:
        1. Eikonal equation: |∇T| = 1/c
        2. Wave equation residual
        """
        # Compute gradients of T
        dx, dy = torch.gradient(T, spacing=(1.0, 1.0))
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

        # Eikonal equation residual
        eikonal_residual = (grad_mag - 1.0 / c).pow(2).mean()

        # Add wave equation residual if needed
        # wave_residual = ...

        return eikonal_residual