import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
from logger import visualize_matdata
from .wave_solver import WaveSolver   # assumes the WaveSolver text‑doc is saved as wave_solver.py on disk
from .differentiable_wave_solver import create_differentiable_wave_solver
from tqdm import tqdm

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
        use_differentiable_solver: bool = False,  # Enable differentiable wave solver
        solver_optimization_level: str = 'standard',  # 'basic', 'standard', or 'optimized'
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

        # solver configuration
        self.use_differentiable_solver = use_differentiable_solver
        self.solver_optimization_level = solver_optimization_level
        
        # Initialize solvers
        self.solver = WaveSolver()  # Original non-differentiable solver
        if self.use_differentiable_solver:
            self.differentiable_solver = create_differentiable_wave_solver(solver_optimization_level)
            print(f"Initialized differentiable wave solver with optimization level: {solver_optimization_level}")
        else:
            self.differentiable_solver = None

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
        print(f"mesh actions : shape {mesh_actions.shape} min/max: {mesh_actions.min()} {mesh_actions.max()}")

        self.c_map[self.num_source_nodes + self.num_receiver_nodes:] += mesh_actions
        xr = self.graph_dataset.x_range
        yr = self.graph_dataset.y_range
        self.full_mesh[xr[0]:xr[1], yr[0]:yr[1]] = self.c_map[self.num_source_nodes + self.num_receiver_nodes:].view(xr[1]-xr[0], yr[1]-yr[0])



        """
        # Create a new full mesh tensor
        new_full_mesh = self.full_mesh.clone()
        
        # Update mesh with agent actions
        for i, (x, y) in enumerate(mesh_positions):
            x_idx = int(round(x))
            y_idx = int(round(y))
            if 0 <= x_idx < new_full_mesh.shape[1] and 0 <= y_idx < new_full_mesh.shape[0]:
                new_full_mesh[y_idx, x_idx] = self.c_map[mesh_start + i]
        # Update the full mesh
        self.full_mesh = new_full_mesh
        """

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



    def _run_wave_simulation(self, src_id):
        source_pos = self.sources_positions[src_id]
        receiver_positions = self.receivers_positions

        # Debug prints
        #print(f"full_mesh shape: {self.full_mesh.shape}")
        #print(f"full_mesh min/max values: {self.full_mesh.min()}, {self.full_mesh.max()}")
        #print(f"full_mesh device: {self.full_mesh.device}")

        # Ensure full_mesh has gradients and create a new tensor to avoid modifying the original
        full_mesh = self.full_mesh.clone().detach().requires_grad_(True)

        # Choose solver based on configuration
        if self.use_differentiable_solver and self.differentiable_solver is not None:
            # Use differentiable solver that maintains gradient flow
            T_grid = self.differentiable_solver.simulate_T(full_mesh, source_pos)
        else:
            # Use original non-differentiable solver
            T_grid = self.solver.simulate_T(full_mesh, source_pos)
        
        T_receivers = self._extract_receiver_values(T_grid, receiver_positions)

        # Compute reward using accuracy-based metric
        reward = self.accuracy_reward(T_receivers, self.tof_matrix[src_id])

        # Check if this source is done
        source_done = self._check_source_done(T_receivers, self.tof_matrix[src_id])

        #print(f"source ID: {src_id}")
        #print(f"simulated TOF values:")
        #print(T_receivers)
        #print(f"real values:")
        #print(self.tof_matrix[src_id])

        T = T_grid  # Time of flight values
        c = full_mesh  # Speed of sound values
        physics_loss = self._compute_physics_loss(T, c)

        criterion = nn.MSELoss()
        boundary_loss = criterion(T_receivers, self.tof_matrix[src_id])

        c_mse_loss = criterion(c, self.sos_matrix)
        t_mse_loss = criterion(T_receivers, self.tof_matrix[src_id])
        return {
            "source": src_id,
            "boundary_loss": boundary_loss,
            "c_mse_loss": c_mse_loss,
            "t_mse_loss": t_mse_loss,
            "reward": reward,
            "source_done": source_done,
            "pde_loss": physics_loss,
            "t_grid": T_grid
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
        total_c_mse_loss = 0.0
        total_t_mse_loss = 0.0
        all_observations = []
        all_infos = []
        done_sources = 0
        checked_sources = 0
        t_grids = []
        with tqdm(total=len(self.selected_sources)) as pbar:
            pbar.set_description("simulating wave propagation with c_map")
            for src_id in self.selected_sources:
                #print(f"checking src {src_id}")
                pbar.update(1)
                info = self._run_wave_simulation(src_id)
                t_grids.append(info["t_grid"].detach().cpu())  # (H, W)
                #print(f"reward: {info['reward']}")
                total_accuracy_rewards += info['reward']
                total_pde_loss += info['pde_loss']
                total_c_mse_loss += info['c_mse_loss']
                total_t_mse_loss += info['t_mse_loss']
                total_accuracy_rewards += info['reward']
                all_infos.append(info)
                checked_sources+=1
                #if info['reward'] < 0.5:
                #    print(f"reward is too low - break")
                #    break

                if info['source_done']:
                    done_sources += 1

        # Stack t_grids: shape (num_sources, H, W)
        t_grids = torch.stack(t_grids, dim=0)  # (num_sources, H, W)
        t_mean = t_grids.mean(dim=0)           # (H, W)
        t_std = t_grids.std(dim=0)             # (H, W)
        #print(f"t_mean min/max values: {t_mean.min()}, {t_mean.max()}")
        #print(f"t_std min/max values: {t_std.min()}, {t_std.max()}")

        # Now, map these mean/std values to the graph nodes
        # You need to know the (x, y) position of each node in the graph
        # Let's assume you have node positions in self.graph_dataset.positions (shape: [num_nodes, 2])
        node_positions = self.graph_dataset.positions  # shape: (num_nodes, 2)
        node_mean = []
        node_std = []
        for x, y in node_positions:
            x_idx = int(x)
            y_idx = int(y)
            node_mean.append(t_mean[y_idx, x_idx])
            node_std.append(t_std[y_idx, x_idx])
        node_mean = torch.tensor(node_mean, device=self.device, dtype=torch.float32)
        node_std = torch.tensor(node_std, device=self.device, dtype=torch.float32)

        # Inject into observation
        self.observation.x[:, 0] = self.c_map
        self.observation.x[:, 1] = node_mean /1000
        self.observation.x[:, 2] = node_std /1000
        all_observations.append(self.observation)

        # Compute average reward across all sources
        avg_reward = -total_accuracy_rewards / checked_sources
        avg_pde_loss = total_pde_loss / checked_sources
        avg_c_mse_loss = total_c_mse_loss / checked_sources
        avg_t_mse_loss = total_t_mse_loss / checked_sources

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
            "avg_c_mse_loss": avg_c_mse_loss,
            "avg_t_mse_loss": avg_t_mse_loss
        }
        return observation, total_accuracy_rewards, mission_done, info

    def accuracy_reward(self, pred, target, power=10, multiple=1e-6):
        """
        High-power accuracy reward computation that gives more rewards to accurate values.
        Uses exponential scaling to heavily reward accurate predictions.
        """
        # Compute relative error
        relative_error = torch.abs(pred - target) / target.clamp(min=1e-6)
        
        # Convert to accuracy (1 - error), clamped to [0, 1]
        accuracy = torch.clamp(1.0 - relative_error, min=0.0, max=1.0)
        
        # Apply high power to emphasize accurate predictions
        reward = multiple * accuracy.pow(power)
        
        return reward.mean()  # average over all receivers



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
