import torch
import numpy as np
from torch_geometric.data import Data

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

    C_BINS = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])  # candidate SoS values


    # -------------------------------------------------------------
    # Construction / Reset
    # -------------------------------------------------------------
    def __init__(self, config, graph_dataset):
        self.device = config.get("device", "cpu")
        self.c_init = config['c_init']

        # positions & ground‑truth ToF
        self.sources_positions = np.asarray(config["sources_positions"], dtype=np.float32)
        self.receivers_positions = np.asarray(config["receivers_positions"], dtype=np.float32)
        self.tof_matrix = torch.tensor(config["tof_matrix"], dtype=torch.float32, device=self.device).clone().detach()

        # selected sources per episode (can be 1..S)
        self.selected_sources = list(config.get("selected_sources", range(len(self.sources_positions))))
        self.curr_source_idx = 0  # index into selected_sources list

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
        self.curr_source_idx = 0
        self.c_map[self.num_source_nodes + self.num_receiver_nodes :] = self.c_init  # mesh reset
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

    def step(self, action):
        """
        action – LongTensor (N,) with discrete indices 0..4 produced by the agent.
                 Only mesh nodes (last *num_mesh_nodes*) are applied.
        """
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        mesh_start = self.num_source_nodes + self.num_receiver_nodes
        mesh_actions = action[mesh_start : mesh_start + self.num_mesh_nodes]
        
        # Update full mesh with agent actions
        self._update_full_mesh(mesh_actions)
        
        # Update c_map for the graph nodes
        self.c_map[mesh_start:] = AcousticEnv.C_BINS[mesh_actions]
        
        src_id = self.selected_sources[self.curr_source_idx]
        return self._run_wave_simulation(src_id)

    def _run_wave_simulation(self, src_id):
        # -----------------------------------------------------
        # Wave simulation for the *current* source
        # -----------------------------------------------------

        (_, data) = next(
            self.graph_dataset.get_graph(
                tof_matrix=self.tof_matrix, selected_sources=[src_id], device=self.device
            )
        )

        # override c‑channel with updated c_map
        data.x[:, 1] = self.c_map

        # simulate T
        T_pred = self.solver.simulate_T(data, src_id)
 
        # put back into data for the agent to observe if desired
        data.x[:, 0] = T_pred
        # -----------------------------------------------------
        # Reward: negative MAE on receiver nodes
        # -----------------------------------------------------
        recv_start = self.num_source_nodes
        recv_end = recv_start + self.num_receiver_nodes

        print(f"source ID: {src_id}")
        #print(T_pred[0:recv_start])
        print(f"simulated TOF values:")
        print(T_pred[recv_start:recv_end])
        print(f"real values:")
        print(self.tof_matrix[src_id])

        #exit()
        mae = (T_pred[recv_start:recv_end] - self.tof_matrix[src_id]).abs().mean()
        reward = -mae.item()
        # -----------------------------------------------------
        # Prepare next observation & done flag
        # -----------------------------------------------------
        self.curr_source_idx += 1
        done = self.curr_source_idx >= len(self.selected_sources)
        observation = data
        info = {"mae": mae.item(), "source": src_id}
        # if episode finished, you might shuffle sources or compute extra stats
        return observation, reward, done, info

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