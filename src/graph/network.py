import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import torch.nn.functional as F


class GraphDataset:
    def __init__(self, **kwargs):
        self.c_init = kwargs.get('c_init', 0.12)
        self.t_init = kwargs.get('t_init', 0)
        self.epsilon = 1e-5
        self.x_range = kwargs.get('x_range', (32, 97))
        self.y_range = kwargs.get('y_range', (32, 97))
        self.nx = kwargs.get('nx', 8)
        self.ny = kwargs.get('ny', 8)

        # Connectivity parameters
        self.mesh_node_k = kwargs.get('mesh_node_k', 20)
        self.sensor_k = kwargs.get('sensor_k', 8)
        self.bidirectional = kwargs.get('bidirectional', True)  # to toggle directed vs undirected

        # S/R counts
        self.num_source_nodes = kwargs.get('num_source_nodes', 32)
        self.num_receiver_nodes = kwargs.get('num_receiver_nodes', 32)
        self.num_mesh_nodes = int((self.x_range[1]-self.x_range[0])/self.nx * (self.y_range[1]-self.y_range[0])/self.ny)
        self.num_sensor_nodes = self.num_source_nodes + self.num_receiver_nodes

        # Graph storage
        self.global_edges = None  # store adjacency for all nodes
        self.positions = None
        self.mesh_positions = None
        self.initialized = False

    def build(self, sources_positions, receivers_positions):
        """
            Build global mesh + adjacency for all nodes (sources, receivers, mesh).
            Creates a single adjacency structure that won't change across sources.
            """
        # 1) Store positions
        #    positions layout: [S(0..S-1), R(S..S+R-1), M(S+R..end)]
        #    indexes: source i => i, receiver j => (num_source_nodes + j), mesh => the rest
        self.mesh_positions = self._create_interior_mesh()
        self.positions = np.concatenate((sources_positions, receivers_positions, self.mesh_positions), axis=0)
        self.initialized = True

        # 2) Build adjacency
        #    a) mesh <-> mesh
        mesh_edges = self._build_mesh_edges()
        #    b) sources -> mesh
        src_edges = self._connect_sensors_to_mesh(
            sensor_indices=range(self.num_source_nodes),
            sensor_positions=sources_positions,sensor_type='S'
        )
        #    c) receivers -> mesh
        rcv_offsets = range(self.num_source_nodes, self.num_source_nodes + self.num_receiver_nodes)
        rcv_edges = self._connect_sensors_to_mesh(
            sensor_indices=rcv_offsets,
            sensor_positions=receivers_positions, sensor_type='R'
        )

        # Combine all edges
        edges_all = np.concatenate((mesh_edges, src_edges, rcv_edges), axis=0)
  
        # Remove duplicates & unify
        structured = np.zeros((edges_all.shape[0], 2))  # (i, j)
        structured[:, 0:2] = edges_all

        unique_set = set()
        unique_list = []
        for row in structured:
            i, j = row
            i, j = int(i), int(j)

            if (i, j) not in unique_set:
                unique_set.add((i, j))
                unique_list.append([i, j])

        unique_arr = np.array(unique_list)

        #  add reverse unless they exist
        total_sensor_nodes = self.num_source_nodes + self.num_receiver_nodes
        if self.bidirectional:
            reverse_list = []
            for e in unique_arr:
                i, j = e
                if i>total_sensor_nodes and j> total_sensor_nodes and (j, i) not in unique_set:
                    reverse_list.append([j, i])
            if len(reverse_list) > 0:
                reverse_arr = np.array(reverse_list)
                unique_arr = np.concatenate((unique_arr, reverse_arr), axis=0)

        self.global_edges = unique_arr.astype(np.int64)  # shape => (E, 2)

    def get_graph(self, tof_matrix, selected_sources, device):
        """
        Generator function that yields node feature matrices for each source i.
        Args:
            tof_matrix: (S, R) tensor or array of measured ToF values (S sources, R receivers),
                        tof_matrix[i, j] = measured time-of-flight from source i to receiver j.
            selected_sources:  the indexes of sources the graph will be based on
            device: GPU / CPU
        Yields:
            A tuple (i, data) for each source i:
             - i is the source index (0 <= i < self.num_source_nodes).
             - data is a PyG Data object with:
                 data.x[:, 0] = ToF values
                 data.x[:, 1] = mean ToF per receiver (0 for source/mesh)
                 data.x[:, 2] = std ToF per receiver (0 for source/mesh)
                 data.x[:, 3] = role encoding (1=source, 2=receiver, 3=mesh)
                 data.x[:, 4:6] = normalized coordinates
        """
        if not self.initialized:
            raise ValueError("GraphDataset not built yet!")

        if not isinstance(tof_matrix, torch.Tensor):
            tof_matrix = torch.tensor(tof_matrix, dtype=torch.float32)

        total_nodes = self.num_source_nodes + self.num_receiver_nodes + self.num_mesh_nodes

        # Convert positions to tensor and normalize coordinates
        pos_tensor = torch.tensor(self.positions, dtype=torch.float32, device=device)
        coords_normalized = pos_tensor / 128.0  # Normalize to [0,1] range

        # Compute ToF statistics per receiver using only selected sources
        selected_tof = tof_matrix[selected_sources]
        tof_mean = selected_tof.mean(dim=0)  # Mean ToF per receiver
        tof_std = selected_tof.std(dim=0)    # Std ToF per receiver
        tof_std = torch.clamp(tof_std, min=1e-6)  # Avoid division by zero

        edge_index = torch.tensor(self.global_edges, dtype=torch.long, device=device).T  # (2, E)

    
        # Initialize feature tensors
        c_init = torch.full((total_nodes,), self.c_init, dtype=torch.float32, device=device)
        mean_values = torch.full((total_nodes,), 1e-6, dtype=torch.float32, device=device)
        std_values = torch.full((total_nodes,), 1e-6, dtype=torch.float32, device=device)

        role_values = torch.zeros((total_nodes,), dtype=torch.float32, device=device)

        # Set receiver values
        receiver_start = self.num_source_nodes
        receiver_end = receiver_start + self.num_receiver_nodes            
        mean_values[receiver_start:receiver_end] = tof_mean
        std_values[receiver_start:receiver_end] = tof_std
        
        role_values[0:receiver_start] = 1.0 # Source role
        role_values[receiver_start:receiver_end] = 2.0  # Receiver role
        role_values[receiver_end:] = 3.0  # Mesh role
        
        # Stack all features
        x = torch.stack([
            c_init,                    # Current SOS values
            mean_values,                    # Mean ToF per receiver
            std_values,                     # Std ToF per receiver
            role_values,                    # Role encoding
        ], dim=-1)

        data = Data(
            x=x,
            edge_index=edge_index,
            pos=pos_tensor
        )

        yield data

    def _create_interior_mesh(self):
        """
        Creates a 2D grid of interior (virtual) nodes using cKDTree-based k-NN connectivity.
        """
        x_nodes_count = int((self.x_range[1]-self.x_range[0]) / self.nx)
        y_nodes_count = int((self.y_range[1]-self.y_range[0]) / self.ny)
        x_values = np.linspace(self.x_range[0], self.x_range[1], x_nodes_count)
        y_values = np.linspace(self.y_range[0], self.y_range[1], y_nodes_count)
        coords = []
        for i in range(x_nodes_count):
            for j in range(y_nodes_count):
                coords.append((x_values[i], y_values[j]))
        return np.array(coords, dtype=np.float32)

    def _build_mesh_edges(self):
        edges = []
        mesh_start = self.num_sensor_nodes
        mesh_positions = self.positions[mesh_start:]

        tree = cKDTree(mesh_positions)
        for i in range(self.num_mesh_nodes):
            pos_i = mesh_positions[i]
            _, nbr_indices = tree.query(pos_i, k=self.mesh_node_k)
            i_global = i + mesh_start
            for nbr in nbr_indices:
                if i == nbr:
                    continue
                j_global = nbr + mesh_start
                edges.append((i_global, j_global))

        return np.array(edges, dtype=np.int64)

    def _connect_sensors_to_mesh(self, sensor_indices, sensor_positions, sensor_type):
        edges = []
        mesh_start = self.num_sensor_nodes
        mesh_positions = self.positions[mesh_start:]
        tree = cKDTree(mesh_positions)

        for local_idx, s_idx_global in enumerate(sensor_indices):
            s_pos = sensor_positions[local_idx]
            _, nbrs = tree.query(s_pos, k=self.sensor_k)
            if self.sensor_k == 1:
                nbrs = [nbrs]

            for nbr in nbrs:
                j_global = nbr + mesh_start
                if s_idx_global == j_global:
                    continue

                edges.append((s_idx_global, j_global))  # out
                edges.append((j_global, s_idx_global))  # in

        return np.array(edges, dtype=np.int64)

