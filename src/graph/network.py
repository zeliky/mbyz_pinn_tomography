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
        self.mesh_node_k = kwargs.get('mesh_node_k', 20)  # was mesh_node_connections
        self.sensor_k = kwargs.get('sensor_k', 8)  # separate from nx to avoid confusion
        self.bidirectional = kwargs.get('bidirectional', True)  # to toggle directed vs undirected

        # S/R counts
        self.num_source_nodes = kwargs.get('num_source_nodes', 32)
        self.num_receiver_nodes = kwargs.get('num_receiver_nodes', 32)
        self.num_mesh_nodes = int((self.x_range[1]-self.x_range[0])/self.nx * (self.y_range[1]-self.y_range[0])/self.ny)
        self.num_sensor_nodes = self.num_source_nodes + self.num_receiver_nodes

        # Graph storage
        self.global_edges = None  # store adjacency for all nodes
        self.global_edge_attrs = None  # store PDE-based edge attributes (e.g., distance)
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
        mesh_edges, mesh_edge_attrs = self._build_mesh_edges()
        #    b) sources -> mesh
        src_edges, src_edge_attrs = self._connect_sensors_to_mesh(
            sensor_indices=range(self.num_source_nodes),
            sensor_positions=sources_positions,sensor_type='S'
        )
        #    c) receivers -> mesh
        rcv_offsets = range(self.num_source_nodes, self.num_source_nodes + self.num_receiver_nodes)
        rcv_edges, rcv_edge_attrs = self._connect_sensors_to_mesh(
            sensor_indices=rcv_offsets,
            sensor_positions=receivers_positions, sensor_type='R'
        )

        # Combine all edges
        edges_all = np.concatenate((mesh_edges, src_edges, rcv_edges), axis=0)
        edge_attrs_all = np.concatenate((mesh_edge_attrs, src_edge_attrs, rcv_edge_attrs), axis=0)
  
        # Remove duplicates & unify
        structured = np.zeros((edges_all.shape[0], 3))  # (i, j, distance)
        structured[:, 0:2] = edges_all
        structured[:, 2] = edge_attrs_all[:, 0]

        unique_set = set()
        unique_list = []
        for row in structured:
            i, j, dist = row
            i, j = int(i), int(j)
            dist_rounded = round(dist, 6)
            if (i, j, dist_rounded) not in unique_set:
                unique_set.add((i, j, dist_rounded))
                unique_list.append([i, j, dist_rounded])

        unique_arr = np.array(unique_list)

        #  add reverse unless they exist
        total_sensor_nodes = self.num_source_nodes +self.num_receiver_nodes
        if self.bidirectional:
            reverse_list = []
            for e in unique_arr:
                i, j, dist = e
                if i>total_sensor_nodes and j> total_sensor_nodes and (j, i, dist) not in unique_set:
                    reverse_list.append([j, i, dist])
            if len(reverse_list) > 0:
                reverse_arr = np.array(reverse_list)
                unique_arr = np.concatenate((unique_arr, reverse_arr), axis=0)

        # separate edges and attributes
        edges = unique_arr[:, 0:2].astype(np.int64)
        edge_attrs = unique_arr[:, 2].reshape(-1, 1).astype(np.float32)

        self.global_edges = edges  # shape => (E, 2)
        self.global_edge_attrs = edge_attrs  # shape => (E, 1)


    def get_graph(self, tof_matrix, selected_sources, device):
        """
        Generator function that yields node feature matrices (x_init) for each source i.
        Args:
            tof_matrix: (S, R) tensor or array of measured ToF values (S sources, R receivers),
                        tof_matrix[i, j] = measured time-of-flight from source i to receiver j.
            selected_sources:  the indexes of sources the graph will be based on
            device: GPU / CPU
        Yields:
            A tuple (i, x_init) for each source i:
             - i is the source index (0 <= i < self.num_source_nodes).
             - x_init is a tensor of shape (num_total_nodes, 2), where:
                 x_init[:, 0] = T layer (time-of-flight)
                 x_init[:, 1] = c layer (speed of sound)
               with the following layout:
                 - x_init[0..S-1, 0]   = inf for all sources except selected one (T=0)
                 - x_init[S..S+R-1, 0] = inf for all receivers (will be computed by solver)
                 - x_init[S+R..end, 0] = inf for mesh nodes
                 - x_init[:, 1]        = c_init for all nodes
        """
        if not self.initialized:
            raise ValueError("GraphDataset not built yet!")

        if not isinstance(tof_matrix, torch.Tensor):
            tof_matrix = torch.tensor(tof_matrix, dtype=torch.float32)

        total_nodes = self.num_source_nodes + self.num_receiver_nodes + self.num_mesh_nodes

        pos_tensor = torch.tensor(self.positions, dtype=torch.float32, device=device)
        edge_index = torch.tensor(self.global_edges, dtype=torch.long, device=device).T  # (2, E)
        edge_attr = torch.tensor(self.global_edge_attrs, dtype=torch.float32, device=device)

        for i in selected_sources:
            # node features => (N, 2)
            x_init = torch.full((total_nodes, 2), float('inf'), dtype=torch.float32, device=device)
            x_init[:, 1] = self.c_init  # fill c

            # Set T=0 only for the current source, all others remain inf
            x_init[i, 0] = 0.0

            data = Data(
                x=x_init,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos_tensor
            )
 
            yield i, data

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
        edge_attrs = []
        mesh_start = self.num_sensor_nodes
        mesh_positions = self.positions[mesh_start:]

        tree = cKDTree(mesh_positions)
        for i in range(self.num_mesh_nodes):
            pos_i = mesh_positions[i]
            dists, nbr_indices = tree.query(pos_i, k=self.mesh_node_k)
            i_global = i + mesh_start
            for dist, nbr in zip(dists, nbr_indices):
                if i == nbr:
                    continue
                j_global = nbr + mesh_start
                edges.append((i_global, j_global))
                edge_attrs.append([dist])

        return np.array(edges, dtype=np.int64), np.array(edge_attrs, dtype=np.float32)


    def _connect_sensors_to_mesh(self, sensor_indices, sensor_positions, sensor_type):
        edges = []
        edge_attrs = []

        mesh_start = self.num_sensor_nodes
        mesh_positions = self.positions[mesh_start:]
        tree = cKDTree(mesh_positions)

        for local_idx, s_idx_global in enumerate(sensor_indices):
            s_pos = sensor_positions[local_idx]
            dists, nbrs = tree.query(s_pos, k=self.sensor_k)
            if self.sensor_k == 1:
                dists = [dists]
                nbrs = [nbrs]

            for dist, nbr in zip(dists, nbrs):
                j_global = nbr + mesh_start
                if s_idx_global == j_global:
                    continue

                # For sources: source -> mesh (wave propagates from source to mesh)
                # For receivers: receiver <- mesh (wave propagates from mesh to receiver)
                if sensor_type == 'S':
                    edges.append((s_idx_global, j_global))  # source -> mesh
                else:  # sensor_type == 'R'
                    edges.append((j_global, s_idx_global))  # mesh -> receiver
                edge_attrs.append([dist])

        return np.array(edges, dtype=np.int64), np.array(edge_attrs, dtype=np.float32)

