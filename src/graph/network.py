import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import cKDTree


class GraphDataset:
    def __init__(self, **kwargs):

        self.c_init = kwargs.get('c_init', 0.12)
        self.epsilon = 1e-8
        self.x_range = kwargs.get('x_range', (32, 96))
        self.y_range = kwargs.get('y_range', (32, 96))
        self.nx = kwargs.get('nx', 64)
        self.ny = kwargs.get('ny', 64)
        self.num_source_nodes = kwargs.get('num_source_nodes', 32)
        self.num_receiver_nodes = kwargs.get('num_receiver_nodes', 32)
        self.num_mesh_nodes = self.nx * self.ny
        self.num_sensor_nodes = self.num_source_nodes + self.num_receiver_nodes
        self.edges = []
        self.positions = []
        self.initialized = False

    def build(self, sources_positions, receivers_positions):
        mesh_positions, mesh_edges = self._create_interior_mesh()
        sources_to_mesh_edges = self._connect_sensors_to_mesh(sources_positions, mesh_positions,group='S' , k=32)
        mesh_to_receivers_edges = self._connect_sensors_to_mesh(receivers_positions, mesh_positions, group='R', k=32)

        self.edges = np.unique(np.concatenate((sources_to_mesh_edges, mesh_edges, mesh_to_receivers_edges)),  axis=0)
        self.positions = np.concatenate((sources_positions, receivers_positions, mesh_positions))
        self.initialized = True

    def get_graph(self, tof_matrix, selected_sources, device):
        """
        Generator function that yields node feature matrices (x_init) for each source i.
        Args:
            tof_matrix: (S, R) tensor or array of measured ToF values (S sources, R receivers),
                        tof_matrix[i, j] = measured time-of-flight from source i to receiver j.
            device: GPU / CPU
        Yields:
            A tuple (i, x_init) for each source i:
             - i is the source index (0 <= i < self.num_source_nodes).
             - x_init is a tensor of shape (num_total_nodes, 2), where:
                 x_init[:, 0] = T layer (time-of-flight)
                 x_init[:, 1] = c layer (speed of sound)
               with the following layout:
                 - x_init[0..S-1, 0]   = 0 for all sources in T-layer
                 - x_init[S..S+R-1, 0] = tof_matrix[i, :] for the active source i
                 - x_init[S+R..end, 0] = 0 for mesh nodes
                 - x_init[:, 1]        = c_init for all nodes
        """

        total_nodes = self.num_source_nodes + self.num_receiver_nodes + self.num_mesh_nodes

        # Ensure tof_matrix is a torch.Tensor for indexing
        if not isinstance(tof_matrix, torch.Tensor):
            tof_matrix = torch.tensor(tof_matrix, dtype=torch.float32)

        # For each source i, build a node feature matrix
        for i in selected_sources:
            source_node_idx = i
            #print(f"source:{source_node_idx}")
            # Initialize x_init: (N, 2) => (T, c)
            x_init = torch.zeros((total_nodes, 2), dtype=torch.float32)

            # Set the entire SoS layer (column=1) to c_init
            x_init[:, 1] = self.c_init
            #x_init[:, 0] = torch.norm(self.positions - self.positions[i], dim=1) + self.epsilon
            x_init[:, 0] = self.epsilon
            print(x_init[:, 0])
            exit()

            # The T layer (column=0) is zero by default for all nodes...
            # except for the receiver nodes for the *active* source i:
            # receiver nodes are in range [self.num_source_nodes .. self.num_source_nodes+self.num_receiver_nodes-1]
            for j in range(self.num_receiver_nodes):
                receiver_node_idx = self.num_source_nodes + j
                x_init[receiver_node_idx, 0] = tof_matrix[i, j].item()
                # This sets T = ToF(source i -> receiver j) on the j-th receiver node.

            # mark the source node with tof =1 (the rest are zeroes)
            x_init[source_node_idx, 0] = 1.0

            # The mesh nodes remain T=0; the sources remain T=0 as well.

            data = Data(
                x=x_init.to(device),  # (S + R + N, 2) node features
                edge_index=torch.tensor(self.edges, dtype=torch.long).T.to(device),  # shape (2, E)
                pos=torch.tensor(self.positions).to(device)
            )
            yield i, data

    def _create_interior_mesh(self):
        """
        Creates a 2D grid of interior (virtual) nodes using cKDTree-based k-NN connectivity.
        """
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.nx)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.ny)

        # Generate grid positions in [nx * ny, 2]
        positions = []
        for i in range(self.nx):
            for j in range(self.ny):
                positions.append((x_values[i], y_values[j]))
        positions = np.array(positions, dtype=np.float32)

        k = 5
        tree = cKDTree(positions)
        edges = []

        for i, pos in enumerate(positions):
            dists, indices = tree.query(pos, k=k)
            # For each neighbor, create bidirectional edges
            for nbr in indices:
                if nbr != i:
                    edges.append((i + self.num_sensor_nodes, nbr + self.num_sensor_nodes))
                    edges.append((nbr + self.num_sensor_nodes, i + self.num_sensor_nodes))

        edges = np.array(edges, dtype=np.int64)
        return positions, edges




    def _connect_sensors_to_mesh(self, sensor_positions, mesh_positions, group, k=1):
        """
        Creates edges from each sensor to its k nearest mesh nodes.

        Args:
            sensor_positions: (S, 2) array of sensor (x, y) positions
            mesh_positions: (N, 2) array of mesh node positions
            group : S:'sources'/ R:'receivers'
            k: Number of nearest neighbors in the mesh to connect each sensor with

        Returns:
            sensor_to_mesh_edges: List of (sensor_idx, mesh_idx) edges
        """

        tree = cKDTree(mesh_positions)
        sensor_to_mesh_edges = []

        for s_idx, spos in enumerate(sensor_positions):
            # Find k nearest neighbors in the mesh
            dists, indices = tree.query(spos, k=k)
            if k == 1:
                indices = [indices]  # ensure iterable
            for idx in indices:
                offset = 0 if group=='S' else self.num_source_nodes
                sensor_to_mesh_edges.append((s_idx+offset, idx + self.num_sensor_nodes))  # sensor -> mesh
                sensor_to_mesh_edges.append((idx + self.num_sensor_nodes, s_idx+offset))  # mesh -> sensor


        return sensor_to_mesh_edges

