import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, MessagePassing
from scipy.interpolate import griddata
from py2mat.msfm2d import msfm2d
from logger import visualize_matdata
class WaveSolver:

    def __init__(self, num_iterations=5):
        self.num_iterations = num_iterations

    def simulate_T(self, data, src_id):
        """
        Compute time-of-flight values for receiver and mesh nodes using msfm2d.

        Parameters:
        - data: PyG Data object containing:
            - data.x[:, 0] => initial T values
            - data.x[:, 1] => c (speed of sound)
            - data.pos => positions (N, 2) in real coordinates (128x128 domain)
        - src_id: Index of the source node

        Returns:
        - T: Tensor of time-of-flight values for all nodes
        """
        # Extract positions and speed of sound
        positions = data.pos.cpu().numpy()   
        c = data.x[:, 1].cpu().numpy()  # Speed of sound

        # Get source position
        source_pos = positions[src_id]

        # Create a grid for the speed function
        # Use the actual domain size (128x128)
        x_min, y_min = 0, 0
        x_max, y_max = 127, 127

        # Create a fine grid for the speed function
        grid_size = 128  # Match the domain size for better accuracy
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        F = griddata(positions, c, (X, Y), method='nearest', fill_value=1.2)
        visualize_matdata(F, 'mesh cmap')

        # Convert source position to grid coordinates
        source_grid = np.array([int(source_pos[0]),int(source_pos[1])]).reshape(1, 2)

        # Run msfm2d
        T_grid = msfm2d(F, source_grid)
        visualize_matdata(T_grid, 'T map')

        # convert back to the graph nodes structure (each node will get the T value at its position)
        T = []
        for x, y in positions:
            x_idx = int(x)
            y_idx = int(y)
            T.append(T_grid[y_idx, x_idx])

        # Convert back to tensor
        T = torch.tensor(T, dtype=torch.float32, device=data.x.device)

        return T


    def BAK_simulate_T(self, data: Data, src_id):
        """
        data: a PyG Data object from GraphDataset.get_graph(), containing:
            - data.x[:, 0] => initial T values
            - data.x[:, 1] => c (speed of sound)
            - data.edge_index => (2, E)
            - data.edge_attr => distances as (E, 1)
            - data.pos => positions (N, 2)

        Returns:
            T (Tensor): The final time-of-flight estimates for each node, shape (N,).
        """
        # Extract relevant fields
        x_init = data.x.clone()  # shape [N, 2]
        c_init = x_init[:, 1]
        N = data.num_nodes
        T = torch.full((N,), float('inf'), device=c_init.device)
        
        # Set T=0 only for the selected source, all other sources remain inf
        T[src_id] = 0.0

        edge_index = data.edge_index  # [2, E]
        edge_attr = data.edge_attr  # [E, 1] => distances
        pos = data.pos  # [N, 2]
        mpnn = FMMMessagePassing()
        
        #print(f"Initial T values for source {src_id}:")
        #print(f"Source : {src_id}")
        #print(f"Source T: {T[src_id]}")
        #print(f"All sources T values: {T[0:32]}")  # Print all source T values
        #print(f"Number of finite T values: {(T != float('inf')).sum().item()}")

        for i in range(self.num_iterations):
            T_old = T.clone()
            T = mpnn(T, c_init, pos, edge_index)
            
            # Debug information
            num_finite = (T != float('inf')).sum().item()
            #print(f"\nIteration {i+1}:")
            #print(f"Number of finite T values: {num_finite}")
            #print(f"T on sources: {T[0:32]}")  # Print all source T values
            #print(f"T on receivers: {T[32:64]}")
            #print(f"Min T value: {T.min().item()}")
            #print(f"Max finite T value: {T[T != float('inf')].max().item() if num_finite > 0 else 'N/A'}")
            
            # Check if T values have converged
            if torch.allclose(T, T_old, rtol=1e-5):
                print(f"Converged after {i+1} iterations")
                break

        return T

    def validate_receivers(self, T, data):
        """
        Validate the final T against known ToF at receivers.
        By convention:
            - source nodes are [0..num_source_nodes-1]
            - receiver nodes are [num_source_nodes..num_source_nodes+num_receiver_nodes-1]
            - mesh nodes are the remainder.

        We compare the predicted T vs. the initial T (i.e., data.x) on receiver nodes.
        Returns a mean absolute error over the receivers.
        """
        # For demonstration, let's assume:
        #   0..31 => sources
        #   32..63 => receivers
        # You can change these constants or pass them in from outside.
        receiver_start = 32
        receiver_end = 64

        predicted = T[receiver_start:receiver_end]
        ground_truth = data.x[receiver_start:receiver_end, 0]
        mae = (predicted - ground_truth).abs().mean()
        return mae.item()

    def check_eikonal_residual(self, T, data):
        """
        Optionally compute a residual measuring how well T satisfies
        T_i = min_j (T_j + dist_ij / c_j).
        We'll do one message-passing step and measure the difference.
        """
        mpnn = FMMMessagePassing()
        T_updated = mpnn(T, data.x[:, 1], data.pos, data.edge_index, data.edge_attr)
        residual = (T_updated - T).abs().mean()
        return residual.item()



class FMMMessagePassing(MessagePassing):
    def __init__(self, aggr='min'):
        # Set flow to source_to_target to ensure proper direction
        super().__init__(aggr=aggr, node_dim=0, flow='source_to_target')

    def forward(self, T, c, pos, edge_index):        
        """
        T: (N,) time-of-flight
        c: (N,) speed of sound
        pos: (N, 2) node positions
        edge_index: (2, E)
        """
        row, col = edge_index
        # Calculate distances between connected nodes
        dist = (pos[row] - pos[col]).norm(dim=1)

        # Debug: Check if we have any finite T values
        num_finite = (T != float('inf')).sum().item()
        if num_finite == 0:
            print("Warning: No finite T values in forward pass")

        return self.propagate(
            edge_index=edge_index,
            T=T,
            c=c,
            dist=dist,
            size=(T.size(0), T.size(0))  # Specify the full size of the graph
        )

    def message(self, T_j, c_j, dist):
        """
        Eikonal update using quadratic scheme from Fast Marching Method.
        For each node, we solve the quadratic equation:
        (∂T/∂x)² + (∂T/∂y)² = 1/c²
        """
        # Only propagate if T_j is finite
        mask = torch.isfinite(T_j)
        
        # Calculate time increment based on distance and speed
        dt = dist / (c_j + 1e-16)
        
        # Add the base time from the source node
        result = torch.where(
            mask,
            T_j + dt,  # Simple first-order approximation
            torch.tensor(float('inf'), device=T_j.device)
        )
        
        return result

    def update(self, aggr_out, T):
        """
        Update rule: Take minimum of all incoming messages
        This implements the min operation in the Eikonal equation
        """
        # Keep the original T value if it's smaller than the minimum of incoming messages
        return torch.minimum(aggr_out, T)
