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

    def simulate_T(self, F, source_pos):
        """
        Compute time-of-flight values using msfm2d on the full mesh.

        Parameters:
        - F: Full mesh tensor (128x128) containing speed of sound values
        - source_pos: Source position (x, y) in real coordinates

        Returns:
        - T_grid: 2D tensor of time-of-flight values for the entire mesh
        """
        # Convert source position to grid coordinates
        source_grid = np.array([int(source_pos[0]), int(source_pos[1])]).reshape(1, 2)

        # Run msfm2d on the full mesh
        T_grid = msfm2d(F.detach().cpu().numpy(), source_grid)
        visualize_matdata(T_grid, 'T map')

        # Convert back to tensor and maintain gradients
        T_tensor = torch.tensor(T_grid, dtype=torch.float32, device=F.device)
        
        # Create a differentiable version of T_grid
        T_diff = T_tensor.clone().detach().requires_grad_(True)
        
        # Compute gradients through interpolation
        dx, dy = torch.gradient(T_diff, spacing=(1.0, 1.0))
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        
        # Eikonal equation: |∇T| = 1/c
        eikonal_residual = (grad_mag - 1.0 / F).pow(2).mean()
        
        # Add gradient information
        T_diff.register_hook(lambda grad: grad * (1.0 + eikonal_residual))
        
        return T_diff

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
        
        for i in range(self.num_iterations):
            T_old = T.clone()
            T = mpnn(T, c_init, pos, edge_index)
            
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
