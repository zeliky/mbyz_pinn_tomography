"""Edge feature computation (e.g. distances) for unified graph schema."""

import torch
from torch_geometric.data import Data


def compute_edge_distances(edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance for each edge. Returns (E, 1) for edge_attr."""
    row, col = edge_index[0], edge_index[1]
    dist = (pos[row] - pos[col]).norm(dim=1, keepdim=True)
    return dist.float()


def add_edge_attr_to_data(data: Data) -> Data:
    """Ensure data has edge_attr (distances). Mutates data if edge_attr missing."""
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        data.edge_attr = compute_edge_distances(data.edge_index, data.pos)
    return data
