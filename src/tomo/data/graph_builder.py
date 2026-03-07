"""Build PyG graph with edge_attr (distances) for operator/RL. One schema for all."""

import torch
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from tomo.operators.edge_features import add_edge_attr_to_data, compute_edge_distances


def build_mesh_edges(
    mesh_positions: np.ndarray,
    sensor_positions: np.ndarray,
    mesh_k: int = 9,
) -> np.ndarray:
    """Build edge list: mesh-mesh k-NN + sensor-mesh bidirectional. Returns (E, 2)."""
    num_sensors = sensor_positions.shape[0]
    num_mesh = mesh_positions.shape[0]
    tree = cKDTree(mesh_positions)
    edges = []
    for i in range(num_mesh):
        _, nbrs = tree.query(mesh_positions[i], k=min(mesh_k, num_mesh))
        nbrs = np.atleast_1d(nbrs)
        for j in nbrs:
            if i != j:
                i_global = num_sensors + i
                j_global = num_sensors + int(j)
                edges.append((i_global, j_global))
                edges.append((j_global, i_global))
        for s in range(num_sensors):
            edges.append((s, num_sensors + i))
            edges.append((num_sensors + i, s))
    return np.unique(np.array(edges, dtype=np.int64), axis=0)


def build_graph(
    sources_positions: np.ndarray,
    receivers_positions: np.ndarray,
    mesh_positions: np.ndarray,
    tof_matrix: torch.Tensor | np.ndarray | None = None,
    c_init: float = 1.5,
    device: torch.device | None = None,
    mesh_k: int = 9,
) -> Data:
    """Build one PyG Data with x, edge_index, pos, edge_attr (distances)."""
    positions = np.concatenate([sources_positions, receivers_positions, mesh_positions], axis=0)
    num_sources = sources_positions.shape[0]
    num_receivers = receivers_positions.shape[0]
    num_mesh = mesh_positions.shape[0]
    num_nodes = num_sources + num_receivers + num_mesh

    edges = build_mesh_edges(
        mesh_positions,
        np.concatenate([sources_positions, receivers_positions], axis=0),
        mesh_k=mesh_k,
    )
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)
    edge_attr = compute_edge_distances(edge_index, pos)

    if tof_matrix is not None:
        if isinstance(tof_matrix, np.ndarray):
            tof_matrix = torch.tensor(tof_matrix, dtype=torch.float32)
        tof_flat = tof_matrix.flatten()
        x_col0 = torch.full((num_nodes,), float("inf"))
        x_col0[: num_sources + num_receivers] = 0.0
        if tof_flat.numel() >= num_nodes:
            x_col0[: tof_flat.numel()] = tof_flat[: num_nodes]
    else:
        x_col0 = torch.full((num_nodes,), float("inf"))
        x_col0[:num_sources] = 0.0

    c_col = torch.full((num_nodes,), c_init, dtype=torch.float32)
    x = torch.stack([x_col0, c_col], dim=-1)
    if device is not None:
        x = x.to(device)
        edge_index = edge_index.to(device)
        pos = pos.to(device)
        edge_attr = edge_attr.to(device)
    data = Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, num_nodes=num_nodes)
    return data
