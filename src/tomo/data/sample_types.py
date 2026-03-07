"""Sample and batch type definitions.

One graph schema used everywhere (operator and RL):
- PyG Data with: x, edge_index, pos, edge_attr (distances [E, 1]).
- Node layout: [sources, receivers, mesh].
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Data


@dataclass
class TomographySample:
    """Single sample: observation + graph for the system."""

    observation: Any  # Observation
    graph: Data  # PyG Data with edge_index, pos, edge_attr (distances)
    # Optional: ground truth SoS for supervised loss
    target_sos: torch.Tensor | None = None
    target_tof: torch.Tensor | None = None


def graph_schema_description() -> str:
    """Return the canonical graph schema for docs."""
    return """
    Graph schema (PyG Data):
    - x: node features, shape (N, F) — e.g. [c_init, tof_mean, tof_std, role, x_norm, y_norm]
    - edge_index: (2, E) long
    - pos: (N, 2) node positions
    - edge_attr: (E, 1) edge distances (required for operator/RL)
    Node order: [sources, receivers, mesh].
    """
