"""Regularization terms."""

import torch


def l2_regularization(c: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
    """L2 penalty on c or on (c - ref)."""
    if ref is not None:
        return ((c - ref) ** 2).mean()
    return (c**2).mean()


def smoothness_regularization(c: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Penalize large differences between neighboring nodes (graph Laplacian style)."""
    row, col = edge_index[0], edge_index[1]
    diff = c[row] - c[col]
    return (diff**2).mean()
