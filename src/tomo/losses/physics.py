"""Physics / eikonal regularization (optional)."""

import torch


def eikonal_residual_loss(
    tof: torch.Tensor,
    c: torch.Tensor,
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> torch.Tensor:
    """Residual: how well T satisfies T_i = min_j (T_j + dist_ij / c_j). Stub for Phase 4."""
    if edge_attr is not None:
        dist = edge_attr.squeeze(-1)
    else:
        row, col = edge_index[0], edge_index[1]
        dist = (pos[row] - pos[col]).norm(dim=1)
    row, col = edge_index[0], edge_index[1]
    T_j = tof[col]
    c_j = c[col].clamp(min=1e-6)
    T_updated = T_j + dist / c_j
    # Compare with current T at row nodes (simplified: mean residual)
    residual = (tof[row] - T_updated).clamp(min=0)  # only penalize when T < updated
    return residual.mean()
