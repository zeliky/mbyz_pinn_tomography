"""FMM-style propagation step: T_i = min_j (T_j + dist_ij / c_j)."""

import torch
from torch_geometric.nn import MessagePassing


class FMMPropagation(MessagePassing):
    """Stateless FMM step. SoS (c) comes from state, not from this module."""

    def __init__(self, flow: str = "source_to_target") -> None:
        super().__init__(aggr="min", node_dim=0, flow=flow)

    def forward(
        self,
        T: torch.Tensor,
        c: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One FMM update. T, c: (N,); edge_index (2,E); pos (N,2); edge_attr (E,1) optional."""
        if edge_attr is not None:
            dist = edge_attr.squeeze(-1)
        else:
            row, col = edge_index[0], edge_index[1]
            dist = (pos[row] - pos[col]).norm(dim=1)
        return self.propagate(edge_index, T=T, c=c, dist=dist, size=(T.size(0), T.size(0)))

    def message(self, T_j: torch.Tensor, c_j: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(T_j)
        dt = dist / (c_j + 1e-8)
        result = torch.where(
            mask,
            T_j + dt,
            torch.full_like(T_j, float("inf"), device=T_j.device),
        )
        return result

    def update(
        self,
        aggr_out: torch.Tensor,
        T: torch.Tensor,
        c: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        return torch.minimum(aggr_out, T)
