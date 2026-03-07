"""GAT/FMM-style operator: stateless forward, SoS from state."""

from typing import Any

import torch
from torch_geometric.data import Data

from tomo.operators.base import Operator
from tomo.operators.propagation import FMMPropagation
from tomo.state.sos_state import SoSState


class GATFMMOperator(Operator):
    """FMM-style propagation: tof_pred = operator(state, observation_graph).

    Uses state.c_values as speed-of-sound; initial T from graph.x[:, 0] if present,
    else inf (sources must be set by caller via graph.x or convention).
    """

    def __init__(
        self,
        num_fmm_iterations: int = 3,
        min_sos: float = 1.2,
        max_sos: float = 2.5,
    ) -> None:
        self.num_fmm_iterations = num_fmm_iterations
        self.min_sos = min_sos
        self.max_sos = max_sos
        self._prop = FMMPropagation()

    def __call__(
        self,
        state: SoSState,
        observation_graph: Data,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict ToF via FMM iterations. c from state; T_init from graph.x[:, 0] or inf."""
        num_nodes = state.c_values.shape[0]
        device = state.c_values.device
        # c in [min_sos, max_sos]
        c = self.min_sos + (self.max_sos - self.min_sos) * torch.sigmoid(state.c_values)

        if observation_graph.x is not None and observation_graph.x.shape[1] >= 1:
            T = observation_graph.x[:, 0].to(device).clone()
            # Replace non-finite with inf for FMM
            T = torch.where(torch.isfinite(T), T, torch.full_like(T, float("inf"), device=device))
        else:
            T = torch.full((num_nodes,), float("inf"), device=device, dtype=c.dtype)

        edge_index = observation_graph.edge_index.to(device)
        pos = observation_graph.pos.to(device)
        edge_attr = getattr(observation_graph, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        for _ in range(self.num_fmm_iterations):
            T = self._prop(T, c, edge_index, pos, edge_attr)

        return T
