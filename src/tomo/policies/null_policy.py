"""NullPolicy: returns zero delta (operator-only baseline)."""

from typing import Any

import torch
from torch_geometric.data import Data

from tomo.policies.base import Policy
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class NullPolicy(Policy):
    """Returns zero delta_c; used for operator-only baseline."""

    def __call__(
        self,
        state: SoSState,
        observation: Observation,
        tof_error: torch.Tensor,
        observation_graph: Data,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return torch.zeros_like(state.c_values, device=state.c_values.device)
