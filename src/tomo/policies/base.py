"""Abstract policy interface."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch_geometric.data import Data

from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class Policy(ABC):
    """Policy proposes an update delta_c, not a final solution."""

    @abstractmethod
    def __call__(
        self,
        state: SoSState,
        observation: Observation,
        tof_error: torch.Tensor,
        observation_graph: Data,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Propose delta_c for state update.

        Args:
            state: Current SoS state.
            observation: ToF observation.
            tof_error: tof_pred - tof_observed (per node or per receiver).
            observation_graph: PyG Data for graph-based policy.
            context: Optional extra (e.g. priors).

        Returns:
            delta_c: Update for c_values, same shape as state.c_values.
        """
        ...
