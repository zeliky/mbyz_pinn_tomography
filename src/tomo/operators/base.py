"""Abstract operator interface."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch_geometric.data import Data

from tomo.state.sos_state import SoSState


class Operator(ABC):
    """Operator owns physics-inspired propagation. Signature: tof_pred = operator(state, observation_graph)."""

    @abstractmethod
    def __call__(
        self,
        state: SoSState,
        observation_graph: Data,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict ToF given current state and observation graph.

        Args:
            state: Current SoS state (c_values, step_idx).
            observation_graph: PyG Data with edge_index, pos, edge_attr (distances).

        Returns:
            tof_pred: Predicted time-of-flight per node, shape (num_nodes,).
        """
        ...
