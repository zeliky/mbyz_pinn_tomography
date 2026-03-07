"""Constant initializer: uniform or configurable c0."""

import torch

from tomo.initializers.base import Initializer
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class ConstantInitializer(Initializer):
    """Output SoSState with uniform c0. No learned parameters."""

    def __init__(
        self,
        c0: float = 1.5,
        num_nodes: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.c0 = c0
        self._num_nodes = num_nodes
        self._device = device

    def __call__(self, observation: Observation) -> SoSState:
        """Propose state with constant c. num_nodes from observation layout or fixed."""
        tof = observation.tof_observed
        if self._num_nodes is not None:
            num_nodes = self._num_nodes
        elif observation.layout_metadata and "num_nodes" in observation.layout_metadata:
            num_nodes = observation.layout_metadata["num_nodes"]
        else:
            # Fallback: use a default if observation doesn't specify (e.g. from graph later)
            num_nodes = tof.numel() if tof.dim() == 1 else 64
        device = self._device or tof.device
        c_values = torch.full((num_nodes,), self.c0, device=device, dtype=tof.dtype)
        return SoSState(c_values=c_values, step_idx=0)
