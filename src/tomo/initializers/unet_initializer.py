"""UNet initializer: learned TOF->SoS map, outputs SoSState only. Stub for full port."""

import torch
import torch.nn as nn

from tomo.initializers.base import Initializer
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class UNetInitializer(Initializer):
    """Learned initializer: TOF input -> c0 grid. Outputs SoSState; no final prediction API.

    Stub: returns constant c0 over num_nodes. Full port from tof_to_sos_net in Phase 5.
    """

    def __init__(
        self,
        num_nodes: int = 64,
        c0_fallback: float = 1.5,
        in_channels: int = 32,
        device: torch.device | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.c0_fallback = c0_fallback
        self._device = device
        # Placeholder for future UNet: self.net = ...
        self._net: nn.Module | None = None

    def __call__(self, observation: Observation) -> SoSState:
        if self._net is not None:
            # tof_observed -> net -> c0 grid -> flatten to c_values
            raise NotImplementedError("UNet forward not yet ported")
        device = self._device or observation.tof_observed.device
        c_values = torch.full(
            (self.num_nodes,), self.c0_fallback, device=device, dtype=observation.tof_observed.dtype
        )
        return SoSState(c_values=c_values, step_idx=0)
