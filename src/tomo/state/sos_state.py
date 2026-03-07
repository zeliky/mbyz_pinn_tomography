"""SoSState: the only persistent object across the pipeline."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SoSState:
    """State holds c_values (speed-of-sound field) and step index.

    Everything revolves around this object; no model bypasses it.
    """

    c_values: torch.Tensor  # shape (num_nodes,) or (H, W) depending on representation
    step_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, delta: torch.Tensor) -> "SoSState":
        """Return a new state with c_values updated by delta (same shape as c_values)."""
        new_c = self.c_values + delta
        return SoSState(
            c_values=new_c,
            step_idx=self.step_idx + 1,
            metadata={**self.metadata},
        )

    def to(self, device: torch.device) -> "SoSState":
        """Return a new state with tensors on the given device."""
        return SoSState(
            c_values=self.c_values.to(device),
            step_idx=self.step_idx,
            metadata={**self.metadata},
        )
