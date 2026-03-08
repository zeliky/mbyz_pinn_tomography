"""SoSState: the only persistent object across the pipeline."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SoSState:
    """State holds c_values (speed-of-sound field) and step index.

    Everything revolves around this object; no model bypasses it.
    c_map_2d, when set, is the (H, W) map; c_values is the flattened view for graph-based stages.
    """

    c_values: torch.Tensor  # shape (num_nodes,) or (H, W) depending on representation
    step_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    c_map_2d: torch.Tensor | None = None  # (H, W) map when available; avoid flattening in initializer

    def apply(self, delta: torch.Tensor) -> "SoSState":
        """Return a new state with c_values updated by delta (same shape as c_values)."""
        new_c = self.c_values + delta
        return SoSState(
            c_values=new_c,
            step_idx=self.step_idx + 1,
            metadata={**self.metadata},
            c_map_2d=self.c_map_2d,
        )

    def to(self, device: torch.device) -> "SoSState":
        """Return a new state with tensors on the given device."""
        c_map_2d = self.c_map_2d.to(device) if self.c_map_2d is not None else None
        return SoSState(
            c_values=self.c_values.to(device),
            step_idx=self.step_idx,
            metadata={**self.metadata},
            c_map_2d=c_map_2d,
        )
