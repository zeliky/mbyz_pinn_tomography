"""Observation: ToF and layout used by initializer and operator."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Observation:
    """ToF observation and optional sensor/mesh layout references.

    Used by initializer and operator; does not hold state.
    """

    tof_observed: torch.Tensor  # e.g. (num_sources, num_receivers) or flattened
    layout_metadata: dict[str, Any] | None = None  # sources_positions, receivers_positions, etc.
