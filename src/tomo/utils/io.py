"""IO and serialization helpers."""

from pathlib import Path

import torch

from tomo.state.sos_state import SoSState


def save_state(state: SoSState, path: str | Path) -> None:
    """Save SoSState to disk (c_values, step_idx, metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "c_values": state.c_values.cpu(),
            "step_idx": state.step_idx,
            "metadata": state.metadata,
        },
        path,
    )


def load_state(path: str | Path, device: torch.device | None = None) -> SoSState:
    """Load SoSState from disk."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    c = data["c_values"]
    if device is not None:
        c = c.to(device)
    return SoSState(
        c_values=c,
        step_idx=data.get("step_idx", 0),
        metadata=data.get("metadata", {}),
    )
