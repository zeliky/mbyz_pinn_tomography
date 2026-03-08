"""Explicit SoS unit conversion: physical <-> scaled [0, 1].

Physical SoS uses the same units as the data (e.g. m/s or mm/s). Scaled SoS
is linear in [min_sos, max_sos] <-> [0, 1]. All helpers support scalars and
arrays (NumPy or torch); min_sos and max_sos are scalars.
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None

ArrayLike = Union[float, np.ndarray, "torch.Tensor"]


def to_scaled_sos(
    physical_sos: ArrayLike,
    min_sos: float,
    max_sos: float,
    *,
    clamp: bool = True,
) -> ArrayLike:
    """Convert physical SoS to scaled [0, 1].

    scaled = (physical_sos - min_sos) / (max_sos - min_sos).
    min_sos and max_sos must be in the same physical units as the data.
    """
    _np = _as_numpy(physical_sos)
    if max_sos == min_sos:
        return _zeros_like(_np) if not clamp else _np
    out = (_np - min_sos) / (max_sos - min_sos)
    if clamp:
        out = np.clip(out, 0.0, 1.0)
    return _same_type(out, physical_sos)


def to_physical_sos(
    scaled_sos: ArrayLike,
    min_sos: float,
    max_sos: float,
) -> ArrayLike:
    """Convert scaled SoS [0, 1] to physical units.

    physical = min_sos + scaled_sos * (max_sos - min_sos).
    """
    _np = _as_numpy(scaled_sos)
    out = min_sos + _np * (max_sos - min_sos)
    return _same_type(out, scaled_sos)


def scaled_delta_to_physical(
    delta_scaled: ArrayLike,
    min_sos: float,
    max_sos: float,
) -> ArrayLike:
    """Convert a delta in scaled space to physical units.

    Deltas are differences; no min_sos offset: delta_phys = delta_scaled * (max_sos - min_sos).
    """
    _np = _as_numpy(delta_scaled)
    out = _np * (max_sos - min_sos)
    return _same_type(out, delta_scaled)


def physical_delta_to_scaled(
    delta_phys: ArrayLike,
    min_sos: float,
    max_sos: float,
) -> ArrayLike:
    """Convert a delta in physical units to scaled space.

    delta_scaled = delta_phys / (max_sos - min_sos).
    """
    _np = _as_numpy(delta_phys)
    if max_sos == min_sos:
        return _zeros_like(_np)
    out = _np / (max_sos - min_sos)
    return _same_type(out, delta_phys)


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=np.float64)


def _zeros_like(arr: np.ndarray) -> np.ndarray:
    return np.zeros_like(arr, dtype=np.float64)


def _same_type(out: np.ndarray, original: ArrayLike) -> ArrayLike:
    if isinstance(original, np.ndarray):
        return out.astype(original.dtype) if original.dtype != np.float64 else out
    if torch is not None and hasattr(original, "detach"):
        return torch.from_numpy(out).to(device=original.device, dtype=original.dtype)
    return float(out) if out.ndim == 0 else out
