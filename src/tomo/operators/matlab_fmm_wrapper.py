"""Forward-only wrapper around the C msfm2d FMM solver for ToF prediction.

Uses py2mat.msfm2d (C library, not MATLAB Engine). No autograd through the solver.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

try:
    from py2mat.msfm2d import msfm2d
except (ImportError, OSError) as e:
    msfm2d = None
    _IMPORT_ERROR = str(e)


def forward_tof(
    sos_map_phys: np.ndarray,
    x_s: np.ndarray,
    x_r: np.ndarray,
    *,
    scale_factor: float = 1.0,
    use_second: bool = True,
    use_cross: bool = True,
) -> np.ndarray:
    """Compute predicted ToF matrix using FMM (forward-only, no gradients).

    Contract: sos_map_phys is in physical SoS units (same as data). scale_factor
    applies only to the ToF output (to match observed ToF units); it does not
    scale or interpret SoS. The solver must not infer SoS scaling internally.

    Args:
        sos_map_phys: (H, W) speed of sound in physical units (m/s or same as data).
        x_s: (S, 2) source positions in grid coordinates (x, y) = (col, row).
        x_r: (R, 2) receiver positions in grid coordinates.
        scale_factor: ToF output scale factor: multiply FMM travel-time output
            by this to match observed ToF units. SoS must already be physical.
        use_second: Use second-order approximations in msfm2d.
        use_cross: Use cross derivatives in msfm2d.

    Returns:
        (S, R) predicted raw ToF matrix, same shape as observed raw_tof.
    """
    if msfm2d is None:
        raise RuntimeError(
            f"Failed to import py2mat.msfm2d: {_IMPORT_ERROR}. "
            "Ensure libmsfm2d.so (Linux) or msfm2d.dll (Windows) is available."
        ) from None

    sos_map = np.asarray(sos_map_phys, dtype=np.float64)
    x_s = np.asarray(x_s, dtype=np.float64)
    x_r = np.asarray(x_r, dtype=np.float64)

    if sos_map.ndim != 2:
        raise ValueError(f"sos_map must be 2D, got shape {sos_map.shape}")
    if x_s.ndim != 2 or x_s.shape[1] != 2:
        raise ValueError(f"x_s must be (S, 2), got shape {x_s.shape}")
    if x_r.ndim != 2 or x_r.shape[1] != 2:
        raise ValueError(f"x_r must be (R, 2), got shape {x_r.shape}")

    H, W = sos_map.shape
    S = x_s.shape[0]
    R = x_r.shape[0]

    tof_pred = np.zeros((S, R), dtype=np.float64)

    for s in range(S):
        x, y = float(x_s[s, 0]), float(x_s[s, 1])
        row = int(round(y))
        col = int(round(x))
        row = max(0, min(row, H - 1))
        col = max(0, min(col, W - 1))
        source_point = np.array([[row, col]], dtype=np.int32)

        T_grid = msfm2d(sos_map, source_point, use_second=use_second, use_cross=use_cross)

        n_inf = np.isinf(T_grid).sum()
        if n_inf > 0:
            logger.warning(
                "FMM returned %d inf values for source %d; replacing with large finite value",
                n_inf,
                s,
            )
            T_grid = np.where(np.isfinite(T_grid), T_grid, 1e10)

        if scale_factor != 1.0:
            T_grid = T_grid * scale_factor

        interp = RegularGridInterpolator(
            (np.arange(H), np.arange(W)),
            T_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        points = np.column_stack([x_r[:, 1], x_r[:, 0]])
        t_at_receivers = interp(points)
        invalid = np.isnan(t_at_receivers)
        if np.any(invalid):
            logger.warning(
                "Interpolation produced %d NaN values for source %d (receivers out of bounds)",
                invalid.sum(),
                s,
            )
            t_at_receivers[invalid] = 1e10
        tof_pred[s, :] = t_at_receivers

    return tof_pred
