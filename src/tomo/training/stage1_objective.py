"""Stage 1 objective: robust ToF loss + smoothness + bounds + anchor."""

from __future__ import annotations

import numpy as np


def _huber_loss(residuals: np.ndarray, delta: float) -> float:
    """Huber loss on flattened residuals."""
    r = np.asarray(residuals).ravel()
    abs_r = np.abs(r)
    quadratic = np.minimum(abs_r, delta)
    linear = abs_r - quadratic
    return float(0.5 * (quadratic**2).sum() + delta * linear.sum())


def _charbonnier_loss(residuals: np.ndarray, delta: float) -> float:
    """Charbonnier loss: sqrt((r/delta)^2 + 1) - 1, scaled by delta^2."""
    r = np.asarray(residuals).ravel()
    scaled = r / (delta + 1e-12)
    return float((delta**2) * (np.sqrt(scaled**2 + 1) - 1).sum())


def _discrete_laplacian_2d(grid: np.ndarray) -> np.ndarray:
    """Discrete Laplacian (5-point stencil) of 2D grid."""
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {g.shape}")
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = (
        g[0:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, 0:-2]
        + g[1:-1, 2:]
        - 4 * g[1:-1, 1:-1]
    )
    return lap


def stage1_objective(
    tof_pred: np.ndarray,
    raw_tof: np.ndarray,
    c_map: np.ndarray,
    c0: np.ndarray,
    c_min: float,
    c_max: float,
    *,
    smooth_correction: np.ndarray | None = None,
    w_tof: float = 1.0,
    w_smooth: float = 0.0,
    w_bounds: float = 0.0,
    w_anchor: float = 0.0,
    loss_type: str = "huber",
    delta: float = 0.01,
) -> float:
    """Compute Stage 1 objective (scalar loss).

    Args:
        tof_pred: (S, R) predicted ToF.
        raw_tof: (S, R) observed ToF.
        c_map: (H, W) current speed-of-sound map.
        c0: (H, W) Stage 0 prior (frozen).
        c_min: Lower bound for c_map.
        c_max: Upper bound for c_map.
        smooth_correction: (H, W) optional correction term for smoothness penalty.
        w_tof: Weight for ToF loss.
        w_smooth: Weight for smoothness penalty on smooth_correction.
        w_bounds: Weight for bounds violation penalty.
        w_anchor: Weight for anchor penalty (c_map close to c0).
        loss_type: "huber" or "charbonnier".
        delta: Robustness parameter for ToF loss.

    Returns:
        Scalar total loss.
    """
    tof_pred = np.asarray(tof_pred, dtype=np.float64)
    raw_tof = np.asarray(raw_tof, dtype=np.float64)
    c_map = np.asarray(c_map, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)

    residuals = tof_pred - raw_tof
    if loss_type == "huber":
        loss_tof = _huber_loss(residuals, delta)
    elif loss_type == "charbonnier":
        loss_tof = _charbonnier_loss(residuals, delta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'huber' or 'charbonnier'.")

    loss_smooth = 0.0
    if w_smooth > 0 and smooth_correction is not None:
        lap = _discrete_laplacian_2d(smooth_correction)
        loss_smooth = float((lap**2).sum())

    loss_bounds = 0.0
    if w_bounds > 0:
        below = np.minimum(c_map - c_min, 0)
        above = np.maximum(c_map - c_max, 0)
        loss_bounds = float((below**2).sum() + (above**2).sum())

    loss_anchor = 0.0
    if w_anchor > 0:
        diff = c_map - c0
        loss_anchor = float((diff**2).sum())

    total = (
        w_tof * loss_tof
        + w_smooth * loss_smooth
        + w_bounds * loss_bounds
        + w_anchor * loss_anchor
    )
    return float(total)
