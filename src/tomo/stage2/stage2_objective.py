"""Stage 2 reward / loss components.

R = -L_tof - lambda_smooth * L_smooth - lambda_bounds * L_bounds - lambda_step * L_step
"""

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
    """Charbonnier loss on flattened residuals."""
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


def stage2_reward(
    tof_pred_new: np.ndarray,
    raw_tof_obs: np.ndarray,
    c_new: np.ndarray,
    delta_c_policy: np.ndarray,
    *,
    c_min: float = 1.45,
    c_max: float = 1.8,
    loss_type: str = "huber",
    delta: float = 0.01,
    lambda_smooth: float = 1.0,
    lambda_bounds: float = 10.0,
    lambda_step: float = 0.1,
) -> float:
    """Compute Stage 2 reward (negative loss).

    R = -L_tof - lambda_smooth * L_smooth - lambda_bounds * L_bounds - lambda_step * L_step
    """
    tof_pred_new = np.asarray(tof_pred_new, dtype=np.float64)
    raw_tof_obs = np.asarray(raw_tof_obs, dtype=np.float64)
    c_new = np.asarray(c_new, dtype=np.float64)
    delta_c_policy = np.asarray(delta_c_policy, dtype=np.float64)

    residuals = tof_pred_new - raw_tof_obs
    if loss_type == "huber":
        loss_tof = _huber_loss(residuals, delta)
    elif loss_type == "charbonnier":
        loss_tof = _charbonnier_loss(residuals, delta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss_smooth = 0.0
    if lambda_smooth > 0:
        lap = _discrete_laplacian_2d(delta_c_policy)
        loss_smooth = float((lap**2).sum())

    loss_bounds = 0.0
    if lambda_bounds > 0:
        below = np.minimum(c_new - c_min, 0)
        above = np.maximum(c_new - c_max, 0)
        loss_bounds = float((below**2).sum() + (above**2).sum())

    loss_step = 0.0
    if lambda_step > 0:
        loss_step = float((delta_c_policy**2).sum())

    total_loss = (
        loss_tof
        + lambda_smooth * loss_smooth
        + lambda_bounds * loss_bounds
        + lambda_step * loss_step
    )
    return -total_loss
