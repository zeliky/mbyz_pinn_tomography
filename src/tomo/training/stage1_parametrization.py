"""Stage 1 parametrization: alpha (1A) and alpha + smooth correction (1B)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def params_to_c_map(
    params: np.ndarray,
    c_base: float,
    delta_c0: np.ndarray,
    stage: str,
    *,
    grid_h: int = 8,
    grid_w: int = 8,
    target_h: int = 128,
    target_w: int = 128,
    apply_gaussian: bool = False,
    sigma: float = 0.5,
) -> np.ndarray:
    """Convert search parameters to c_map.

    Stage 1A: params = [alpha], c_map = c_base + alpha * delta_c0.
    Stage 1B: params = [alpha, z_0, ..., z_{n-1}], smooth_correction from z grid.

    Args:
        params: [alpha] for stage1a, [alpha, z...] for stage1b.
        c_base: Fluid baseline speed.
        delta_c0: (H, W) residual from Stage 0 (c0_physical - c_base).
        stage: "stage1a" or "stage1b".
        grid_h, grid_w: Resolution of smooth correction grid (stage1b).
        target_h, target_w: Output c_map shape.
        apply_gaussian: Apply Gaussian blur to smooth_correction.
        sigma: Gaussian sigma for blur.

    Returns:
        (target_h, target_w) c_map.
    """
    params = np.asarray(params, dtype=np.float64)
    delta_c0 = np.asarray(delta_c0, dtype=np.float64)

    if delta_c0.ndim != 2:
        raise ValueError(f"delta_c0 must be 2D, got shape {delta_c0.shape}")

    if delta_c0.shape != (target_h, target_w):
        from scipy.ndimage import zoom

        zoom_factors = (target_h / delta_c0.shape[0], target_w / delta_c0.shape[1])
        delta_c0 = zoom(delta_c0, zoom_factors, order=1)

    alpha = float(params[0])
    c_map = c_base + alpha * delta_c0

    if stage == "stage1b" and len(params) > 1:
        n_z = grid_h * grid_w
        z = params[1 : 1 + n_z]
        if len(z) < n_z:
            z = np.pad(z, (0, n_z - len(z)), constant_values=0)
        z = z[:n_z].reshape(grid_h, grid_w)

        from scipy.ndimage import zoom

        zoom_factors = (target_h / grid_h, target_w / grid_w)
        smooth_correction = zoom(z, zoom_factors, order=1)

        if apply_gaussian:
            smooth_correction = gaussian_filter(smooth_correction, sigma=sigma, mode="nearest")

        c_map = c_map + smooth_correction

    return c_map


def get_bounds(
    stage: str,
    alpha_min: float,
    alpha_max: float,
    *,
    grid_h: int = 8,
    grid_w: int = 8,
    z_min: float = -50.0,
    z_max: float = 50.0,
) -> list[tuple[float, float]]:
    """Get (min, max) bounds for each parameter.

    Stage 1A: [(alpha_min, alpha_max)].
    Stage 1B: [(alpha_min, alpha_max), (z_min, z_max) x n_z].
    """
    bounds: list[tuple[float, float]] = [(alpha_min, alpha_max)]
    if stage == "stage1b":
        n_z = grid_h * grid_w
        bounds.extend([(z_min, z_max)] * n_z)
    return bounds


def params_to_smooth_correction(
    params: np.ndarray,
    *,
    grid_h: int = 8,
    grid_w: int = 8,
    target_h: int = 128,
    target_w: int = 128,
    apply_gaussian: bool = False,
    sigma: float = 0.5,
) -> np.ndarray:
    """Extract smooth correction term from params (Stage 1B only)."""
    if len(params) <= 1:
        return np.zeros((target_h, target_w), dtype=np.float64)
    n_z = grid_h * grid_w
    z = params[1 : 1 + n_z]
    if len(z) < n_z:
        z = np.pad(z, (0, n_z - len(z)), constant_values=0)
    z = z[:n_z].reshape(grid_h, grid_w)
    from scipy.ndimage import zoom

    zoom_factors = (target_h / grid_h, target_w / grid_w)
    smooth = zoom(z, zoom_factors, order=1)
    if apply_gaussian:
        smooth = gaussian_filter(smooth, sigma=sigma, mode="nearest")
    return smooth


def get_initial_params(
    stage: str,
    alpha_init: float = 1.0,
    *,
    grid_h: int = 8,
    grid_w: int = 8,
    init_scale: float = 0.0,
) -> np.ndarray:
    """Get initial parameter vector for search."""
    if stage == "stage1a":
        return np.array([alpha_init], dtype=np.float64)
    n_z = grid_h * grid_w
    z_init = np.full(n_z, init_scale, dtype=np.float64)
    return np.concatenate([[alpha_init], z_init])
