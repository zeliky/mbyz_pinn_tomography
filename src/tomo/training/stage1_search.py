"""Stage 1 derivative-free search: line search (1A), Powell/Nelder-Mead (1B)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from tomo.operators.matlab_fmm_wrapper import forward_tof
from tomo.training.stage1_objective import stage1_objective
from tomo.training.stage1_parametrization import (
    get_bounds,
    get_initial_params,
    params_to_c_map,
    params_to_smooth_correction,
)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_sample(
    batch: dict[str, Any],
    sample_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw_tof, x_s, x_r for a single sample."""
    raw_tof = batch["raw_tof"]
    if raw_tof.dim() == 3:
        raw_tof = raw_tof[sample_idx]
    else:
        raw_tof = raw_tof[sample_idx]
    raw_tof = _to_numpy(raw_tof)

    x_s = batch["x_s"][sample_idx]
    x_r = batch["x_r"][sample_idx]
    x_s = _to_numpy(x_s)
    x_r = _to_numpy(x_r)

    if x_s.ndim == 1:
        x_s = x_s.reshape(1, -1)
    if x_r.ndim == 1:
        x_r = x_r.reshape(1, -1)

    return raw_tof, x_s, x_r


def _ensure_grid_coords(
    x_s: np.ndarray,
    x_r: np.ndarray,
    grid_shape: tuple[int, int],
    coord_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert physical coords to grid indices if coord_range is given."""
    if coord_range is None:
        return x_s, x_r
    lo, hi = coord_range
    H, W = grid_shape
    scale = (W - 1) / (hi - lo) if hi != lo else 1.0
    x_s = (x_s - lo) * scale
    x_r = (x_r - lo) * scale
    x_s = np.clip(x_s, 0, W - 1)
    x_r = np.clip(x_r, 0, W - 1)
    return x_s, x_r


def run_stage1a_search(
    batch: dict[str, Any],
    delta_c0: np.ndarray,
    c0: np.ndarray,
    c_base: float,
    c_min: float,
    c_max: float,
    config: dict[str, Any],
    *,
    sample_idx: int = 0,
    scale_factor: float = 1.0,
    coord_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run Stage 1A: optimize alpha only via line search.

    Returns:
        best_alpha, best_loss, best_c_map, tof_pred, raw_tof.
    """
    raw_tof, x_s, x_r = _extract_sample(batch, sample_idx)
    H, W = delta_c0.shape
    x_s, x_r = _ensure_grid_coords(x_s, x_r, (H, W), coord_range)

    alpha_cfg = config.get("alpha_search", {})
    method = alpha_cfg.get("method", "line_search")
    if method == "grid_search":
        method = "line_search"
    alpha_min = alpha_cfg.get("min", 0.0)
    alpha_max = alpha_cfg.get("max", 2.0)
    num_steps = alpha_cfg.get("num_steps", 21)

    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "huber")
    delta = loss_cfg.get("delta", 0.01)
    w_tof = loss_cfg.get("w_tof", 1.0)
    w_bounds = loss_cfg.get("w_bounds", 10.0)
    w_anchor = loss_cfg.get("w_anchor", 0.1)

    best_alpha = alpha_min
    best_loss = float("inf")
    best_c_map = None
    best_tof_pred = None

    alphas = np.linspace(alpha_min, alpha_max, num_steps)
    for alpha in alphas:
        c_map = params_to_c_map(
            np.array([alpha]),
            c_base,
            delta_c0,
            "stage1a",
            target_h=H,
            target_w=W,
        )
        tof_pred = forward_tof(c_map, x_s, x_r, scale_factor=scale_factor)
        loss = stage1_objective(
            tof_pred,
            raw_tof,
            c_map,
            c0,
            c_min,
            c_max,
            w_tof=w_tof,
            w_bounds=w_bounds,
            w_anchor=w_anchor,
            loss_type=loss_type,
            delta=delta,
        )
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
            best_c_map = c_map.copy()
            best_tof_pred = tof_pred.copy()

    return {
        "best_alpha": best_alpha,
        "best_loss": best_loss,
        "best_c_map": best_c_map,
        "tof_pred": best_tof_pred,
        "raw_tof": raw_tof,
    }


def run_stage1b_search(
    batch: dict[str, Any],
    delta_c0: np.ndarray,
    c0: np.ndarray,
    c_base: float,
    c_min: float,
    c_max: float,
    config: dict[str, Any],
    *,
    stage1a_result: dict[str, Any] | None = None,
    sample_idx: int = 0,
    scale_factor: float = 1.0,
    coord_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run Stage 1B: optimize alpha + smooth correction via Powell/Nelder-Mead."""
    raw_tof, x_s, x_r = _extract_sample(batch, sample_idx)
    H, W = delta_c0.shape
    x_s, x_r = _ensure_grid_coords(x_s, x_r, (H, W), coord_range)

    smooth_cfg = config.get("smooth_correction", {})
    grid_h = smooth_cfg.get("grid_h", 8)
    grid_w = smooth_cfg.get("grid_w", 8)
    init_scale = smooth_cfg.get("init_scale", 0.0)
    apply_gaussian = smooth_cfg.get("apply_gaussian", False)
    sigma = smooth_cfg.get("sigma", 0.5)

    search_cfg = config.get("search", {})
    method = search_cfg.get("method", "powell")
    max_iter = search_cfg.get("max_iter", 100)

    alpha_init = 1.0
    if stage1a_result is not None:
        alpha_init = stage1a_result.get("best_alpha", 1.0)

    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "huber")
    delta = loss_cfg.get("delta", 0.01)
    w_tof = loss_cfg.get("w_tof", 1.0)
    w_smooth = loss_cfg.get("w_smooth", 5.0)
    w_bounds = loss_cfg.get("w_bounds", 10.0)
    w_anchor = loss_cfg.get("w_anchor", 0.1)

    alpha_cfg = config.get("alpha_search", config.get("search", {}))
    alpha_min = alpha_cfg.get("min", 0.0)
    alpha_max = alpha_cfg.get("max", 2.0)

    def objective_fn(params: np.ndarray) -> float:
        c_map = params_to_c_map(
            params,
            c_base,
            delta_c0,
            "stage1b",
            grid_h=grid_h,
            grid_w=grid_w,
            target_h=H,
            target_w=W,
            apply_gaussian=apply_gaussian,
            sigma=sigma,
        )
        tof_pred = forward_tof(c_map, x_s, x_r, scale_factor=scale_factor)
        smooth_correction = params_to_smooth_correction(
            params,
            grid_h=grid_h,
            grid_w=grid_w,
            target_h=H,
            target_w=W,
            apply_gaussian=apply_gaussian,
            sigma=sigma,
        )
        return stage1_objective(
            tof_pred,
            raw_tof,
            c_map,
            c0,
            c_min,
            c_max,
            smooth_correction=smooth_correction if np.any(smooth_correction != 0) else None,
            w_tof=w_tof,
            w_smooth=w_smooth,
            w_bounds=w_bounds,
            w_anchor=w_anchor,
            loss_type=loss_type,
            delta=delta,
        )

    x0 = get_initial_params(
        "stage1b",
        alpha_init=alpha_init,
        grid_h=grid_h,
        grid_w=grid_w,
        init_scale=init_scale,
    )
    bounds = get_bounds(
        "stage1b",
        alpha_min,
        alpha_max,
        grid_h=grid_h,
        grid_w=grid_w,
    )

    if method.lower() in ("powell", "nelder-mead"):
        result = minimize(
            objective_fn,
            x0,
            method=method,
            options={"maxiter": max_iter},
        )
    else:
        result = minimize(
            objective_fn,
            x0,
            method="powell",
            options={"maxiter": max_iter},
        )

    best_params = result.x
    best_loss = float(result.fun)
    best_c_map = params_to_c_map(
        best_params,
        c_base,
        delta_c0,
        "stage1b",
        grid_h=grid_h,
        grid_w=grid_w,
        target_h=H,
        target_w=W,
        apply_gaussian=apply_gaussian,
        sigma=sigma,
    )
    best_tof_pred = forward_tof(best_c_map, x_s, x_r, scale_factor=scale_factor)

    n_z = grid_h * grid_w
    best_z = best_params[1 : 1 + n_z].copy() if len(best_params) > 1 else np.array([])

    return {
        "best_alpha": float(best_params[0]),
        "best_z": best_z,
        "best_loss": best_loss,
        "best_c_map": best_c_map,
        "tof_pred": best_tof_pred,
        "raw_tof": raw_tof,
        "optimization_success": result.success,
    }
