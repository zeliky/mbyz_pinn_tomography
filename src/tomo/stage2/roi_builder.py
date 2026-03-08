"""Build ROI and support masks from Stage 1 residual structure.

M1: |delta_stage1| > tau_delta
M2: backprojection(|tof_residual|) > tau_residual
ROI = dilate(M1 | M2)
Support = ring around ROI for smooth corrections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from tomo.stage2.backprojection import backproject_tof_residuals


@dataclass
class ROIMasks:
    """ROI and support masks plus coordinate arrays."""

    roi_mask: np.ndarray  # (H, W) bool
    support_mask: np.ndarray  # (H, W) bool
    roi_coords: np.ndarray  # (N_roi, 2) (row, col) for ROI pixels
    support_coords: np.ndarray  # (N_support, 2) for support pixels
    m1_mask: np.ndarray  # (H, W) |delta| > tau_delta
    m2_mask: np.ndarray  # (H, W) backprojection > tau_residual
    backproj_map: np.ndarray  # (H, W) backprojection values


def build_roi_and_support(
    delta_stage1: np.ndarray,
    tof_residual: np.ndarray,
    x_s: np.ndarray,
    x_r: np.ndarray,
    *,
    tau_delta: float = 0.02,
    tau_residual: float = 0.01,
    dilation_iterations: int = 2,
    support_width: int = 1,
) -> ROIMasks:
    """Build ROI and support masks from Stage 1 residuals.

    Args:
        delta_stage1: (H, W) c_stage1 - c_base.
        tof_residual: (S, R) tof_pred_stage1 - raw_tof_obs.
        x_s: (S, 2) source positions (x, y) in grid coords.
        x_r: (R, 2) receiver positions (x, y) in grid coords.
        tau_delta: Threshold for |delta_stage1|.
        tau_residual: Threshold for backprojected residual.
        dilation_iterations: Iterations for ROI dilation.
        support_width: Extra dilation for support ring.

    Returns:
        ROIMasks with roi_mask, support_mask, coord arrays.
    """
    H, W = delta_stage1.shape

    m1_mask = np.abs(delta_stage1) > tau_delta

    backproj_map = backproject_tof_residuals(tof_residual, x_s, x_r, (H, W))
    m2_mask = backproj_map > tau_residual

    combined = m1_mask | m2_mask
    kernel = np.ones((3, 3), dtype=bool)
    roi_mask = ndimage.binary_dilation(combined, structure=kernel, iterations=dilation_iterations)

    support_dilated = ndimage.binary_dilation(roi_mask, structure=kernel, iterations=support_width)
    support_mask = support_dilated & ~roi_mask

    roi_coords = np.argwhere(roi_mask)
    support_coords = np.argwhere(support_mask)

    return ROIMasks(
        roi_mask=roi_mask,
        support_mask=support_mask,
        roi_coords=roi_coords,
        support_coords=support_coords,
        m1_mask=m1_mask,
        m2_mask=m2_mask,
        backproj_map=backproj_map,
    )
