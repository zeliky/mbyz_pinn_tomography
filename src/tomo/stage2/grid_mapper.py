"""Map policy ROI corrections back to 128x128 SoS grid.

Scatter ROI deltas -> Gaussian smooth -> clamp -> apply step size.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def map_roi_corrections_to_grid(
    roi_coords: np.ndarray,
    roi_delta: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    sigma: float = 0.5,
    max_delta: float = 0.02,
    eta: float = 0.5,
) -> np.ndarray:
    """Map ROI corrections to full grid, smooth, and scale.

    Args:
        roi_coords: (N_roi, 2) (row, col) for each ROI pixel.
        roi_delta: (N_roi,) policy output per ROI node.
        grid_shape: (H, W).
        sigma: Gaussian smoothing sigma.
        max_delta: Clamp before eta.
        eta: Step size (c_new = c_stage1 + eta * delta).

    Returns:
        (H, W) delta_c_policy to add to c_stage1.
    """
    H, W = grid_shape
    delta_grid = np.zeros((H, W), dtype=np.float64)

    roi_delta = np.asarray(roi_delta, dtype=np.float64)
    roi_delta = np.clip(roi_delta, -max_delta, max_delta)

    for i, (r, c) in enumerate(roi_coords):
        if 0 <= r < H and 0 <= c < W:
            delta_grid[r, c] = roi_delta[i]

    if sigma > 0:
        delta_grid = ndimage.gaussian_filter(delta_grid, sigma=sigma, mode="constant", cval=0)

    delta_grid = np.clip(delta_grid, -max_delta, max_delta)
    delta_grid = delta_grid * eta

    return delta_grid
