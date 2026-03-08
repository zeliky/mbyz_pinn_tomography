"""Straight-line ray backprojection of ToF residuals onto the grid.

Used to build M2 mask for ROI: regions where backprojected residual magnitude
exceeds a threshold indicate likely error locations.
"""

from __future__ import annotations

import numpy as np


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Bresenham line pixels (row, col) from (y0, x0) to (y1, x1)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    pixels = []
    x, y = x0, y0
    while True:
        pixels.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return np.array(pixels, dtype=np.int32)


def backproject_tof_residuals(
    tof_residual: np.ndarray,
    x_s: np.ndarray,
    x_r: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Backproject |tof_residual| onto grid via straight-line rays.

    For each source s and receiver r with residual r_sr, spread |r_sr|/path_length
    along the straight line from source to receiver.

    Args:
        tof_residual: (S, R) residuals (tof_pred - raw_tof).
        x_s: (S, 2) source positions (x, y) in grid coords.
        x_r: (R, 2) receiver positions (x, y) in grid coords.
        grid_shape: (H, W).

    Returns:
        (H, W) backprojected magnitude map (non-negative).
    """
    H, W = grid_shape
    backproj = np.zeros((H, W), dtype=np.float64)

    S = tof_residual.shape[0]
    R = tof_residual.shape[1]

    for s in range(S):
        xs, ys = float(x_s[s, 0]), float(x_s[s, 1])
        ix0 = int(round(xs))
        iy0 = int(round(ys))
        ix0 = max(0, min(ix0, W - 1))
        iy0 = max(0, min(iy0, H - 1))

        for r in range(R):
            xr, yr = float(x_r[r, 0]), float(x_r[r, 1])
            ix1 = int(round(xr))
            iy1 = int(round(yr))
            ix1 = max(0, min(ix1, W - 1))
            iy1 = max(0, min(iy1, H - 1))

            r_sr = float(np.abs(tof_residual[s, r]))
            if r_sr < 1e-12:
                continue

            pixels = bresenham_line(ix0, iy0, ix1, iy1)
            n_pixels = len(pixels)
            if n_pixels < 2:
                n_pixels = 2
            contrib = r_sr / n_pixels

            for py, px in pixels:
                if 0 <= py < H and 0 <= px < W:
                    backproj[py, px] += contrib

    return backproj
