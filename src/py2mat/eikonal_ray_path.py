import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from logger import log_message, log_image

def eikonal_ray_path(x, y, s, source, receiver, tmap, doPlot=False):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    step_size = min(dx, dy) / 4
    max_steps = 10000
    stop_tol = min(dx, dy)

    # Transpose tmap so that tmap[i, j] ~ t(x[j], y[i])
    tmap_T = tmap.T

    # Gradients in MATLAB are (∂t/∂y, ∂t/∂x) due to row/column layout
    dt_dy, dt_dx = np.gradient(tmap_T, y, x)  # i.e., gradient w.r.t axes (y, x)

    # Interpolators over correct axis: (x, y)
    interp_U = RegularGridInterpolator((x, y), -dt_dx.T, bounds_error=False, fill_value=np.nan)
    interp_V = RegularGridInterpolator((x, y), -dt_dy.T, bounds_error=False, fill_value=np.nan)

    current_point = np.array(receiver, dtype=np.float64)
    path = [current_point.copy()]

    for _ in range(max_steps):
        point_xy = current_point
        direction = np.array([
            interp_U(point_xy),
            interp_V(point_xy)
        ]).reshape(2)

        if np.any(np.isnan(direction)) or np.linalg.norm(direction) == 0:
            log_message("Terminating: NaN or zero gradient.")
            break

        direction /= np.linalg.norm(direction)
        next_point = current_point + step_size * direction

        if not (x[0] <= next_point[0] <= x[-1] and y[0] <= next_point[1] <= y[-1]):
            log_message("Integration out of bounds.")
            break

        path.append(next_point.copy())
        current_point = next_point

        if np.linalg.norm(current_point - source) < stop_tol:
            break
    else:
        log_message("Warning: Reached max steps without reaching source.")

    raypath = np.array(path)

    # Truncate at source
    distances = np.linalg.norm(raypath - source, axis=1)
    close_indices = np.where(distances < stop_tol)[0]
    if len(close_indices) > 0:
        raypath = raypath[:close_indices[0]+1]
        raypath = np.vstack([raypath, source])
    else:
        log_message("Warning: No close point to source found in path.")
        raypath = np.empty((0, 2))

    if raypath.shape[0] > 1:
        diffs = np.diff(raypath, axis=0)
        raylength = np.sum(np.linalg.norm(diffs, axis=1))
    else:
        raylength = 0.0

    if doPlot:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(s, extent=[x.min(), x.max(), y.min(), y.max()],
                       origin='lower', aspect='equal', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Slowness')
        if raypath.shape[0] > 1:
            ax.plot(raypath[:, 0], raypath[:, 1], 'w-', linewidth=2, label='Ray path')
        ax.plot(source[0], source[1], 'go', markersize=8, label='Source')
        ax.plot(receiver[0], receiver[1], 'ro', markersize=8, label='Receiver')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Eikonal Ray Path over Slowness Map')
        ax.legend()
        log_image(fig)

    return raylength, raypath
