import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from eikonal import eikonal
from logger import log_message, log_image

def eikonal_raylength(x, y, v, S, R, tS=None, do_plot=False):
    """
    Computes the ray length from S (source) to R (receiver) using the eikonal equation.
    
    Parameters:
        x, y : Grid coordinates
        v    : Velocity field
        S    : Source coordinates (x, y)
        R    : Receiver coordinates (x, y)
        tS   : Precomputed travel time field (optional)
        do_plot : Whether to plot the ray path (default: False)
        
    Returns:
        raylength : Total path length of the computed ray
    """

    # Handle missing tS by computing it
    if tS is None:
        log_message("[eikonal_raylength.py]: Computing tS using eikonal solver")
        tS = eikonal(x, y, v, S)
    
    # Squeeze `tS` to remove unnecessary singleton dimensions
    tS = np.squeeze(tS)
    log_message(f"[eikonal_raylength.py]: tS.shape after squeeze = {tS.shape}")

    # --- Ensure `x` and `y` Are 1D ---
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"x and y must be 1D arrays, but got x.shape={x.shape}, y.shape={y.shape}")

    # Compute spacing
    dx = np.diff(x[:2])[0] if len(x) > 1 else x[0]
    dy = np.diff(y[:2])[0] if len(y) > 1 else y[0]

    # Compute spatial gradients of travel time
    U, V = np.gradient(tS, dx, dy)

    # --- Ensure `R` is (N,2) ---
    R = np.array(R).squeeze()
    #log_message(f"[eikonal_raylength.py]: Original R.shape = {R.shape}")

    if R.ndim == 1:
        if len(R) % 2 != 0:
            raise ValueError(f"R has an odd number of elements ({len(R)}), cannot split into (x,y) pairs.")
        R = R.reshape(-1, 2)
    elif R.shape[0] == 1 and R.shape[1] > 2:
        R = R.reshape(-1, 2)
    elif R.shape[1] != 2:
        raise ValueError(f"R should have shape (N, 2), but got {R.shape}")

    log_message(f"[eikonal_raylength.py]: R.shape after reshape = {R.shape}")

    # --- Ensure `S` is (N,2) ---
    S = np.array(S).squeeze()
    if S.ndim == 1:
        if len(S) % 2 != 0:
            raise ValueError(f"S has an odd number of elements ({len(S)}), cannot split into (x,y) pairs.")
        S = S.reshape(-1, 2)
    elif S.shape[0] == 1 and S.shape[1] > 2:
        S = S.reshape(-1, 2)
    elif S.shape[1] != 2:
        raise ValueError(f"S should have shape (N, 2), but got {S.shape}")

    #log_message(f"[eikonal_raylength.py]: S.shape after reshape = {S.shape}")

    # Use the first receiver (R[0]) for ray tracing
    start_point = R[0].copy()  # Ensure correct shape (2,)

    # Define ODE function for the streamline
    def ray_ode(state, t):
        """ Computes the velocity at the given state (x, y). """
        state = np.clip(state, [x.min(), y.min()], [x.max(), y.max()])  # Clamp state within bounds

        x_interp = RegularGridInterpolator((x, y), -U.T, method="linear", bounds_error=False, fill_value=None)
        y_interp = RegularGridInterpolator((x, y), -V.T, method="linear", bounds_error=False, fill_value=None)

        x_vel = x_interp(state)[0]
        y_vel = y_interp(state)[0]
        
        norm_factor = np.sqrt(x_vel**2 + y_vel**2)
        if norm_factor == 0:
            return [0, 0]  # Stop integration if gradient is zero

        return [x_vel / norm_factor, y_vel / norm_factor]

    # Time vector for integration
    t_vals = np.linspace(0, 5, 500)  # Adjust step size and range as needed

    # Solve for the ray path using numerical integration
    raypath = odeint(ray_ode, start_point, t_vals)

    # --- Fix Shape Mismatch ---
    target_source = S[0]  # Select only the first source
    dist_to_source = np.linalg.norm(raypath - target_source, axis=1)

    valid_indices = np.where(dist_to_source > dx / 10)[0]

    if len(valid_indices) == 0:
        log_message("[eikonal_raylength.py]: No valid ray path found, returning NaN")
        return np.nan

    raypath = np.vstack((raypath[valid_indices], target_source))  # Ensure correct final source position

    # Compute total path length
    raylength = np.sum(np.sqrt(np.sum(np.diff(raypath, axis=0) ** 2, axis=1)))

    # Plot if requested
    if do_plot:
        fig = plt.figure()
        plt.imshow(tS.T, extent=[x[0], x[-1], y[0], y[-1]], origin="lower")
        plt.colorbar(label="Travel Time")
        plt.plot(raypath[:, 0], raypath[:, 1], "k-*", label="Ray Path")
        plt.scatter(S[0, 0], S[0, 1], c="r", marker="o", label="Source")
        plt.scatter(R[0, 0], R[0, 1], c="b", marker="x", label="Receiver")
        plt.legend()
        plt.show()
        log_image(fig)

    return raylength
