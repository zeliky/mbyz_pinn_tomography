from logger import log_message
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from eikonal import eikonal

def eikonal_traveltime(x, y, z, V, S, R, iuse=None, solver_type=1):
    """
    Compute travel time using the eikonal equation.

    Parameters:
        x, y, z (ndarray): Grid coordinates.
        V (ndarray): Velocity field.
        S (ndarray): Source positions (N, 2 or N, 3).
        R (ndarray): Receiver positions (M, 2 or M, 3).
        iuse (list, optional): Indices of sources to use.
        solver_type (int, optional): Type of solver. Default is 1 (Fast Marching).

    Returns:
        tmap (ndarray): Computed travel time map.
        t (ndarray): Travel time values at receiver locations.
    """

    # Ensure sources and receivers are arrays
    S, R = np.asarray(S), np.asarray(R)

    # If iuse is None, use all sources
    if iuse is None:
        iuse = np.arange(len(S))

    # Select relevant sources and receivers
    S, R = S[iuse, :], R[iuse, :]

    # Ensure sources & receivers have at least (N,3) shape when necessary
    if S.shape[1] < 3:
        S = np.column_stack((S, np.ones(S.shape[0]) * np.atleast_1d(z)[0]))
    if R.shape[1] < 3:
        R = np.column_stack((R, np.ones(R.shape[0]) * np.atleast_1d(z)[0]))

    # Compute unique sources
    unique_sources = np.unique(S, axis=0)

    # Solve the eikonal equation
    tmap = eikonal(x, y, z, V, unique_sources, solver_type)
    tmap = np.squeeze(tmap)  # Ensure correct dimensionality

    # Set up interpolation
    if tmap.ndim == 2:  # 2D case
        interp_func = RegularGridInterpolator(
            (x.ravel(), y.ravel()), tmap, method='linear', bounds_error=False, fill_value=None
        )
    elif tmap.ndim == 3:  # 3D case
        interp_func = RegularGridInterpolator(
            (x.ravel(), y.ravel(), z.ravel()), tmap, method='linear', bounds_error=False, fill_value=None
        )
    else:
        raise ValueError(f"[ERROR] Unexpected tmap.shape: {tmap.shape}")

    # Compute travel times at receiver locations
    t = np.zeros(len(R))
    for i, source in enumerate(unique_sources):
        indices = np.where((S == source).all(axis=1))[0]
        if tmap.ndim == 2:
            t[indices] = interp_func(R[indices, :2])
        else:
            t[indices] = interp_func(R[indices])

    return tmap, t
