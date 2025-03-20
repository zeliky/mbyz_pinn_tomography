import numpy as np
from msfm import msfm  # Calls msfm, which in turn calls msfm2d
from logger import log_message

def eikonal(x, y, z, V, Sources=None, solver_type=1):
    """
    Solves the eikonal equation using Fast Marching (solver_type=1).
    Converts source positions to grid indices and computes traveltime maps.

    Parameters:
    x, y, z : 1D arrays defining the grid coordinates.
    V : 2D or 3D numpy array representing the velocity field.
    Sources : (N, 3) array containing source positions (x, y, z).
    solver_type : Type of solver (1 = Fast Marching, 2 = Finite Differences).

    Returns:
    tmap : Array of computed traveltimes of shape (ny, nx, ns).
    """
    if Sources is None:
        Sources = np.array([[x[0], y[0], z[0]]])  # Default source position

    # Ensure `x`, `y`, `z` are 1D arrays
    x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)

    # Ensure `V` is at least 2D
    V = np.atleast_3d(V)  
    ny, nx, nz = V.shape  

    if nz > 1:
        log_message("[eikonal.py]: 3D velocity field detected, using first slice for 2D.")
        V = V[:, :, 0]  # Take the first depth slice for 2D processing

    dx = x[1] - x[0]  # Grid spacing

    # Ensure Sources has at least 3 columns (x, y, z)
    if Sources.shape[1] < 3:
        Sources = np.column_stack((Sources, np.full((Sources.shape[0], 1), np.atleast_1d(z)[0])))

    # Convert Sources to a NumPy array and ensure correct data type
    Sources = np.array(Sources, dtype=float)  # Force conversion to numeric type

    # Find unique sources
    unique_sources = np.unique(Sources, axis=0)
    num_sources = unique_sources.shape[0]

    tmap = np.zeros((ny, nx, num_sources))  # Initialize traveltime array

    for i, source in enumerate(unique_sources):
        reference = np.array([x[0], y[0], z[0]])  # Reference for indexing
        
        source = np.asarray(source)
        reference = np.asarray(reference)
        
        # Convert source positions to grid indices
        source_index = np.round((source[:2] - reference[:2]) / dx).astype(int)

        # Swap X and Y (matches MATLAB msfm call)
        source_index[[0, 1]] = source_index[[1, 0]]

        # Commented out to reduce excessive output
        # log_message(f"[eikonal.py]: Calling msfm with V.shape={V.shape}, source_index.shape={source_index.shape}")

        # Call `msfm`
        # tmap[..., i] = msfm(V, source_index, True) * dx  # Use 2D fast marching and scale by dx
        #log_message(f'V.shape = {V.shape}, source_index.shape = {source_index.shape}')
        #log_message(f'source_index = {source_index}')
        tmap[..., i] = msfm(V, source_index, use_second=True, use_cross=True) * dx  # Use 2D fast marching and scale by dx
        #log_message('use_second=True, use_cross=True')

    return tmap
