import numpy as np
import ctypes

# Load the compiled C shared library (keep the original DLL path)
dll_path = "D:/Michael/Yaron/PINNs_Code/non_linear_tomography/src/c/msfm2d.dll"
msfm2d_lib = ctypes.CDLL(dll_path)

# Define argument and return types for the function
msfm2d_lib.msfm2d.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # T (output)
    ctypes.POINTER(ctypes.c_double),  # F (input)
    ctypes.POINTER(ctypes.c_int),     # source_points
    ctypes.c_int,                     # num_sources
    ctypes.c_int,                     # rows
    ctypes.c_int,                     # cols
    ctypes.c_bool,                     # use_second
    ctypes.c_bool                      # use_cross
]

def msfm2d(F, source_points, use_second=True, use_cross=True):
    """
    Python wrapper for the msfm2d C implementation.

    Parameters:
    - F: Speed function (numpy array)
    - source_points: List of source points (numpy array)
    - use_second: Use second-order approximations
    - use_cross: Use cross derivatives

    Returns:
    - T: Arrival times (numpy array)
    """

    # Ensure inputs are column-major order (Fortran-contiguous) like MATLAB
    F = np.asfortranarray(F, dtype=np.float64)
    source_points = np.asfortranarray(source_points, dtype=np.int32)

    rows, cols = F.shape
    num_sources = source_points.shape[0]

    # Ensure source_points is a column vector (MATLAB-style)
    if source_points.shape[1] != 2:
        source_points = source_points.reshape(-1, 2)
    
    # Allocate memory for the output array T
    T = np.full(F.shape, np.inf, dtype=np.float64)  # Initialize T with infinity

    # Get pointers to the data
    T_ptr = T.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    F_ptr = F.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    source_points_ptr = source_points.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Call the C function
    msfm2d_lib.msfm2d(
        T_ptr, F_ptr, source_points_ptr, num_sources, rows, cols, use_second, use_cross
    )

    return T

# gcc -c common.c -o common.o
# gcc -c msfm2d.c -o msfm2d.o
# gcc -shared -o msfm2d.dll msfm2d.o common.o -lm
