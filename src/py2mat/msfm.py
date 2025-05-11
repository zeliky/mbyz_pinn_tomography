import numpy as np
from msfm2d import msfm2d  # Ensure msfm2d is correctly imported

def msfm(F, source_index, solver_type=1, use_second=True, use_cross=True):
    """
    Calls the appropriate solver for the Fast Marching method.
    Supports both 2D (`msfm2d`) and 3D (`msfm3d`) cases.

    Parameters:
    F : 2D or 3D numpy array representing the velocity field.
    source_index : (sy, sx) or (ny, nx, nz) coordinates of the source.
    solver_type : Type of solver (1 = Fast Marching, 2 = Finite Differences).
    use_second : Boolean, whether to use second-order derivatives.
    use_cross : Boolean, whether to use cross-neighbor calculations.

    Returns:
    tmap : 2D or 3D numpy array of computed traveltimes.
    """
    # Ensure F is correctly shaped
    F = np.squeeze(F)

    if F.ndim not in [2, 3]:
        raise ValueError(f"[msfm.py]:Unexpected F.shape = {F.shape}. Expected a 2D or 3D array.")

    # Ensure source_index is at least 2D
    source_index = np.atleast_2d(source_index)  # Ensure 2D shape

    # Debugging output before reshaping
    # print(f"Original source_index shape: {source_index.shape}, values: {source_index}")

    # 🛠 Fix: Ensure `source_index` is correctly shaped (N,2) for 2D cases
    if F.ndim == 2:
        if source_index.shape[1] != 2:
            source_index = source_index.T if source_index.shape[0] == 2 else source_index.reshape(-1, 2)
    
    # 🛠 Fix: Ensure `source_index` is in int32 format
    source_index = np.asfortranarray(source_index, dtype=np.int32)
    
    # print(f"Final source_index (before msfm2d call) shape: {source_index.shape}, values: {source_index}")

    # Call the appropriate solver
    if F.ndim == 2:
        return msfm2d(F, source_index, use_second, use_cross)
    else:
        print('[msfm.py]: msfm3d not used')
        return None  # Placeholder for 3D case

# import numpy as np
# #from my_msfm2d import my_msfm2d
# from msfm2d import msfm2d  # Ensure msfm2d is correctly imported
# # from msfm3d import msfm3d  # Ensure msfm3d is correctly imported

# def msfm(F, source_index, solver_type=1, use_second=True, use_cross=True):
#     """
#     Calls the appropriate solver for the Fast Marching method.
#     Supports both 2D (`msfm2d`) and 3D (`msfm3d`) cases.

#     Parameters:
#     F : 2D or 3D numpy array representing the velocity field.
#     source_index : (sy, sx) or (ny, nx, nz) coordinates of the source.
#     solver_type : Type of solver (1 = Fast Marching, 2 = Finite Differences).
#     use_second : Boolean, whether to use second-order derivatives.
#     use_cross : Boolean, whether to use cross-neighbor calculations.

#     Returns:
#     tmap : 2D or 3D numpy array of computed traveltimes.
#     """
#     # Ensure F is correctly shaped
#     F = np.squeeze(F)

#     if F.ndim not in [2, 3]:
#         raise ValueError(f"Unexpected F.shape = {F.shape}. Expected a 2D or 3D array.")

#     # Ensure source_index is at least 2D
#     source_index = np.atleast_2d(source_index)  # Ensure 2D shape

#     # 🛠 Fix: Reshape `source_index` to (2, N) for 2D cases
#     if F.ndim == 2 and source_index.shape[0] != 2:
#         source_index = source_index.T if source_index.shape[1] == 2 else source_index.reshape(2, -1)

#     if F.ndim == 3 and source_index.shape[0] != 3:
#         raise ValueError(f"Unexpected source_index.shape = {source_index.shape}. Expected (3, N) for 3D case.")

#     # Call the appropriate solver
#     if F.ndim == 2:
#         return msfm2d(F, source_index, use_second, use_cross)
#         #return my_msfm2d(F, source_index, use_second, use_cross)
#     else:
#         print('msfm3d not used')
#         #return msfm3d(F, source_index, use_second, use_cross)
