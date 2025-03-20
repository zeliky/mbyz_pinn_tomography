import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from eikonal_traveltime import eikonal_traveltime
from eikonal_ray_path import eikonal_ray_path
from logger import log_message
import time
from TimeMeasurement.time_measurement import convert

def ConvertMatCell2mat(mat_cell):
    rows_cell = len(mat_cell)
    cols_cell = len(mat_cell[0])
    rows, cols = mat_cell[0][0].shape
    mat = np.zeros((rows_cell * cols_cell, rows * cols))
    for row_idx in range(rows_cell):
        for col_idx in range(cols_cell):
            item_idx = row_idx * cols_cell + col_idx
            mat[item_idx, :] = mat_cell[row_idx][col_idx].flatten()
    return mat

def save_checkpoint(data):
    checkpoint_file = "../myOutputs/calculateL_checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint():
    checkpoint_file = "../myOutputs/calculateL_checkpoint.pkl"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def CalculateL(s, input_to_L):
    x = np.asarray(input_to_L["x"], dtype=np.float64).flatten()
    y = np.asarray(input_to_L["y"], dtype=np.float64).flatten()
    z = np.asarray(input_to_L["z"], dtype=np.float64)
    xs_receivers = np.asarray(input_to_L["xs_receivers"], dtype=np.float64).flatten()
    ys_receivers = np.asarray(input_to_L["ys_receivers"], dtype=np.float64).flatten()
    xs_sources = np.asarray(input_to_L["xs_sources"], dtype=np.float64).flatten()
    ys_sources = np.asarray(input_to_L["ys_sources"], dtype=np.float64).flatten()
    doPlotRaypathsInCalculateL = input_to_L["doPlotRaypathsInCalculateL"]
    number_of_receivers = len(xs_receivers)
    number_of_sources = len(xs_sources)
    S = np.column_stack((xs_sources, ys_sources))
    m = len(x)
    n = len(y)
    mydist = np.zeros((n, m))
    
    # # Load checkpoint if available
    # checkpoint = load_checkpoint()
    # if checkpoint:
    #     RL, LL, start_ir, start_isrc = checkpoint
    #     log_message(f'[CalculateL.py]: This is a restart run, start_ir = {start_ir}, start_isrc = {start_isrc}')
    # else:
    #     LL = [[None for _ in range(number_of_receivers)] for _ in range(number_of_sources)]
    #     RL = np.zeros((number_of_sources, number_of_receivers))
    #     start_ir, start_isrc = 0, 0
        
    LL = [[None for _ in range(number_of_receivers)] for _ in range(number_of_sources)]
    RL = np.zeros((number_of_sources, number_of_receivers))
    start_ir, start_isrc = 0, 0
    
    for ir in range(start_ir, number_of_receivers):
        log_message('.')
        log_message(f'[CalculateL.py] Receiver {ir+1} out of {number_of_receivers} recievers')
        st = time.process_time()
        R1 = np.column_stack((
            np.full((number_of_sources,), xs_receivers[ir]),
            np.full((number_of_sources,), ys_receivers[ir])
        ))
        tmap, t0 = eikonal_traveltime(x, y, z, s, R1, S)
        from_receiver = np.array([xs_receivers[ir], ys_receivers[ir]])
        
        # intermediate_end_time = time.process_time()
        # hours, minutes, seconds = convert(intermediate_end_time - st)
        # log_message(f"[calculateL.py]: Receiver {ir+1} before running over sources, execution time: {int(hours)} hours, {int(minutes)} Minutes, {int(seconds)} seconds")
        
        for isrc in range(start_isrc, number_of_sources):
            LL[isrc][ir] = np.zeros((n, m))
            raylength = 0
            if xs_receivers[ir] != xs_sources[isrc]:
                to_source = np.array([xs_sources[isrc], ys_sources[isrc]])
                #log_message('')
                #log_message(f"[calculateL.py]: calling eikonal_ray_path, receiver {ir+1}, source {isrc+1}")
                raylength, raypath = eikonal_ray_path(x, y, s, from_receiver, to_source, tmap, doPlot=doPlotRaypathsInCalculateL)
                x_new, y_new = raypath[:, 0], raypath[:, 1]
                raydist = np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2)
                for ixy in range(len(x_new) - 1):
                    x_idx, y_idx = round(x_new[ixy]), round(y_new[ixy])
                    if 0 <= x_idx < m and 0 <= y_idx < n:
                        mydist[y_idx, x_idx] = raydist[ixy]
            LL[isrc][ir] = mydist.copy()
            mydist.fill(0)
            RL[isrc, ir] = raylength
            
            # Save checkpoint after each source-receiver pair
            # save_checkpoint((RL, LL, ir, isrc + 1))
        et = time.process_time()
        hours, minutes, seconds = convert(et - st)
        log_message(f"[calculateL.py]: Receiver {ir+1} for all sources, execution time: {int(hours)} hours, {int(minutes)} Minutes, {int(seconds)} seconds")
        start_isrc = 0  # Reset for next receiver
        # save_checkpoint((RL, LL, ir + 1, 0))
    
    L = ConvertMatCell2mat(LL)
    L[np.isnan(L)] = 0
    RL[np.isnan(RL)] = 0
    # os.remove("../myOutputs/calculateL_checkpoint.pkl")  # Remove checkpoint after successful completion
    return RL, L


# ==========================================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from eikonal_traveltime import eikonal_traveltime
# from eikonal_ray_path import eikonal_ray_path
# from logger import log_message
# import time
# from TimeMeasurement.time_measurement import convert

# def ConvertMatCell2mat(mat_cell):
#     """
#     Converts a list of 2D NumPy arrays (mat_cell) into a single 2D NumPy array.

#     Parameters:
#         mat_cell (list of lists): A 2D list where each element is a NumPy array of the same shape.

#     Returns:
#         mat (ndarray): A 2D NumPy array where each row is the flattened version of an element in mat_cell.
#     """

#     rows_cell = len(mat_cell)
#     cols_cell = len(mat_cell[0])

#     rows, cols = mat_cell[0][0].shape

#     mat = np.zeros((rows_cell * cols_cell, rows * cols))

#     for row_idx in range(rows_cell):
#         for col_idx in range(cols_cell):
#             item_idx = row_idx * cols_cell + col_idx
#             mat[item_idx, :] = mat_cell[row_idx][col_idx].flatten()

#     return mat

# def CalculateL(s, input_to_L):
#     """
#     Computes the ray path lengths and matrix L using the eikonal equation.

#     Parameters:
#         s (ndarray): Velocity field.
#         input_to_L (dict): Dictionary containing grid and source/receiver positions.

#     Returns:
#         RL (ndarray): Ray lengths.
#         L (ndarray): Matrix representation of ray paths.
#     """
    
#     x = np.asarray(input_to_L["x"], dtype=np.float64).flatten()
#     y = np.asarray(input_to_L["y"], dtype=np.float64).flatten()
#     z = np.asarray(input_to_L["z"], dtype=np.float64)
    
#     xs_receivers = np.asarray(input_to_L["xs_receivers"], dtype=np.float64).flatten()
#     ys_receivers = np.asarray(input_to_L["ys_receivers"], dtype=np.float64).flatten()
#     xs_sources = np.asarray(input_to_L["xs_sources"], dtype=np.float64).flatten()
#     ys_sources = np.asarray(input_to_L["ys_sources"], dtype=np.float64).flatten()
    
#     doPlotRaypathsInCalculateL = input_to_L["doPlotRaypathsInCalculateL"]

#     number_of_receivers = len(xs_receivers)
#     number_of_sources = len(xs_sources)

#     # Ensure S has correct shape (number_of_sources, 2)
#     S = np.column_stack((xs_sources, ys_sources))  # MATLAB: `S = [xs_sources' ys_sources']`

#     m = len(x)
#     n = len(y)

#     mydist = np.zeros((n, m))
#     LL = [[None for _ in range(number_of_receivers)] for _ in range(number_of_sources)]

#     RL = np.zeros((number_of_sources, number_of_receivers))

#     for ir in range(number_of_receivers):  # Loop over receivers
#         # Ensure R1 is correctly formatted as a (number_of_sources, 2) array
#         st = time.process_time()
#         R1 = np.column_stack((
#             np.full((number_of_sources,), xs_receivers[ir]),
#             np.full((number_of_sources,), ys_receivers[ir])
#         ))  # MATLAB: `R1=[ones(number_of_sources,1)*xs_receivers(ir) ones(number_of_sources,1)*ys_receivers(ir)]`

#         # Compute travel times (reversed from receivers to sources)
#         tmap, t0 = eikonal_traveltime(x, y, z, s, R1, S)

#         from_receiver = np.array([xs_receivers[ir], ys_receivers[ir]])

#         for isrc in range(number_of_sources):  # Loop over sources
#             LL[isrc][ir] = np.zeros((n, m))
#             raylength = 0
            
#             if xs_receivers[ir] != xs_sources[isrc]:
#                 # Extract the correct source coordinates (MATLAB column-wise indexing equivalent)
#                 to_source = np.array([xs_sources[isrc], ys_sources[isrc]])

#                 log_message(f"[calculateL.py]: calling eikonal_ray_path, receiver {ir+1}, source {isrc+1}")

#                 # Compute ray path
#                 raylength, raypath = eikonal_ray_path(x, y, s, from_receiver, to_source, tmap, doPlot=doPlotRaypathsInCalculateL)

#                 x_new, y_new = raypath[:, 0], raypath[:, 1]

#                 # if doPlotRaypathsInCalculateL:
#                 #     plt.plot(x_new, y_new, 'r')

#                 raydist = np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2)

#                 for ixy in range(len(x_new) - 1):
#                     x_idx, y_idx = round(x_new[ixy]), round(y_new[ixy])
#                     if 0 <= x_idx < m and 0 <= y_idx < n:
#                         mydist[y_idx, x_idx] = raydist[ixy]

#             LL[isrc][ir] = mydist.copy()
#             mydist.fill(0)

#             RL[isrc, ir] = raylength
#             et = time.process_time()
#             res = et - st
#             hours, minutes, seconds = convert(res)
#             log_message(f"[claculateL.py]: Receiver {ir}, execution time: {int(hours)} hours, {int(minutes)} Minutes, {int(seconds)} seconds")

#     # Convert the list of matrices to a single matrix
#     L = ConvertMatCell2mat(LL)
#     L[np.isnan(L)] = 0
#     RL[np.isnan(RL)] = 0
    
#     return RL, L

