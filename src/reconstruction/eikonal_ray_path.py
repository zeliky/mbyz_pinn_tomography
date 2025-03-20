import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from logger import log_message, log_image

def eikonal_ray_path(x, y, v, S, R, tmap, doPlot=False):
    """
    Computes the ray path and ray length using Eikonal ray tracing.

    Parameters:
    x, y : 1D arrays defining the grid
    v : velocity map
    S : tuple (sx, sy) - source coordinates
    R : tuple (rx, ry) - receiver coordinates
    tmap : travel time map
    doPlot : Boolean flag to enable plotting

    Returns:
    raylength : Total ray path length
    raypath : List of (x, y) coordinates of the ray path
    """

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    xx, yy = np.meshgrid(x, y, indexing='ij')  # Match MATLAB indexing
    U, V = np.gradient(tmap, dx, dy)  # Compute gradients with correct dx, dy order
    U, V = -U.T, -V.T  # Flip and transpose to match MATLAB
    U, V = -U, -V  # Flip again to ensure the correct direction

    current_pos = np.array(R, dtype=np.float64)
    raypath = [current_pos.copy()]

    max_steps = 2000
    step_size = min(dx, dy) / 4  # Ensure MATLAB-compatible step size

    for step in range(max_steps):
        # Interpolate gradients at the current position
        u_interp = map_coordinates(U, [[current_pos[0]], [current_pos[1]]], order=1)[0]
        v_interp = map_coordinates(V, [[current_pos[0]], [current_pos[1]]], order=1)[0]

        if np.isnan(u_interp) or np.isnan(v_interp):
            break  # Stop if the velocity field is undefined

        velocity_magnitude = np.hypot(u_interp, v_interp)
        if velocity_magnitude < 1e-4:
            break  # Stop if velocity magnitude is too small

        # Interpolate velocity at current position
        speed_interp = map_coordinates(v, [[current_pos[0]], [current_pos[1]]], order=1)[0]

        # Ensure integration moves in the correct direction and scales correctly with velocity
        step_vector = np.array([u_interp, v_interp]) / (velocity_magnitude + 1e-6)
        current_pos += (step_size / speed_interp) * step_vector  # Scale step size properly

        raypath.append(current_pos.copy())

        # Fix stopping condition (stop when reaching the source)
        if np.linalg.norm(current_pos - S) < step_size:  # Stop when close to the source
            break

    raypath = np.array(raypath)
    raylength = np.sum(np.sqrt(np.diff(raypath[:, 0])**2 + np.diff(raypath[:, 1])**2))

    # Print debugging output for MATLAB-Python comparison
    print("==================== PYTHON: Computed Ray Path ====================")
    print(f"Total steps in ray path: {len(raypath)}")
    print(f"First point: [{raypath[0, 0]:.6f}, {raypath[0, 1]:.6f}]")
    print(f"Last point: [{raypath[-1, 0]:.6f}, {raypath[-1, 1]:.6f}]")
    print(f"Ray length: {raylength:.6f}")

    # Print five selected intermediate points
    num_steps = len(raypath)
    indices = np.round(np.linspace(1, num_steps-1, 5)).astype(int)  # Select 5 evenly spaced indices
    print("Selected intermediate points:")
    for i in indices:
        print(f"Step {i}: [{raypath[i, 0]:.6f}, {raypath[i, 1]:.6f}]")

    # Check gradients at start and end
    start_x, start_y = int(round(raypath[0, 0])), int(round(raypath[0, 1]))
    end_x, end_y = int(round(raypath[-1, 0])), int(round(raypath[-1, 1]))

    print(f"Gradient at first point (U, V): [{U[start_y, start_x]:.6f}, {V[start_y, start_x]:.6f}]")
    print(f"Gradient at last point (U, V): [{U[end_y, end_x]:.6f}, {V[end_y, end_x]:.6f}]")

    # Print step size used
    print(f"Step size: {step_size:.6f}")

    print("Stopping for verification.")
    exit()

    # Restore visualization part
    if doPlot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(U, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        axes[0, 0].set_title('U Gradient')

        axes[0, 1].imshow(V, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        axes[0, 1].set_title('V Gradient')

        axes[1, 0].imshow(tmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        axes[1, 0].set_title('Travel Time Map')

        axes[1, 1].imshow(v, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        axes[1, 1].set_title('Velocity Map & Ray Path')
        axes[1, 1].plot(raypath[:, 0], raypath[:, 1], 'w')

        log_image(fig)
        plt.close()

    return raylength, raypath



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import map_coordinates

# from logger import log_message, log_image

# def eikonal_ray_path(x, y, v, S, R, tmap, doPlot=False):
#     """
#     Efficiently computes the ray path and ray length based on the Eikonal equation.

#     Parameters:
#     x, y : 1D arrays defining the grid
#     v : velocity map
#     S : tuple (sx, sy) - source coordinates
#     R : tuple (rx, ry) - receiver coordinates
#     tmap : travel time map
#     doPlot : Boolean flag to enable plotting

#     Returns:
#     raylength : Total ray path length
#     raypath : List of (x, y) coordinates of the ray path
#     """

#     dx = abs(x[1] - x[0])
#     dy = abs(y[1] - y[0])

#     xx, yy = np.meshgrid(x, y, indexing='ij')  # Match MATLAB indexing
    
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # U, V = np.gradient(tmap, dx, dy)
#     # U, V = -U, -V  # Reverse gradients
 
#     # Fix gradient computation to match MATLAB
#     U, V = np.gradient(tmap, dy, dx)  # Swap order to match MATLAB
#     U, V = -U.T, -V.T  # Reverse gradients and transpose
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++


#     current_pos = np.array(R, dtype=np.float64)
#     raypath = [current_pos.copy()]

#     max_steps = 2000
#     step_size = min(dx, dy) / 4

#     for step in range(max_steps):
#         # Efficient interpolation using `map_coordinates`
#         u_interp = map_coordinates(U, [[current_pos[0]], [current_pos[1]]], order=1)[0]
#         v_interp = map_coordinates(V, [[current_pos[0]], [current_pos[1]]], order=1)[0]
    
        
#         if np.isnan(u_interp) or np.isnan(v_interp):
#             #log_message(f"[eikonal_ray_path.py]: Stopping due to NaN in velocity field at step {step}.")
#             break
        
#         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         # velocity_magnitude = np.hypot(u_interp, v_interp)
#         # if velocity_magnitude < 1e-4:
#         #     #log_message(f"[eikonal_ray_path.py]: Stopping due to low velocity at step {step}.")
#         #     break
         
        
#         # Ensure stopping condition matches MATLAB
 
#         # Fix step size in integration
#         velocity_magnitude = np.hypot(u_interp, v_interp)
#         adaptive_step = step_size / (velocity_magnitude + 1e-6)  # Adjusted step scaling
#         current_pos += adaptive_step * np.array([u_interp, v_interp])
        
#         # # Adaptive step size
#         # adaptive_step = step_size / (velocity_magnitude + 1e-6)
#         # current_pos += adaptive_step * np.array([u_interp, v_interp])
        
#         raypath.append(current_pos.copy())

#         if np.linalg.norm(current_pos - S) < 2 * step_size:  # More relaxed stopping
#             log_message(f"[eikonal_ray_path.py]: Reached source at step {step}.")
#             break
        
#         # if np.linalg.norm(current_pos - S) < step_size:
#         #     #log_message(f"[eikonal_ray_path.py]: Reached source at step {step}.")
#         #     break

#         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
#     raypath = np.array(raypath)
#     raylength = np.sum(np.sqrt(np.diff(raypath[:, 0])**2 + np.diff(raypath[:, 1])**2))

#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     print("==================== PYTHON: Computed Ray Path ====================")
#     print(f"Total steps in ray path: {len(raypath)}")
#     print(f"First point: [{raypath[0, 0]}, {raypath[0, 1]}]")
#     print(f"Last point: [{raypath[-1, 0]}, {raypath[-1, 1]}]")
#     print(f"Ray length: {raylength}")

#     # Print five selected intermediate points
#     num_steps = len(raypath)
#     indices = np.round(np.linspace(1, num_steps-1, 5)).astype(int)  # Select 5 evenly spaced indices
#     print("Selected intermediate points:")
#     for i in indices:
#         print(f"Step {i}: [{raypath[i, 0]}, {raypath[i, 1]}]")

#     # Check gradients at start and end
#     start_x, start_y = int(round(raypath[0, 0])), int(round(raypath[0, 1]))
#     end_x, end_y = int(round(raypath[-1, 0])), int(round(raypath[-1, 1]))

#     print(f"Gradient at first point (U, V): [{U[start_y, start_x]}, {V[start_y, start_x]}]")
#     print(f"Gradient at last point (U, V): [{U[end_y, end_x]}, {V[end_y, end_x]}]")

#     # Print step size used
#     print(f"Step size: {min(dx, dy) / 4}")

#     print("Stopping for verification.")
#     exit()

#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
#     # Restore visualization part
#     if doPlot:
#         fig, axes = plt.subplots(2, 2, figsize=(10, 10))

#         axes[0, 0].imshow(U, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[0, 0].set_title('U Gradient')

#         axes[0, 1].imshow(V, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[0, 1].set_title('V Gradient')

#         axes[1, 0].imshow(tmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[1, 0].set_title('Travel Time Map')

#         axes[1, 1].imshow(v, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         tmp_title = f'Velocity Map & Ray Path, last step = {step}'
#         axes[1, 1].set_title(tmp_title)
#         axes[1, 1].plot(raypath[:, 0], raypath[:, 1], 'w')

#         log_image(fig)
#         plt.close()

#     return raylength, raypath

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# import time
# from logger import log_message, log_image
# from TimeMeasurement.time_measurement import convert

# def eikonal_ray_path(x, y, v, S, R, tmap, doPlot=False):
#     """
#     Computes the ray path and ray length based on the Eikonal equation.
    
#     Parameters:
#     x, y: 1D arrays defining the grid
#     v: velocity map
#     S: tuple (sx, sy) - source coordinates
#     R: tuple (rx, ry) - receiver coordinates
#     tmap: travel time map
#     doPlot: Boolean flag to enable plotting
    
#     Returns:
#     raylength: Total ray path length
#     raypath: List of (x, y) coordinates of the ray path
#     """
    
#     st = time.process_time()
    
#     x = np.asarray(x, dtype=np.float64).flatten()
#     y = np.asarray(y, dtype=np.float64).flatten()
    
#     dx = abs(x[1] - x[0]) if len(x) > 1 else abs(x[0])
#     dy = abs(y[1] - y[0]) if len(y) > 1 else abs(y[0])
    
#     xx, yy = np.meshgrid(x, y, indexing='ij')  # MATLAB-like indexing
    
#     U, V = np.gradient(tmap, dx, dy)
#     U, V = -U, -V  # Reverse gradients to follow decreasing travel time
    
#     S = np.asarray(S, dtype=np.float64).flatten()
#     R = np.asarray(R, dtype=np.float64).flatten()
    
#     if S.shape != (2,):
#         raise ValueError(f"Unexpected shape for S: {S.shape}, expected (2,)")
#     if R.shape != (2,):
#         raise ValueError(f"Unexpected shape for R: {R.shape}, expected (2,)")
    
#     log_message(f"[eikonal_ray_path.py]: S: {S}, R: {R}")
    
#     raypath = [R]
#     current_pos = R.copy()
#     max_steps = 5000
#     step_size = min(dx, dy) / 4  # Reduce step size for better accuracy
    
#     for step in range(max_steps):
#         u_interp = griddata((xx.ravel(), yy.ravel()), U.ravel(), (current_pos[0], current_pos[1]), method='linear')
#         v_interp = griddata((xx.ravel(), yy.ravel()), V.ravel(), (current_pos[0], current_pos[1]), method='linear')
        
#         if np.isnan(u_interp) or np.isnan(v_interp):
#             log_message(f"[eikonal_ray_path.py]: Stopping due to NaN in velocity field at step {step}.")
#             break
        
#         velocity_magnitude = np.linalg.norm([u_interp, v_interp])
#         if velocity_magnitude < 1e-4:
#             log_message(f"[eikonal_ray_path.py]: Stopping due to low velocity at step {step}.")
#             break
        
#         adaptive_step = step_size / (velocity_magnitude + 1e-6)
#         current_pos += adaptive_step * np.array([u_interp, v_interp])
#         raypath.append(current_pos.copy())
        
#         if np.linalg.norm(current_pos - S) < step_size:
#             log_message(f"[eikonal_ray_path.py]: Reached source at step {step}.")
#             break
    
#     raypath = np.array(raypath)
#     raylength = np.sum(np.sqrt(np.diff(raypath[:, 0])**2 + np.diff(raypath[:, 1])**2))
    
#     if doPlot:
#         fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
#         axes[0, 0].imshow(U, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[0, 0].set_title('U Gradient')

#         axes[0, 1].imshow(V, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[0, 1].set_title('V Gradient')

#         axes[1, 0].imshow(tmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[1, 0].set_title('Travel Time Map')

#         axes[1, 1].imshow(v, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
#         axes[1, 1].set_title('Velocity Map & Ray Path')

#         axes[1, 1].plot(raypath[:, 0], raypath[:, 1], 'w')

#         log_image(fig)
#         plt.close()
    
#     et = time.process_time()
#     res = et - st
#     hours, minutes, seconds = convert(res)
#     log_message(f"[eikonal_ray_path.py]: Execution time: {int(hours)} hours, {int(minutes)} Minutes, {int(seconds)} seconds")
    
#     return raylength, raypath

# =================================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# import time
# from logger import log_message, log_image
# from TimeMeasurement.time_measurement import convert

# def eikonal_ray_path(x, y, v, S, R, tmap, doPlot=False):
#     """
#     Computes the ray path and ray length based on the Eikonal equation.
    
#     Parameters:
#     x, y: 1D arrays defining the grid
#     v: velocity map
#     S: tuple (sx, sy) - source coordinates
#     R: tuple (rx, ry) - receiver coordinates
#     tmap: travel time map
#     doPlot: Boolean flag to enable plotting
    
#     Returns:
#     raylength: Total ray path length
#     raypath: List of (x, y) coordinates of the ray path
#     """
    
#     st = time.process_time()
    
#     x = np.asarray(x, dtype=np.float64).flatten()
#     y = np.asarray(y, dtype=np.float64).flatten()
    
#     dx = abs(x[1] - x[0]) if len(x) > 1 else abs(x[0])
#     dy = abs(y[1] - y[0]) if len(y) > 1 else abs(y[0])
    
#     xx, yy = np.meshgrid(x, y, indexing='xy')  # Ensure MATLAB-style indexing
    
#     U, V = np.gradient(tmap, dx, dy)
#     U, V = -U, -V  # Reverse gradients to follow decreasing travel time
    
#     S = np.asarray(S, dtype=np.float64).flatten()
#     R = np.asarray(R, dtype=np.float64).flatten()
    
#     if S.shape != (2,):
#         raise ValueError(f"Unexpected shape for S: {S.shape}, expected (2,)")
#     if R.shape != (2,):
#         raise ValueError(f"Unexpected shape for R: {R.shape}, expected (2,)")
    
#     log_message(f"[eikonal_ray_path.py]: S: {S}, R: {R}")
    
#     raypath = [R]
#     current_pos = R.copy()
#     max_steps = 5000
#     step_size = min(dx, dy) / 4  # Reduce step size for better accuracy
    
#     for step in range(max_steps):
#         u_interp = griddata((xx.ravel(), yy.ravel()), U.ravel(), (current_pos[0], current_pos[1]), method='linear', fill_value=np.nan)
#         v_interp = griddata((xx.ravel(), yy.ravel()), V.ravel(), (current_pos[0], current_pos[1]), method='linear', fill_value=np.nan)
        
#         if np.isnan(u_interp) or np.isnan(v_interp):
#             log_message(f"[eikonal_ray_path.py]: Stopping due to NaN in velocity field at step {step}.")
#             break
        
#         velocity_magnitude = np.linalg.norm([u_interp, v_interp])
#         if velocity_magnitude < 1e-4:  # Adjusted threshold
#             log_message(f"[eikonal_ray_path.py]: Stopping due to low velocity at step {step}.")
#             break
        
#         adaptive_step = step_size / (velocity_magnitude + 1e-6)  # Adaptive step size
#         current_pos += adaptive_step * np.array([u_interp, v_interp])
#         raypath.append(current_pos.copy())
        
#         if np.linalg.norm(current_pos - S) < step_size:
#             log_message(f"[eikonal_ray_path.py]: Reached source at step {step}.")
#             break
    
#     raypath = np.array(raypath)
#     raylength = np.sum(np.sqrt(np.diff(raypath[:, 0])**2 + np.diff(raypath[:, 1])**2))
        
#     if doPlot:
#         fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
#         axes[0, 0].imshow(U, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[0, 0].set_title('U Gradient')

#         axes[0, 1].imshow(V, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[0, 1].set_title('V Gradient')

#         axes[1, 0].imshow(tmap, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[1, 0].set_title('Travel Time Map')

#         axes[1, 1].imshow(v, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[1, 1].set_title('Velocity Map & Ray Path')

#         axes[1, 1].plot(raypath[:, 0], raypath[:, 1], 'w')

#         log_image(fig)
#         plt.close()
    
#     et = time.process_time()
#     res = et - st
#     hours, minutes, seconds = convert(res)
#     log_message(f"[eikonal_ray_path.py]: Execution time: {int(hours)} hours, {int(minutes)} Minutes, {int(seconds)} seconds")
    
#     return raylength, raypath

# ==============================================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# import time
# from logger import log_message, log_image
# from TimeMeasurement.time_measurement import convert

# def eikonal_ray_path(x, y, v, S, R, tmap, doPlot=False):
#     """
#     Computes the ray path and ray length based on the Eikonal equation.

#     Parameters:
#     x, y: 1D arrays defining the grid
#     v: velocity map
#     S: tuple (sx, sy) - source coordinates
#     R: tuple (rx, ry) - receiver coordinates
#     tmap: travel time map
#     doPlot: Boolean flag to enable plotting

#     Returns:
#     raylength: Total ray path length
#     raypath: List of (x, y) coordinates of the ray path
#     """

#     # Start measuring execution time
#     st = time.process_time()
#     #log_message("[eikonal_ray_path.py]: Starting computation.")

#     # Ensure `x` and `y` are 1D numpy arrays
#     x = np.asarray(x, dtype=np.float64).flatten()
#     y = np.asarray(y, dtype=np.float64).flatten()

#     # Compute grid spacing dx and dy (ensure correct ordering)
#     dx = abs(x[1] - x[0]) if len(x) > 1 else abs(x[0])
#     dy = abs(y[1] - y[0]) if len(y) > 1 else abs(y[0])

#     # Create meshgrid using MATLAB-like ordering
#     xx, yy = np.meshgrid(x, y, indexing='xy')  # MATLAB indexing

#     # Compute gradients (MATLAB computes ∇t directly)
#     U, V = np.gradient(tmap, dx, dy)
#     U, V = -U, -V  # Reverse gradients to follow decreasing travel time

#     # Ensure `S` and `R` are properly extracted coordinate pairs
#     S = np.asarray(S, dtype=np.float64).flatten()
#     R = np.asarray(R, dtype=np.float64).flatten()

#     if S.shape != (2,):
#         raise ValueError(f"Unexpected shape for S: {S.shape}, expected (2,)")

#     if R.shape != (2,):
#         raise ValueError(f"Unexpected shape for R: {R.shape}, expected (2,)")

#     log_message(f"[eikonal_ray_path.py]: S: {S}, R: {R}")

#     # Initialize ray path starting at receiver
#     raypath = [R]

#     # Explicit Euler integration for streamline tracing (replacing odeint)
#     current_pos = R.copy()
#     max_steps = 5000  # Prevent infinite loops
#     step_size = min(dx, dy) / 2  # Use a small step size

#     # log_message("[eikonal_ray_path.py]: Starting explicit Euler integration.")

#     for step in range(max_steps):
#         # Interpolate velocity field at current position
#         u_interp = griddata((xx.ravel(), yy.ravel()), U.ravel(), (current_pos[0], current_pos[1]), method='linear', fill_value=0)
#         v_interp = griddata((xx.ravel(), yy.ravel()), V.ravel(), (current_pos[0], current_pos[1]), method='linear', fill_value=0)

#         # Stop if velocity is too small (prevents infinite loops)
#         if np.linalg.norm([u_interp, v_interp]) < 1e-6:
#             log_message(f"[eikonal_ray_path.py]: Stopping path calculation due to low velocity at step {step}.")
#             break

#         # Move in the negative gradient direction
#         current_pos = current_pos + step_size * np.array([u_interp, v_interp])

#         # Append new position to path
#         raypath.append(current_pos.copy())

#         # Stop if close to the source
#         if np.linalg.norm(current_pos - S) < step_size:
#             log_message(f"[eikonal_ray_path.py]: Stopping, reached source at step {step}.")
#             break

#     # Convert raypath to NumPy array
#     raypath = np.array(raypath)

#     # Compute ray length
#     raylength = np.sum(np.sqrt(np.diff(raypath[:, 0])**2 + np.diff(raypath[:, 1])**2))

#     # Plot if requested
#     if doPlot:
#         fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
#         axes[0, 0].imshow(U, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[0, 0].set_title('U Gradient')

#         axes[0, 1].imshow(V, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[0, 1].set_title('V Gradient')

#         axes[1, 0].imshow(tmap, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[1, 0].set_title('Travel Time Map')

#         axes[1, 1].imshow(v, extent=[x.min(), x.max(), y.min(), y.max()]) #, origin='lower')
#         axes[1, 1].set_title('Velocity Map & Ray Path')

#         axes[1, 1].plot(raypath[:, 0], raypath[:, 1], 'w')

#         log_image(fig)
#         plt.close()

#     # Measure and log execution time
#     et = time.process_time()  # End time
#     res = et - st  # Total execution time in seconds
#     hours, minutes, seconds = convert(res)
#     log_message(f"[eikonal_ray_path.py]: Execution time: {int(hours)} hours, {int(minutes)} Minutes, {int(seconds)} seconds")
#     log_message('.')

#     return raylength, raypath

