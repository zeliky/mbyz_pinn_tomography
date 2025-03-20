import matlab.engine
import numpy as np
import os

def my_msfm2d(F, SourcePoints, use_second=True, use_cross=True):
    """
    Calls the MATLAB MEX function msfm2d.mexw64 from Python.

    Parameters:
    F : 2D NumPy array (speed image)
    SourcePoints : (2, N) NumPy array (source locations)
    use_second : Boolean, whether to use second-order derivatives.
    use_cross : Boolean, whether to use cross neighbors.

    Returns:
    T_np : 2D NumPy array of computed travel times.
    """

    # Start MATLAB Engine
    eng = matlab.engine.start_matlab()

    # Define the path to the `mex` sub-folder
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    mex_path = os.path.join(script_dir, "mex")  # Path to mex sub-folder

    # Add `mex` folder to MATLAB path
    eng.addpath(mex_path, nargout=0)

    # Convert NumPy arrays to MATLAB format
    F_matlab = matlab.double(F.tolist())  
    SourcePoints_matlab = matlab.double(SourcePoints.tolist())

    # Call the MEX function through MATLAB
    T_matlab = eng.msfm2d(F_matlab, SourcePoints_matlab, use_second, use_cross)

    # Convert MATLAB output back to NumPy
    T_np = np.array(T_matlab)

    # Stop MATLAB Engine
    eng.quit()

    return T_np

