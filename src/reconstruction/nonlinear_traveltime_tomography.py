# nonlinear_traveltime_tomography.py
import numpy as np
import scipy.io
import time
from eikonal_traveltime import eikonal_traveltime

from sart import sart

from logger import log_message
from visualization import plot_tof, plot_sos

from calculateL import CalculateL

def nonlinear_traveltime_tomography(D_in):
    start_time = time.time()
    
    D = D_in
    t_obs = D['t_obs']
    title = 'Observed'
    plot_tof(t_obs, title)
    x, y, z = D['x'], D['y'], D['z']
    log_message(f'[nonlinear... .py]: x.shape = {x.shape}, y.shape = {y.shape}, z.shape = {z.shape}')
    
    xs_receivers, ys_receivers = D['xs_receivers'], D['ys_receivers']
    xs_sources, ys_sources = D['xs_sources'], D['ys_sources']
    log_message(f'[nonlinear... .py]: xs_sources.shape = {xs_sources.shape}, ys_sources.shape = {ys_sources.shape}')
    
    number_of_receivers = len(xs_receivers)
    number_of_sources = len(xs_sources)
    
    # ---------------- calculate TOF from exact SOS ----------------
    dx = x[:,1]-x[:,0]
    dy = y[:,1]-y[:,0]
    V = D['V']
    mystr = 'Exact'
    plot_sos(V, dx, dy, mystr)
    log_message(f'[nonlinear... .py]: calling calculate_t for {mystr}')
    t_calc = calculate_t(V, D)
    
    title = 'Calculated from exact'
    plot_tof(t_calc, title)
    log_message(f'[nonlinear... .py]: finished {title} by calculate_t')        
    
    t_calc_exact = t_calc.copy()
    
    delta_t = (t_obs - t_calc_exact)
    title = 't_obs - t_calc_exact'
    plot_tof(t_obs - t_calc_exact, title)

    if D.get('use_input_s0', False):
        log_message("[nonlinear... .py]: Using D_in.s for s0")
        s0 = D['V']
    else:
        s0 = np.ones((len(y[0]), len(x[0])))
        log_message(f"[nonlinear... .py]: Using ones for s0, y.shape[1], x.shape[1] = {y.shape[1]}, {x.shape[1]}")
    
    s = s0.copy()
    mystr = 'Initial'
    plot_sos(s, dx, dy, mystr)
    
    log_message(f'[nonlinear_traveltime_tomography.py]: s.shape = {s.shape}')
    
    doPlotRaypathsInCalculateL = False
    # Assemble input_to_L dictionary
    input_to_L = {
        "x": x,
        "y": y,
        "z": z,
        "xs_receivers": xs_receivers,
        "ys_receivers": ys_receivers,
        "xs_sources": xs_sources,
        "ys_sources": ys_sources,
        "doPlotRaypathsInCalculateL": doPlotRaypathsInCalculateL
    }
    
    for iteration in range(3, 21): 
        # log_message(f'[nonlinear... .py]: calling calculate_t for iteraion {iteration}')    
        t_calc = calculate_t(s, D)       
        title = f'Iteration {iteration} of Calculated'
        plot_tof(t_calc, title)
        # log_message(f'[nonlinear... .py]: finished {title} TOF')
        
        # log_message(f'[nonlinear... .py]: type(t_calc_exact): {type(t_calc_exact)}, type(t_calc): {type(t_calc)}')        
        #delta_t = t_obs - t_calc
        delta_t = t_calc_exact - t_calc
        title = f'Iteration {iteration} delta_t (t_calc_exact - t_calc)'
        plot_tof(delta_t, title)
        
        b = delta_t.flatten()
        
        log_message(f'[nonlinear... .py]: calling calculateL for iteraion {iteration}')
        log_message('.') 
        RL_calc, L = CalculateL(s, input_to_L)
        log_message(f'[nonlinear... .py]: finished calculateL for iteraion {iteration}')
        
        A = -L
        
        #A = -L.reshape((number_of_sources * number_of_receivers, -1))  # Auto-calculate last dim
        # expected_A_shape = (number_of_sources * number_of_receivers, len(y) * len(x))
        # A = -L.reshape(expected_A_shape)  # Ensure correct reshaping

        # log_message(f"[nonlinear...]: A.shape = {A.shape}, expected_A_shape = {expected_A_shape}, b.shape = {b.shape}")
        # assert A.shape[0] == b.shape[0], f"Mismatch: A.shape = {A.shape}, b.shape = {b.shape}"
        k = 600
         
        log_message(f'[nonlinear... .py]: input to sart. A.shape = {A.shape}, b.shape = {b.shape}, k = {k}, ')
       
        options = {'lambda': 'psi2mod', 'stoprule': 'NCP', 'nonneg': True}
        #xk, info_sart = sart(A, b, k, np.zeros_like(b), options)
        xk, info_sart = sart(A, b, k, x0=None, options=options)
        #s += xk.reshape(len(y), len(x))
        s += xk.reshape(len(y.squeeze()), len(x.squeeze()))
        
        mystr = f'Intermediate s of iteration {iteration}'
        plot_sos(s, dx, dy, mystr)
        
        if iteration % 1 == 0:
            log_message(f"[nonlinear... .py]: Iteration {iteration}: Updating visualization...")
    
    mystr = 'Final s'
    plot_sos(s, dx, dy, mystr)
     
    D['s'] = s
    D['elapsed_time'] = time.time() - start_time
    log_message(f"[nonlinear... .py]: Overall Elapsed Time: {D['elapsed_time']} sec")
    
    return D

def calculate_t(V, D):
    """
    Computes the travel time `tcalc` from sources to receivers using the eikonal solver.

    Parameters:
        V (ndarray): The velocity field (speed map).
        D (dict): Dictionary containing the grid and sensor information:
            - x, y, z: Grid coordinates
            - xs_receivers, ys_receivers: Receiver positions
            - xs_sources, ys_sources: Source positions

    Returns:
        tcalc (ndarray): Travel time matrix of shape (num_sources, num_receivers).
    """

    # Extract grid and sensor positions
    x, y, z = np.asarray(D['x']).squeeze(), np.asarray(D['y']).squeeze(), np.asarray(D['z']).squeeze()
    xs_receivers, ys_receivers = np.asarray(D['xs_receivers']).squeeze(), np.asarray(D['ys_receivers']).squeeze()
    xs_sources, ys_sources = np.asarray(D['xs_sources']).squeeze(), np.asarray(D['ys_sources']).squeeze()

    num_receivers = len(xs_receivers)
    num_sources = len(xs_sources)

    # Ensure `R` (receivers) is correctly shaped
    R = np.column_stack((xs_receivers, ys_receivers))  # Shape (num_receivers, 2)
    tcalc = np.zeros((num_sources, num_receivers))

    # Compute travel times for each source
    for is_ in range(num_sources):
        x_source = float(xs_sources[is_])  # Ensure scalar value
        y_source = float(ys_sources[is_])  # Ensure scalar value

        S1 = np.column_stack((np.full(num_receivers, x_source),
                              np.full(num_receivers, y_source)))  # Shape (num_receivers, 2)

        # Solve the eikonal equation to get the travel time
        try:
            tmap, t = eikonal_traveltime(x, y, z, V, S1, R)
            tcalc[is_, :] = t.squeeze()
        except IndexError as e:
            print(f"[ERROR] Source {is_}: Issue with msfm2d input. Check source coordinates.")
            raise e

    return tcalc

