import scipy.io
from nonlinear_traveltime_tomography import nonlinear_traveltime_tomography
from logger import log_message

def run_nonlinear():
    D_in = scipy.io.loadmat('../TimeOfFlightData/Time_Of_Flight_data2.mat')
    D_in['use_input_s0'] = False
    D = nonlinear_traveltime_tomography(D_in)
    return D

if __name__ == "__main__":
    result = run_nonlinear()
    log_message("[run_nonlinear.py]: Processing complete.")