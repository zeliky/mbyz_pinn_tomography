import numpy as np
from logger import log_message, log_image

def sart(A, b, K, x0=None, options=None):
    """
    Simultaneous Algebraic Reconstruction Technique (SART) method.

    Args:
        A (ndarray): (m, n) system matrix.
        b (ndarray): (m, 1) right-hand side.
        K (int): Number of iterations.
        x0 (ndarray, optional): (n, 1) starting vector.
        options (dict, optional): Parameters for relaxation and stopping.

    Returns:
        X (ndarray): Solution vector.
        info (tuple): (stopping condition, iterations).
    """

    m, n = A.shape

    # Ensure b is a column vector (m,1)
    b = b.reshape((m, 1))
    log_message(f'm = {m}, n = {n}, b.shape = {b.shape}')
    
    # Initialize x0 as a column vector (n,1)
    if x0 is None:
        x0 = np.zeros((n, 1))  # Fix: Explicitly use n, not b.shape
    elif x0.shape != (n, 1):
        raise ValueError(f"Expected x0 of shape ({n},1), got {x0.shape}")

    AT = A.T
    rxk = b - A @ x0

    # Compute weight matrices (faithful to MATLAB)
    W = np.reciprocal(np.maximum(A.sum(axis=1, keepdims=True), 1e-8))  # (m,1)
    Vm = np.reciprocal(np.maximum(A.sum(axis=0, keepdims=True), 1e-8)).reshape((n,1))  # (n,1)

    # Retrieve options
    lambda_val = 1 # forcing a default value of 1 instead of: options.get('lambda', 1) if options else 1
    stoprule = options.get('stoprule', 'none') if options else 'none'
    nonneg = options.get('nonneg', False) if options else False

    xk = x0.copy()

    log_message(f'[sart.py]: lambda_val = {lambda_val}, Vm.shape = {Vm.shape}, AT.shape = {AT.shape}, W.shape = {W.shape}, rxk.shape = {rxk.shape}')
    for k in range(K):
        # Compute update step
        delta_x = lambda_val * (Vm * (AT @ (W * rxk)))
        xk1 = xk + delta_x

        if nonneg:
            xk1 = np.maximum(xk1, 0)

        rxk1 = b - A @ xk1

        # Stopping criteria (ensuring MATLAB equivalence)
        if stoprule == 'DP' and np.linalg.norm(rxk1) <= 1e-6:
            return xk1, (2, k + 1)
        elif stoprule == 'ME' and np.linalg.norm(rxk1 - rxk) / np.linalg.norm(rxk1) < 1e-6:
            return xk1, (3, k + 1)
        elif stoprule == 'NCP' and k > 0 and np.linalg.norm(rxk1) > np.linalg.norm(rxk):
            return xk1, (1, k + 1)

        xk, rxk = xk1, rxk1

    return xk, (0, K)

# import numpy as np

# def sart(A, b, K, x0=None, options=None):
#     """
#     Simultaneous Algebraic Reconstruction Technique (SART) method

#     Args:
#         A (ndarray): m x n system matrix.
#         b (ndarray): m x 1 vector containing the right-hand side.
#         K (int): Number of iterations.
#         x0 (ndarray, optional): n x 1 starting vector. Defaults to zero vector.
#         options (dict, optional): Dictionary with parameters:
#             - lambda: Relaxation parameter or method ('line', 'psi1', etc.)
#             - stoprule: Stopping criterion {'none', 'DP', 'ME', 'NCP'}
#             - nonneg: Boolean for enforcing nonnegativity.

#     Returns:
#         X (ndarray): Solution vector.
#         info (tuple): (stopping condition, number of iterations).
#     """
#     m, n = A.shape
    
#     # Ensure b is a column vector (m,1)
#     b = b.reshape((m, 1))
    
#     # Initialize x0 as a column vector (n,1)
#     if x0 is None:
#         x0 = np.zeros((n, 1))
#     elif x0.shape != (n, 1):
#         x0 = x0.reshape((n, 1))
    
#     AT = A.T
#     rxk = b - A @ x0
    
#     # Compute weight matrices (faithful to MATLAB implementation)
#     W = 1 / np.maximum(A.sum(axis=1, keepdims=True), 1e-8)
#     Vm = 1 / np.maximum(A.sum(axis=0, keepdims=True), 1e-8)
    
#     # Retrieve options
#     lambda_val = options.get('lambda', 1) if options else 1
#     stoprule = options.get('stoprule', 'none') if options else 'none'
#     nonneg = options.get('nonneg', False) if options else False
    
#     xk = x0.copy()
    
#     for k in range(K):
#         # Compute update step
#         delta_x = lambda_val * (Vm * (AT @ (W * rxk)))
#         xk1 = xk + delta_x
        
#         # Apply nonnegativity constraint if needed
#         if nonneg:
#             xk1 = np.maximum(xk1, 0)
        
#         # Compute new residual
#         rxk1 = b - A @ xk1
        
#         # MATLAB-like stopping conditions (checked BEFORE updating xk)
#         if stoprule == 'DP' and np.linalg.norm(rxk1) <= 1e-6:
#             return xk1, (2, k + 1)
#         elif stoprule == 'ME' and np.linalg.norm(rxk1 - rxk) / np.linalg.norm(rxk1) < 1e-6:
#             return xk1, (3, k + 1)
#         elif stoprule == 'NCP' and k > 0 and np.linalg.norm(rxk1) > np.linalg.norm(rxk):
#             return xk1, (1, k + 1)
        
#         # Update variables for next iteration
#         xk, rxk = xk1, rxk1
    
#     return xk, (0, K)

# =================================================================================
# import numpy as np

# def sart(A, b, K, x0=None, options=None):
#     """
#     Simultaneous Algebraic Reconstruction Technique (SART) method

#     Args:
#         A (ndarray): m x n system matrix.
#         b (ndarray): m x 1 vector containing the right-hand side.
#         K (int): Number of iterations.
#         x0 (ndarray, optional): n x 1 starting vector. Defaults to zero vector.
#         options (dict, optional): Dictionary with parameters:
#             - lambda: Relaxation parameter or method ('line', 'psi1', etc.)
#             - stoprule: Stopping criterion {'none', 'DP', 'ME', 'NCP'}
#             - nonneg: Boolean for enforcing nonnegativity.

#     Returns:
#         X (ndarray): Solution vector.
#         info (tuple): (stopping condition, number of iterations).
#     """
#     m, n = A.shape
    
#     # Ensure b is correctly shaped
#     b = b.reshape(-1, 1)  # Ensure b is a column vector (m,1)
#     if b.shape[0] != m:
#         raise ValueError("The size of A and b do not match")
    
#     # Initialize x0 correctly as a column vector (n,1)
#     if x0 is None:
#         x0 = np.zeros((n, 1))
#     elif x0.shape[0] != n:
#         raise ValueError(f"The size of x0 ({x0.shape}) does not match the problem (expected ({n},1))")
    
#     AT = A.T
#     rxk = b - A @ x0
    
#     # Compute weight matrices (Ensure correct column vector shape)
#     W = 1 / np.maximum(A.sum(axis=1, keepdims=True), 1e-8)  # Shape (m,1)
#     Vm = 1 / np.maximum(A.sum(axis=0, keepdims=True), 1e-8)  # Shape (n,1)
    
#     # Retrieve options
#     lambda_val = options.get('lambda', 1) if options else 1
#     stoprule = options.get('stoprule', 'none') if options else 'none'
#     nonneg = options.get('nonneg', False) if options else False
    
#     xk = x0.copy()
    
#     for k in range(K):
#         # Compute next iteration (Ensure correct broadcasting)
#         xk1 = xk + lambda_val * (Vm * (AT @ (W * rxk)))
        
#         # Apply nonnegativity constraint
#         if nonneg:
#             xk1 = np.maximum(xk1, 0)
        
#         # Compute new residual
#         rxk1 = b - A @ xk1
        
#         # Stopping conditions (adjusted for zero-based indexing in Python)
#         if stoprule == 'DP' and np.linalg.norm(rxk1) <= 1e-6:
#             return xk1, (2, k + 1)  # Adjust for MATLAB's 1-based indexing
#         elif stoprule == 'ME' and np.linalg.norm(rxk1 - rxk) / np.linalg.norm(rxk1) < 1e-6:
#             return xk1, (3, k + 1)
#         elif stoprule == 'NCP' and k > 0 and np.linalg.norm(rxk1) > np.linalg.norm(rxk):
#             return xk1, (1, k + 1)
        
#         # Update variables
#         xk, rxk = xk1, rxk1
    
#     return xk, (0, K)

# ======================================================================================
# import numpy as np

# def sart(A, b, K, x0=None, options=None):
#     """
#     Simultaneous Algebraic Reconstruction Technique (SART) method

#     Args:
#         A (ndarray): m x n system matrix.
#         b (ndarray): m x 1 vector containing the right-hand side.
#         K (int): Number of iterations.
#         x0 (ndarray, optional): n x 1 starting vector. Defaults to zero vector.
#         options (dict, optional): Dictionary with parameters:
#             - lambda: Relaxation parameter or method ('line', 'psi1', etc.)
#             - stoprule: Stopping criterion {'none', 'DP', 'ME', 'NCP'}
#             - nonneg: Boolean for enforcing nonnegativity.

#     Returns:
#         X (ndarray): Solution vector.
#         info (tuple): (stopping condition, number of iterations).
#     """
#     m, n = A.shape
    
#     # Ensure b is correctly shaped
#     b = b.reshape(-1, 1)  # Ensure b is a column vector (m,1)
#     if b.shape[0] != m:
#         raise ValueError("The size of A and b do not match")
    
#     # Initialize x0 correctly as a column vector (n,1)
#     if x0 is None:
#         x0 = np.zeros((n, 1))
#     elif x0.shape != (n, 1):
#         raise ValueError(f"The size of x0 ({x0.shape}) does not match the problem (expected ({n},1))")
    
#     AT = A.T
#     rxk = b - A @ x0
    
#     # Compute weight matrices (Ensure correct column vector shape)
#     W = 1 / np.maximum(A.sum(axis=1, keepdims=True), 1e-8)  # Shape (m,1)
#     Vm = 1 / np.maximum(A.sum(axis=0, keepdims=True), 1e-8)  # Shape (n,1)
    
#     # Retrieve options
#     lambda_val = options.get('lambda', 1) if options else 1
#     stoprule = options.get('stoprule', 'none') if options else 'none'
#     nonneg = options.get('nonneg', False) if options else False
    
#     xk = x0.copy()
    
#     for k in range(K):
#         # Compute next iteration (Ensure correct broadcasting)
#         xk1 = xk + lambda_val * (Vm * (AT @ (W * rxk)))
        
#         # Apply nonnegativity constraint
#         if nonneg:
#             xk1 = np.maximum(xk1, 0)
        
#         # Compute new residual
#         rxk1 = b - A @ xk1
        
#         # Stopping conditions (adjusted for zero-based indexing in Python)
#         if stoprule == 'DP' and np.linalg.norm(rxk1) <= 1e-6:
#             return xk1, (2, k + 1)  # Adjust for MATLAB's 1-based indexing
#         elif stoprule == 'ME' and np.linalg.norm(rxk1 - rxk) / np.linalg.norm(rxk1) < 1e-6:
#             return xk1, (3, k + 1)
#         elif stoprule == 'NCP' and k > 0 and np.linalg.norm(rxk1) > np.linalg.norm(rxk):
#             return xk1, (1, k + 1)
        
#         # Update variables
#         xk, rxk = xk1, rxk1
    
#     return xk, (0, K)


