import numpy as np
from msfm2d import msfm2d

# Test with small grid
if __name__ == "__main__":
    rows, cols = 5, 5
    V = np.ones((rows, cols), dtype=np.float64)
    source_points = np.array([[2, 1]], dtype=np.int32).reshape(-1, 2)  # Ensure shape (num_sources, 2)
    
    T = msfm2d(V, source_points)
    print('T in python')
    print(T)
