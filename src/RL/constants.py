import torch

# Speed of Sound (SoS) value bins for discretization
C_BINS = torch.tensor([
    0.1, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 
    1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4
])  # cm/µs

# Default accuracy thresholds
DEFAULT_RECEIVER_ACCURACY_THRESHOLD = 0.8  # 80% of receivers must be accurate
DEFAULT_RECEIVER_TOF_THRESHOLD = 0.2      # 20% error threshold for each receiver
DEFAULT_SOURCE_COMPLETION_THRESHOLD = 0.8  # 80% of sources must be done

# Physics-informed training constants
PHYSICS_LOSS_WEIGHT = 0.1  # Weight for physics loss in total loss
EIKONAL_TOLERANCE = 1e-6   # Small constant to prevent division by zero in gradient magnitude
WAVE_SPEED_MIN = 0.1      # Minimum allowed wave speed (cm/µs)
WAVE_SPEED_MAX = 2.4      # Maximum allowed wave speed (cm/µs) 