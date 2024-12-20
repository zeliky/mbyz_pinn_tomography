import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def compute_eikonal_loss(predicted_tof, sos_map, epsilon=1e-8):
    """
    Computes the Eikonal loss for the predicted ToF and SoS map in a numerically stable manner.

    Parameters:
    predicted_tof (torch.Tensor):
        The predicted ToF tensor of shape (C, W, H), where C is the number of sources (32).
    sos_map (torch.Tensor):
        The speed of sound map of shape (W, H).
    epsilon (float, optional):
        A small constant to prevent division by zero and numerical instability. Defaults to 1e-8.

    Returns:
    torch.Tensor:
        The computed Eikonal loss as a scalar tensor.
    """


    # Validate input dimensions
    if predicted_tof.dim() != 3:
        raise ValueError(f"predicted_tof must be a 3D tensor of shape (C, W, H), but got shape {predicted_tof.shape}")
    if sos_map.dim() != 2:
        raise ValueError(f"sos_map must be a 2D tensor of shape (W, H), but got shape {sos_map.shape}")

    C, W, H = predicted_tof.shape
    print(predicted_tof)
    predicted_tof_safe = torch.clamp(predicted_tof, min=epsilon)

    # Ensure sos_map has no zeros by clamping
    sos_map_safe = torch.clamp(sos_map, min=epsilon)  # Shape: (W, H)


    # Expand sos_map to match the number of channels in predicted_tof
    sos_map_safe = sos_map_safe.unsqueeze(0).expand_as(predicted_tof)  # Shape: (C, W, H)

    # Compute gradients using Sobel filters
    grad_x, grad_y = compute_gradients(predicted_tof)  # Both shape: (C, W, H)

    # Compute the norm of the gradient
    grad_tof_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2) + epsilon  # Shape: (C, W, H)

    # Compute the inverse of the speed of sound
    inv_sos = 1.0 / sos_map_safe  # Shape: (C, W, H)

    # Compute the Eikonal loss using Mean Squared Error
    eikonal_loss = F.mse_loss(grad_tof_norm, inv_sos)

    return eikonal_loss


def get_sobel_kernels(device, dtype):
    """
    Returns Sobel kernels for x and y directions.

    Parameters:
    - device: torch.device
    - dtype: torch.dtype

    Returns:
    - Gx: torch.Tensor of shape (1, 1, 3, 3)
    - Gy: torch.Tensor of shape (1, 1, 3, 3)
    """
    # Define Sobel kernels
    Gx = torch.tensor([[-1., 0., +1.],
                       [-2., 0., +2.],
                       [-1., 0., +1.]], device=device, dtype=dtype).view(1, 1, 3, 3)

    Gy = torch.tensor([[-1., -2., -1.],
                       [0., 0., 0.],
                       [+1., +2., +1.]], device=device, dtype=dtype).view(1, 1, 3, 3)

    return Gx, Gy


def compute_gradients(predicted_tof):
    """
    Computes the gradients of the predicted ToF tensor using Sobel filters.

    Parameters:
    - predicted_tof: torch.Tensor of shape (C, W, H)

    Returns:
    - grad_x: torch.Tensor of shape (C, W, H)
    - grad_y: torch.Tensor of shape (C, W, H)
    """
    C, W, H = predicted_tof.shape
    device = predicted_tof.device
    dtype = predicted_tof.dtype

    # Add batch dimension for conv2d: (N, C, W, H)
    predicted_tof = predicted_tof.unsqueeze(0)  # Shape: (1, C, W, H)

    # Define Sobel kernels
    Gx, Gy = get_sobel_kernels(device, dtype)

    # Repeat kernels for each channel
    Gx = Gx.repeat(C, 1, 1, 1)  # Shape: (C, 1, 3, 3)
    Gy = Gy.repeat(C, 1, 1, 1)  # Shape: (C, 1, 3, 3)

    # Apply convolution with groups=C to compute per-channel gradients
    grad_x = F.conv2d(predicted_tof, Gx, padding=1, groups=C)  # Shape: (1, C, W, H)
    grad_y = F.conv2d(predicted_tof, Gy, padding=1, groups=C)  # Shape: (1, C, W, H)

    # Remove batch dimension
    grad_x = grad_x.squeeze(0)  # Shape: (C, W, H)
    grad_y = grad_y.squeeze(0)  # Shape: (C, W, H)

    return grad_x, grad_y
