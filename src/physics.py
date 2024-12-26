import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def eikonal_loss(pred_tof, sos_map):
    """
    pred_tof: (B, n_src, H, W)
    sos_map:  (H, W) or (B, 1, H, W)
    returns:  scalar (tensor) for PDE loss
    """
    # If sos_map is [H, W], expand to [B, 1, H, W] to match batch shape
    if sos_map.dim() == 2:
        sos_map = sos_map.unsqueeze(0).unsqueeze(0)  # => [1,1,H,W]
        sos_map = sos_map.expand(pred_tof.shape[0], 1, *sos_map.shape[2:]) # => [B,1,H,W]
        T_min_ms = 0.0,  # in milliseconds, e.g. 0
        T_max_ms = 100.0,  # in milliseconds, e.g. 100
        c_min_mm_us = 0.45,  # e.g. 0.45 mm/us
        c_max_mm_us = 0.455,  # e.g. 0.455 mm/us
        pixel_size_m = 1e-3  # 1 pixel = 1 mm => 1e-3 m

    eps = 1e-8
    pde_losses = []

    # Iterate over each source channel
    n_src = pred_tof.shape[1]
    for i in range(n_src):
        # pred_tof_i shape: [B, 1, H, W]
        pred_tof_i = pred_tof[:, i : i+1, :, :]

        # Finite differences
        dx = pred_tof_i[:, :, :, 1:] - pred_tof_i[:, :, :, :-1]   # [B,1,H,W-1]
        dy = pred_tof_i[:, :, 1:, :] - pred_tof_i[:, :, :-1, :]   # [B,1,H-1,W]

        # Crop them so shapes match for sqrt:
        #   dx has shape (B,1,H,W-1)
        #   dy has shape (B,1,H-1,W)
        # We'll do something like
        dx_cropped = dx[:, :, :-1, :]    # [B,1,H-1,W-1]
        dy_cropped = dy[:, :, :, :-1]    # [B,1,H-1,W-1]

        grad_mag = torch.sqrt(dx_cropped**2 + dy_cropped**2 + eps)  # [B,1,H-1,W-1]

        # Similarly, crop the sos map
        sos_cropped = sos_map[:, :, : grad_mag.shape[2], : grad_mag.shape[3]]
        # => [B,1,H-1,W-1]

        # PDE residual = |grad T| - 1/c
        residual = grad_mag - 1.0 / (sos_cropped + eps)

        pde_loss_i = torch.mean(residual**2)
        pde_losses.append(pde_loss_i)

    # Sum or average PDE loss across all sources
    return sum(pde_losses) / n_src
