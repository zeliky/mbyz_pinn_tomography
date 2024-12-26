import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def eikonal_loss(
        pred_tof,  # [B, ..., H, W]  normalized T in [0..1]
        sos_map,  # [B, 1, H, W] or [1, 1, H, W], normalized c in [0..1]
        T_min_ms=0.0,  # in milliseconds, e.g. 0
        T_max_ms=100.0,  # in milliseconds, e.g. 100
        c_min_mm_us=0.14,  # e.g. 0.45 mm/us
        c_max_mm_us=0.145,  # e.g. 0.455 mm/us
        pixel_size_m=1e-3  # 1 pixel = 1 mm => 1e-3 m
    ):
    """
    pred_tof: (B, n_src, H, W)
    sos_map:  (H, W) or (B, 1, H, W)
    returns:  scalar (tensor) for PDE loss
    """
    eps = 1e-8
    pde_losses = []

    # Iterate over each source channel
    n_src = pred_tof.shape[1]
    for i in range(n_src):
        # pred_tof_i shape: [B, 1, H, W]
        pred_tof_i = pred_tof[:, i : i+1, :, :]
        T_ms = pred_tof_i * (T_max_ms - T_min_ms) + T_min_ms  # [B, ..., H, W] in ms
        T_s = T_ms * 1e-3  # [B, ..., H, W] in second

        c_mm_us = sos_map * (c_max_mm_us - c_min_mm_us) + c_min_mm_us
        c_m_s = c_mm_us * 1e3

        # Finite differences
        dx = T_s[:, :, :, 1:] - T_s[:, :, :, :-1] / pixel_size_m   # [B,1,H,W-1]
        dy = T_s[:, :, 1:, :] - T_s[:, :, :-1, :] / pixel_size_m  # [B,1,H-1,W]

        # Crop them so shapes match for sqrt:
        #   dx has shape (B,1,H,W-1)
        #   dy has shape (B,1,H-1,W)
        # We'll do something like
        dx_cropped = dx[:, :, :-1, :]    # [B,1,H-1,W-1]
        dy_cropped = dy[:, :, :, :-1]    # [B,1,H-1,W-1]

        grad_mag = torch.sqrt(dx_cropped**2 + dy_cropped**2 + eps)  # [B,1,H-1,W-1]

        # Similarly, crop the sos map
        sos_cropped = c_m_s[:, :, : grad_mag.shape[2], : grad_mag.shape[3]] # => [B,1,H-1,W-1]

        # PDE residual = |grad T| - 1/c
        residual = grad_mag - 1.0 / (sos_cropped + eps)

        pde_loss_i = torch.mean(residual**2)
        pde_losses.append(pde_loss_i)

    # Sum or average PDE loss across all sources
    return sum(pde_losses) / n_src
