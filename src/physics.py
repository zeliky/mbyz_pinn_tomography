import torch
import math
import torch.nn.functional as F
from settings import app_settings


def _to_sec(v):
    t_min_ms = app_settings.min_tof   # in milliseconds
    t_max_ms = app_settings.max_tof   # in milliseconds

    t_ms = v * (t_max_ms - t_min_ms) + t_min_ms  # [B, ..., H, W] in ms
    t_s = t_ms * 1e-3 #in second
    return t_s

def _to_mps(v):
    c_min_mm_us = app_settings.min_sos   # in mm/us
    c_max_mm_us = app_settings.max_sos   # in mm/us

    c_mm_us = v * (c_max_mm_us - c_min_mm_us) + c_min_mm_us
    c_m_s = c_mm_us * 1e3
    return c_m_s


def initial_loss(pred_tof, src_loc):
    loss = 0.0
    c = 0
    for s_idx, sl in enumerate(src_loc):
        p_tof = _to_sec(pred_tof[s_idx, sl[0], sl[1]])
        # print(f"{p_tof}")
        loss += torch.pow(p_tof,2)    # compare to 0
        c += 1
    return loss / c


def boundary_loss(pred_tof, known_tof, src_loc, rec_loc):
    loss = 0.0
    c = 0
    for s_idx, s in enumerate(src_loc):
        for r_idx, r in enumerate(rec_loc):
            k_tof = _to_sec(known_tof[s_idx, r_idx]) # known tof (obs)
            p_tof = _to_sec(pred_tof[s_idx, r[0], r[1]])  # predicted tof on s layer, receiver position
            # print(f"{k_tof - p_tof}")
            loss += torch.pow(k_tof-p_tof, 2)  # compare to original tof
            c += 1
    return loss / c

def eikonal_loss(
        pred_tof,  # [B, ..., H, W]
        sos_map,  # [B, 1, H, W] or [1, 1, H, W],

    ):
    """
    pred_tof: (B, n_src, H, W)
    sos_map:  (H, W) or (B, 1, H, W)
    returns:  scalar (tensor) for PDE loss
    """

    pixel_size_m = app_settings.pixel_to_mm
    eps = 1e-8
    pde_losses = []

    # Iterate over each source channel
    n_src = pred_tof.shape[0]
    for i in range(n_src):
        # pred_tof_i shape: [ 1, H, W]s
        pred_tof_i = pred_tof[i : i+1, :, :].squeeze()  # [  ..., H, W] in ms

        # T_ms = pred_tof_i * (T_max_ms - T_min_ms) + T_min_ms
        # t_s = T_ms * 1e-3  # [B, ..., H, W] in second
        t_s = _to_sec(pred_tof_i) # [B, ..., H, W] in second

        #c_mm_us = sos_map * (c_max_mm_us - c_min_mm_us) + c_min_mm_us
        #c_m_s = c_mm_us * 1e3
        c_m_s = _to_mps(sos_map.squeeze())


        # Finite differences --- change to torch.gradient
        #dx = t_s[ :, 1:] - t_s[ :, :-1] / pixel_size_m   # [1,H,W-1]
        #dy = t_s[ 1:, :] - t_s[:-1, :] / pixel_size_m  # [1,H-1,W]
        dy, dx = torch.gradient(
            t_s,
            spacing=(1.0, 1.0)
        )


        # Crop them so shapes match for sqrt:
        #   dx has shape (1,H,W-1)
        #   dy has shape (1,H-1,W)

        #dx_cropped = dx[ :, :-1, :]    # [1,H-1,W-1]
        #dy_cropped = dy[  :, :, :-1]    # [1,H-1,W-1]

        grad_mag = torch.sqrt(dx**2 + dy**2 + eps)  # [1,H-1,W-1]

        # crop the sos map
        #sos_cropped = c_m_s[ :, : grad_mag.shape[1], : grad_mag.shape[2]] # => [1,H-1,W-1]

        # PDE residual = |grad T| - 1/c
        residual = grad_mag - 1.0 / (c_m_s + eps)

        pde_loss_i = torch.mean(residual**2)
        pde_losses.append(pde_loss_i)

    # Sum or average PDE loss across all sources

    return sum(pde_losses) / n_src
