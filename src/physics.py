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


def boundary_loss_v1(pred_tof, known_tof, src_loc, rec_loc):
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


def boundary_loss_v2(pred_tof, known_tof, positions_mask):
    """
    pred_tof:    (S, H, W) => predicted TOF from each of S sources
    known_tof:   (S, R)    => known TOF from source s to receiver r
    positions_mask: (H, W) => integer codes:
        = s for source s,
        = 100 + r for receiver r, (for example)
        = anything else for "no constraint" region
    """
    device = pred_tof.device
    S, H, W = pred_tof.shape
    R = known_tof.shape[1]

    loss = torch.tensor(0.0, device=device)
    count = 0

    # 1) Source condition: If positions_mask[y,x] == s, enforce T(s,y,x)=0
    for s in range(S):
        # Find all (y,x) where mask==s
        # Usually you have exactly 1 pixel for each source, but let's be general:
        coords = (positions_mask == s).nonzero(as_tuple=False)  # shape [N,2], each row=[y,x]
        for c in coords:
            y, x = c
            loss += (pred_tof[s, y, x] - 0.0) ** 2
            count += 1

    # 2) Receiver condition: If positions_mask[y,x] == (100 + r), then
    #   compare pred_tof[s, y, x] to known_tof[s, r]
    for r in range(R):
        coords = (positions_mask == (100 + r)).nonzero(as_tuple=False)
        for c in coords:
            y, x = c
            # sum over s => we expect pred_tof[s,y,x] == known_tof[s,r]
            for s in range(S):
                #print(f"({s}, {y}, {x}) ---  {pred_tof[s, y, x]} -{known_tof[s, r]}")
                loss += (pred_tof[s, y, x] - known_tof[s, r]) ** 2
                count += 1

    return loss / (count if count > 0 else 1)






def boundary_loss(pred_tof, known_tof, positions_mask):
    """
    pred_tof:    (S, H, W)  => predicted TOF for each of S sources
    known_tof:   (S, R)     => known/measured TOF for source s to receiver r
                  shape[0] = S, shape[1] = R
    positions_mask: (H, W)  => integer codes:
        = s            for source s  (enforce pred_tof[s,y,x] = 0)
        = 100 + r      for receiver r (enforce pred_tof[s,y,x] = known_tof[s,r])
        = anything else => no constraint at that pixel
    Returns:
        A scalar MSE loss enforcing source=0 and receiver=known_tof constraints.
    """
    device = pred_tof.device
    S, H, W = pred_tof.shape
    R = known_tof.shape[1]  # number of receivers

    # -------------------------------------------------------------------------
    # 1) Source condition: positions_mask[y,x] == s ==> pred_tof[s,y,x] ~ 0
    #
    # We'll build a boolean mask of shape (S,H,W) that is True where
    # positions_mask[y,x] == s for each s in [0..S-1].
    # Vectorized approach:
    #   - Expand positions_mask to shape (S,H,W).
    #   - Compare with a broadcasted s_index of shape (S,1,1).
    # -------------------------------------------------------------------------
    posmask_expanded = positions_mask.unsqueeze(0).expand(S, H, W)  # (S,H,W)
    s_index = torch.arange(S, device=device).view(-1,1,1)           # (S,1,1)

    # True where positions_mask == s
    source_mask = (posmask_expanded == s_index)  # (S,H,W), bool

    # We want pred_tof[s,y,x] ~ 0 at those pixels:
    # => MSE => (pred_tof[s,y,x] - 0)^2
    diff_src = pred_tof**2
    masked_src = diff_src * source_mask  # zeroes everywhere except source pixels
    sum_src = masked_src.sum()
    count_src = source_mask.sum()

    # -------------------------------------------------------------------------
    # 2) Receiver condition: positions_mask[y,x] == (100+r)
    #     => pred_tof[s,y,x] ~ known_tof[s,r] for *all* s
    #
    # We'll build a boolean mask for each receiver r in [0..R-1], where
    # positions_mask[y,x] == (100 + r). Then we compare pred_tof[s,y,x]
    # to known_tof[s,r].
    # -------------------------------------------------------------------------
    # Create a [R,H,W] mask: for each r, rec_mask[r,y,x] = True if mask[y,x]=(100+r)
    # Steps:
    #   - r_index => shape (R,)
    #   - positions_mask => shape (H,W), compare with (100 + r_index) => shape (R,H,W)
    r_index = torch.arange(R, device=device)
    mask_2d = positions_mask.unsqueeze(0)                  # (1,H,W)
    rec_mask_2d = (mask_2d == (100 + r_index.view(-1,1,1))) # (R,H,W), bool

    # Expand to shape (S,R,H,W), so each source sees the same set of receiver pixels.
    rec_mask_4d = rec_mask_2d.unsqueeze(0).expand(S, -1, H, W)  # (S,R,H,W)

    # Expand pred_tof => (S,1,H,W) => (S,R,H,W)
    pred_tof_exp = pred_tof.unsqueeze(1).expand(-1, R, -1, -1)  # (S,R,H,W)

    # Expand known_tof => (S,R) => (S,R,1,1) => (S,R,H,W)
    known_tof_exp = known_tof.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)  # (S,R,H,W)

    # Compare => (pred_tof_exp - known_tof_exp)^2
    diff_rec = (pred_tof_exp - known_tof_exp)**2
    masked_rec = diff_rec * rec_mask_4d
    sum_rec = masked_rec.sum()
    count_rec = rec_mask_4d.sum()

    # -------------------------------------------------------------------------
    # Combine source & receiver constraints
    # -------------------------------------------------------------------------
    total_loss = sum_src + sum_rec
    total_count = count_src + count_rec
    loss = total_loss / (total_count + 1e-9)

    return loss







