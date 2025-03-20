import torch
import torch.nn.functional as F
from settings import app_settings

def _to_sec(v):
    t_min_ms = app_settings.min_tof
    t_max_ms = app_settings.max_tof

    t_ms = v * (t_max_ms - t_min_ms) + t_min_ms
    t_s = t_ms * 1e-3
    return t_s

def _to_mps(v):
    c_min_mm_us = app_settings.min_sos
    c_max_mm_us = app_settings.max_sos

    c_mm_us = v * (c_max_mm_us - c_min_mm_us) + c_min_mm_us
    c_m_s = c_mm_us * 1e4
    return c_m_s

def initial_loss(pred_tof, src_loc):
    loss = 0.0
    c = 0
    for s_idx, sl in enumerate(src_loc):
        p_tof = _to_sec(pred_tof[s_idx, sl[0], sl[1]])
        loss += torch.pow(p_tof, 2)
        c += 1
    return loss / c

def boundary_loss(sos_pred, known_tof, num_samples=7):
    total_loss = 0.0
    sos_pred.requires_grad_(True)
    eps = 1e-8

    center_t = torch.linspace(0.4, 0.6, num_samples - 2, device=sos_pred.device)
    t = torch.cat([torch.tensor([0.0], device=sos_pred.device), center_t, torch.tensor([0.99], device=sos_pred.device)])

    for x_s, y_s, x_r, y_r, tof in known_tof:
        x_path = x_s + t * (x_r - x_s)
        y_path = y_s + t * (y_r - y_s)

        sos_path = bilinear_interpolate(sos_pred, x_path, y_path)
        sos_path = _to_mps(sos_path)

        dx = x_path[1:] - x_path[:-1]
        dy = y_path[1:] - y_path[:-1]
        ds = torch.sqrt(dx ** 2 + dy ** 2)

        tof_pred = torch.sum(ds / (sos_path[:-1] + eps))
        total_loss += (tof - tof_pred) ** 2

    return total_loss / len(known_tof)

def eikonal_loss(pred_tof, sos_map):
    pixel_size_m = app_settings.pixel_to_mm
    eps = 1e-8

    grad_T = torch.autograd.grad(
        outputs=pred_tof.sum(),
        inputs=pred_tof,
        create_graph=True
    )[0]

    grad_x = grad_T[:, :, 1:-1, 2:] - grad_T[:, :, 1:-1, :-2]
    grad_y = grad_T[:, :, 2:, 1:-1] - grad_T[:, :, :-2, 1:-1]

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    sos_clipped = sos_map.clamp(min=eps)

    residual = grad_mag - 1.0 / sos_clipped
    loss = torch.mean(residual**2)
    return loss

def eikonal_loss_multi(sos_pred, solver, source, roi_start=20, roi_end=100, eps=1e-8):
    # Convert and clamp sos_pred
    sos_pred = sos_pred + torch.randn_like(sos_pred) * 0.01  # Add slight variations for testing
    sos_pred = _to_mps(sos_pred).clamp(min=eps, max=1e3)  # Clamp to avoid extreme values
    sos_pred.requires_grad_(True)  # Ensure sos_pred requires gradients

    H, W = app_settings.anatomy_height, app_settings.anatomy_width

    # Ensure source coordinates are a tensor on the correct device
    source = torch.tensor([source[0], source[1]], dtype=torch.long, device=sos_pred.device)

    # Initialize the travel time field
    T_init = torch.full((1, 1, H, W), float('inf'), device=sos_pred.device)
    T_init[0, 0, source[0].item(), source[1].item()] = 0  # Set source location to 0
    T_init.requires_grad_(True)  # Ensure T_init requires gradients

    # Solve the Eikonal equation
    T = solver(T_init, sos_pred)
    T = T.clamp(min=0.0, max=1e3)  # Clamp to avoid invalid values
    T = _to_sec(T)
    T.requires_grad_(True)  # Enforce T requires gradients

    # Compute gradients using automatic differentiation
    grad_T = torch.autograd.grad(
        outputs=T.sum(),  # Scalar output
        inputs=T,         # Variable to differentiate
        create_graph=True  # Required for higher-order derivatives
    )[0]

    # Compute gradients in x and y directions
    grad_x = grad_T[:, :, :, 1:] - grad_T[:, :, :, :-1]
    grad_y = grad_T[:, :, 1:, :] - grad_T[:, :, :-1, :]

    # Compute magnitude of the gradient
    grad_mag = torch.sqrt(
        grad_x[:, :, roi_start:roi_end, roi_start:roi_end] ** 2 +
        grad_y[:, :, roi_start:roi_end, roi_start:roi_end] ** 2 + eps
    )

    # Crop speed of sound to the ROI and ensure no division by zero
    sos_clipped = sos_pred[:, :, roi_start:roi_end, roi_start:roi_end].clamp(min=eps)

    # Compute the Eikonal loss
    residual = grad_mag - 1.0 / sos_clipped
    loss = torch.mean(residual ** 2)

    return loss

def bilinear_interpolate(grid, x, y):
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, grid.size(2) - 1)
    x1 = torch.clamp(x1, 0, grid.size(2) - 1)
    y0 = torch.clamp(y0, 0, grid.size(3) - 1)
    y1 = torch.clamp(y1, 0, grid.size(3) - 1)

    Ia = grid[:, :, x0, y0]
    Ib = grid[:, :, x0, y1]
    Ic = grid[:, :, x1, y0]
    Id = grid[:, :, x1, y1]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    return wa * Ia + wb * Ib + wc * Ic + wd * Id
