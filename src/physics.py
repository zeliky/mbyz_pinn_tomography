import torch

import torch
import torch.nn as nn
from torch.autograd import grad


class PINNLoss(nn.Module):
    def __init__(self, physics_loss_weight, L_x, L_y):
        super(PINNLoss, self).__init__()
        self.physics_loss_weight = physics_loss_weight
        self.mse_loss = nn.MSELoss()
        self.L_x = L_x
        self.L_y = L_y

    def forward(self, model, tof_input, x_r, x_s, observed_tof, x_coords, x_s_grid):
        # Forward pass through the model
        c_map, T_pred = model(tof_input, x_r, x_s)

        # Data loss (MSE between predicted travel times and observed ToF)
        data_loss = self.mse_loss(T_pred.squeeze(), observed_tof.squeeze())

        # Physics loss (enforcing the eikonal equation)
        physics_loss = self.compute_physics_loss(model.fcnn, x_coords, x_s_grid, c_map)

        # Total loss
        total_loss = data_loss + self.physics_loss_weight * physics_loss

        return total_loss, data_loss.item(), physics_loss.item()

    def compute_physics_loss(self, fcnn, x_coords, x_s, c_map):
        batch_size, num_points, _ = x_coords.shape
        x_coords_flat = x_coords.view(batch_size * num_points, 2).requires_grad_(True)
        x_s_flat = x_s.view(batch_size * num_points, 2)

        # Predict travel times at grid points
        T_pred = fcnn(x_coords_flat, x_s_flat)

        # Compute gradient of T with respect to x_coords
        grad_T = torch.autograd.grad(
            outputs=T_pred,
            inputs=x_coords_flat,
            grad_outputs=torch.ones_like(T_pred),
            create_graph=True
        )[0]

        # Interpolate c(x) at x_coords
        c_at_coords = interpolate_c(c_map, x_coords_flat, self.L_x, self.L_y)
        grad_T_norm = torch.norm(grad_T, dim=1)
        physics_loss = torch.mean((grad_T_norm - 1.0 / c_at_coords) ** 2)
        return physics_loss


def interpolate_c(c_map, spatial_coords, L_x, L_y):
    """
    Interpolates the speed of sound map at the given spatial coordinates.
    Args:
        c_map: Tensor of shape [batch_size, 1, H, W]
        spatial_coords: Tensor of shape [batch_size * num_points, 2], in meters
        L_x, L_y: Physical dimensions of the tissue in meters
    Returns:
        c_at_coords: Tensor of shape [batch_size * num_points]
    """
    batch_size = c_map.shape[0]
    H, W = c_map.shape[2], c_map.shape[3]
    num_points_total = spatial_coords.shape[0]

    # Normalize spatial coordinates to [-1, 1] for grid_sample
    x_norm = (spatial_coords[:, 0] / L_x) * 2.0 - 1.0
    y_norm = (spatial_coords[:, 1] / L_y) * 2.0 - 1.0
    grid = torch.stack((x_norm, y_norm), dim=1)  # Shape: [batch_size * num_points, 2]

    # Reshape grid for grid_sample
    grid = grid.view(batch_size, -1, 1, 2)  # Shape: [batch_size, num_points, 1, 2]

    # Interpolate c_map at grid points
    c_at_coords = F.grid_sample(c_map, grid, mode='bilinear',
                                align_corners=False)  # Shape: [batch_size, 1, num_points, 1]
    c_at_coords = c_at_coords.squeeze(1).squeeze(2)  # Shape: [batch_size, num_points]
    c_at_coords = c_at_coords.view(-1)  # Flatten to [batch_size * num_points]
    return c_at_coords
