import torch

import torch
import torch.nn as nn
from torch.autograd import grad


import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    def __init__(self, physics_loss_weight, L_x, L_y):
        super(PINNLoss, self).__init__()
        self.physics_loss_weight = physics_loss_weight
        self.mse_loss = nn.MSELoss()
        self.L_x = L_x
        self.L_y = L_y

    def forward(self, model, tof_input, x_r, x_s, observed_tof, x_coords):
        """
        Computes the total loss combining data loss and physics loss.
        Args:
            model: The PINNModel instance.
            tof_input: Tensor of shape [batch_size, 1, H_in, W_in], input ToF images.
            x_r: Tensor of shape [batch_size, 2], receiver positions.
            x_s: Tensor of shape [batch_size, 2], source positions.
            observed_tof: Tensor of shape [batch_size], observed ToF measurements.
            x_coords: Tensor of shape [batch_size, num_points, 2], grid points for physics loss.

        Returns:
            total_loss: Scalar tensor representing the total loss.
            data_loss_value: Scalar float, value of the data loss.
            physics_loss_value: Scalar float, value of the physics loss.
        """
        # Forward pass through the model
        c_map, T_pred, T_grid = model(tof_input, x_r, x_s, x_coords)

        # Data loss (MSE between predicted travel times and observed ToF)
        data_loss = self.mse_loss(T_pred.squeeze(), observed_tof.squeeze())

        # Physics loss (enforcing the eikonal equation)
        physics_loss = self.compute_physics_loss(T_grid, x_coords, c_map)

        # Total loss
        total_loss = data_loss + self.physics_loss_weight * physics_loss

        return total_loss, data_loss.item(), physics_loss.item()

    def compute_physics_loss(self, T_grid, x_coords, c_map):

        # Flatten tensors for gradient computation
        T_grid_flat = T_grid.view(-1)  # Shape: [batch_size * num_points]
        x_coords_flat = x_coords.view(-1, 2)  # Shape: [batch_size * num_points, 2]


        # Compute gradient of T with respect to x_coords
        grad_T = torch.autograd.grad(
            outputs=T_grid_flat,
            inputs=x_coords_flat,
            grad_outputs=torch.ones_like(T_grid_flat),
            create_graph=True
        )[0]  # Shape: [batch_size * num_points, 2]

        # Interpolate c(x) at x_coords
        c_at_coords = self.interpolate_c(c_map, x_coords)

        # Flatten c_at_coords
        c_at_coords_flat = c_at_coords.view(-1)  # Shape: [batch_size * num_points]

        # Compute norm of grad_T
        grad_T_norm = torch.norm(grad_T, dim=1)  # Shape: [batch_size * num_points]

        # Compute physics loss
        physics_loss = torch.mean((grad_T_norm - 1.0 / c_at_coords_flat) ** 2)
        return physics_loss

    def interpolate_c(self, c_map, x_coords):
        batch_size, num_points, _ = x_coords.shape
        H, W = c_map.shape[2], c_map.shape[3]

        # Normalize spatial coordinates to [-1, 1] for grid_sample
        x_norm = (x_coords[:, :, 0] / self.L_x) * 2.0 - 1.0
        y_norm = (x_coords[:, :, 1] / self.L_y) * 2.0 - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1)  # Shape: [batch_size, num_points, 2]

        # Reshape grid for grid_sample
        grid = grid.view(batch_size, num_points, 1, 2)  # Shape: [batch_size, num_points, 1, 2]

        # Interpolate c_map at grid points
        c_at_coords = F.grid_sample(c_map, grid, mode='bilinear', align_corners=False)  # Shape: [batch_size, 1, num_points, 1]
        c_at_coords = c_at_coords.squeeze(1).squeeze(2)  # Shape: [batch_size, num_points]

        return c_at_coords

