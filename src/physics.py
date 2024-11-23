import torch

import torch
import torch.nn as nn
from torch.autograd import grad


class PINNLoss(nn.Module):
    def __init__(self):
        super(PINNLoss, self).__init__()

    def forward(self, c_pred, T_interp):
        """
        c_pred: Predicted speed of sound (128x128)
        T_interp: Interpolated travel time (128x128)
        """
        # Compute gradients of T_interp (travel time)
        grad_T_x = grad(T_interp, T_interp, grad_outputs=torch.ones_like(T_interp), create_graph=True)[0]
        grad_T_y = grad(T_interp, T_interp, grad_outputs=torch.ones_like(T_interp), create_graph=True)[1]

        # Compute gradient magnitude |∇T|
        grad_mag = torch.sqrt(grad_T_x ** 2 + grad_T_y ** 2 + 1e-8)

        # Compute the residual of the eikonal equation
        residual = grad_mag * c_pred - 1

        # Physics-informed loss
        physics_loss = torch.mean(residual ** 2)
        return physics_loss


# Define the total loss
class TotalLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_physics=1.0):
        super(TotalLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss()
        self.pinn_loss = PINNLoss()

    def forward(self, c_pred, c_true, T_interp):
        data_loss = self.mse_loss(c_pred, c_true)  # Supervised MSE loss
        physics_loss = self.pinn_loss(c_pred, T_interp)  # Physics-informed loss
        total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss
        return total_loss





def compute_eikonal_loss(coords, pinn_output, sos_values):
    """
    Computes the physics-informed loss using the Eikonal equation.

    Args:
        coords (torch.Tensor): Coordinates where the Eikonal equation is evaluated.
        pinn_output (torch.Tensor): Model output at the given coordinates.
        sos_values (torch.Tensor): Speed of sound values at the coordinates.

    Returns:
        torch.Tensor: Physics-informed loss based on the Eikonal equation.
    """
    coords.requires_grad = True

    gradients = torch.autograd.grad(
        outputs=pinn_output,
        inputs=coords,
        grad_outputs=torch.ones_like(pinn_output),
        create_graph=True
    )[0]

    gradient_norm = torch.sqrt(torch.sum(gradients**2, dim=1))
    sos_values = sos_values.view(-1)  # Flatten sos_values to 1D

    if gradient_norm.shape != sos_values.shape:
        raise ValueError(f"Shape mismatch: gradient_norm {gradient_norm.shape} vs. sos_values {sos_values.shape}")

    loss = torch.mean((gradient_norm - 1.0 / sos_values)**2)
    return loss
