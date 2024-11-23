import torch

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
