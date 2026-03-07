"""ToF reconstruction loss."""

import torch


def tof_reconstruction_loss(
    tof_pred: torch.Tensor,
    tof_observed: torch.Tensor,
    receiver_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between predicted and observed ToF, optionally at receiver nodes only.

    Args:
        tof_pred: (N,) or (B, N) predicted ToF per node.
        tof_observed: same shape or (num_receivers,) / (B, num_receivers); will be broadcast/indexed.
        receiver_mask: optional boolean (N,) or indices; if None, use all nodes.

    Returns:
        Scalar loss.
    """
    if receiver_mask is not None:
        tof_pred = tof_pred[receiver_mask]
        if tof_observed.numel() == tof_pred.numel():
            tof_observed = tof_observed.flatten()
        else:
            tof_observed = tof_observed.expand_as(tof_pred)
    else:
        if tof_observed.numel() != tof_pred.numel():
            tof_observed = tof_observed.flatten().expand(tof_pred.shape[0])
    return torch.nn.functional.mse_loss(tof_pred.float(), tof_observed.float().to(tof_pred.device))
