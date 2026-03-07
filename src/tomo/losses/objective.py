"""Composed objective: loss_dict = objective(batch, rollout_output)."""

from typing import Any

import torch
from torch_geometric.data import Data

from tomo.losses.physics import eikonal_residual_loss
from tomo.losses.regularization import l2_regularization, smoothness_regularization
from tomo.losses.tof import tof_reconstruction_loss


def objective(
    batch: dict[str, Any],
    rollout_output: dict[str, Any],
    *,
    weight_tof: float = 1.0,
    weight_eikonal: float = 0.0,
    weight_l2: float = 0.0,
    weight_smooth: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute loss dict from batch and rollout. No loss logic inside models.

    Args:
        batch: at least 'tof_observed', optional 'receiver_mask', 'graph'.
        rollout_output: 'tof_pred', 'state', 'observation_graph'.

    Returns:
        loss_dict with keys 'loss_total', 'loss_tof', and optional others.
    """
    tof_pred = rollout_output["tof_pred"]
    state = rollout_output["state"]
    observation_graph: Data = rollout_output["observation_graph"]
    tof_observed = batch.get("tof_observed")
    if tof_observed is None:
        tof_observed = batch.get("tof_obs")
    receiver_mask = batch.get("receiver_mask")

    if tof_observed is None:
        tof_observed = torch.zeros_like(tof_pred, device=tof_pred.device)

    loss_tof = tof_reconstruction_loss(tof_pred, tof_observed, receiver_mask)

    loss_dict: dict[str, torch.Tensor] = {"loss_tof": loss_tof}

    if weight_eikonal > 0 and observation_graph.edge_index is not None:
        loss_eik = eikonal_residual_loss(
            tof_pred,
            state.c_values,
            observation_graph.edge_index,
            observation_graph.pos,
            getattr(observation_graph, "edge_attr", None),
        )
        loss_dict["loss_eikonal"] = loss_eik
    else:
        loss_dict["loss_eikonal"] = torch.tensor(0.0, device=tof_pred.device)

    if weight_l2 > 0:
        loss_dict["loss_l2"] = l2_regularization(state.c_values)
    else:
        loss_dict["loss_l2"] = torch.tensor(0.0, device=tof_pred.device)

    if weight_smooth > 0 and observation_graph.edge_index is not None:
        loss_dict["loss_smooth"] = smoothness_regularization(
            state.c_values, observation_graph.edge_index
        )
    else:
        loss_dict["loss_smooth"] = torch.tensor(0.0, device=tof_pred.device)

    loss_total = (
        weight_tof * loss_dict["loss_tof"]
        + weight_eikonal * loss_dict["loss_eikonal"]
        + weight_l2 * loss_dict["loss_l2"]
        + weight_smooth * loss_dict["loss_smooth"]
    )
    loss_dict["loss_total"] = loss_total
    return loss_dict
