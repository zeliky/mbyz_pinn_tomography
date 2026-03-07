"""Trainer: single entrypoint orchestration. Uses TomographySystem + objective."""

from typing import Any

import torch

from tomo.losses.objective import objective
from tomo.systems.tomography_system import TomographySystem
from tomo.state.observation import Observation


class Trainer:
    """Minimal trainer: one system, one objective, no competing pipelines."""

    def __init__(
        self,
        system: TomographySystem,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        weight_tof: float = 1.0,
        weight_eikonal: float = 0.0,
        weight_l2: float = 0.0,
        weight_smooth: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.system = system
        self.optimizer = optimizer
        self.weight_tof = weight_tof
        self.weight_eikonal = weight_eikonal
        self.weight_l2 = weight_l2
        self.weight_smooth = weight_smooth
        self.device = device or torch.device("cpu")

    def train_step(
        self,
        batch: dict[str, Any],
    ) -> dict[str, float]:
        """One training step: rollout -> objective -> backward."""
        observation = batch.get("observation")
        observation_graph = batch.get("observation_graph") or batch.get("graph")
        if observation is None or observation_graph is None:
            raise ValueError("batch must contain 'observation' and 'observation_graph'")

        if self.optimizer:
            self.optimizer.zero_grad()

        state, preds = self.system.rollout(observation, observation_graph)
        tof_pred = preds[-1] if preds else self.system.operator(state, observation_graph)

        rollout_output = {
            "tof_pred": tof_pred,
            "state": state,
            "observation_graph": observation_graph,
        }
        loss_dict = objective(
            batch,
            rollout_output,
            weight_tof=self.weight_tof,
            weight_eikonal=self.weight_eikonal,
            weight_l2=self.weight_l2,
            weight_smooth=self.weight_smooth,
        )

        loss_total = loss_dict["loss_total"]
        if self.optimizer and loss_total.requires_grad:
            loss_total.backward()
            self.optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
