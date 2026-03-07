"""TomographySystem: one object that owns the execution flow."""

from typing import Any

import torch
from torch_geometric.data import Data

from tomo.operators.base import Operator
from tomo.policies.base import Policy
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class TomographySystem:
    """Composes optional initializer, operator, and policy.

    Main loop:
      ToF_obs -> Initializer (optional) -> c0
      -> repeat T steps: tof_pred = Operator(c), error = tof_pred - tof_obs,
         delta_c = Policy(...), c = c + alpha * delta_c
      -> final c
    """

    def __init__(
        self,
        operator: Operator,
        policy: Policy,
        initializer: Any | None = None,
        rollout_steps: int = 1,
        alpha: float = 1.0,
    ) -> None:
        self.operator = operator
        self.policy = policy
        self.initializer = initializer
        self.rollout_steps = rollout_steps
        self.alpha = alpha

    def run_one_step(
        self,
        state: SoSState,
        observation: Observation,
        observation_graph: Data,
    ) -> tuple[SoSState, torch.Tensor]:
        """One rollout step: operator -> error -> policy -> state update."""
        tof_pred = self.operator(state, observation_graph)
        # Placeholder: tof_observed should come from observation; match shape if needed
        tof_observed = observation.tof_observed
        if tof_observed.dim() == 2:
            # Flatten or index receivers as needed; for stub use same device
            num_nodes = tof_pred.shape[0]
            tof_observed_flat = tof_observed.flatten().to(tof_pred.device)
            if tof_observed_flat.numel() != num_nodes:
                tof_error = tof_pred - torch.zeros_like(tof_pred, device=tof_pred.device)
            else:
                tof_error = tof_pred - tof_observed_flat[:num_nodes]
        else:
            tof_error = tof_pred - tof_observed.to(tof_pred.device)
        delta = self.policy(state, observation, tof_error, observation_graph)
        new_state = state.apply(self.alpha * delta)
        return new_state, tof_pred

    def rollout(
        self,
        observation: Observation,
        observation_graph: Data,
        initial_state: SoSState | None = None,
    ) -> tuple[SoSState, list[torch.Tensor]]:
        """Multi-step rollout. If initial_state is None and initializer set, use initializer."""
        if initial_state is not None:
            state = initial_state
        elif self.initializer is not None:
            state = self.initializer(observation)
        else:
            num_nodes = observation_graph.num_nodes
            device = observation.tof_observed.device
            state = SoSState(
                c_values=torch.ones(num_nodes, device=device) * 1.5,
                step_idx=0,
            )
        preds: list[torch.Tensor] = []
        for _ in range(self.rollout_steps):
            state, tof_pred = self.run_one_step(state, observation, observation_graph)
            preds.append(tof_pred)
        return state, preds
