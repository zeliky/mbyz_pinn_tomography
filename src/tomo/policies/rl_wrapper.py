"""RL wrapper: PPO training mode using TomographySystem. Env talks to system, not duplicate solver."""

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch_geometric.data import Data

from tomo.policies.gnn_policy import GNNPolicy
from tomo.systems.tomography_system import TomographySystem
from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class RolloutBuffer:
    """Lightweight buffer for PPO rollouts."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.observations: list[Data] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[torch.Tensor] = []


class RLPolicyAdapter(nn.Module):
    """Wraps GNNPolicy for PPO: returns (action_dist, value). Action = delta_c sample."""

    def __init__(self, gnn_policy: GNNPolicy, action_std_init: float = 0.1) -> None:
        super().__init__()
        self.gnn_policy = gnn_policy
        self.num_sensor_nodes = gnn_policy.num_sensor_nodes
        self.action_log_std = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(action_std_init)))
        self.value_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, data: Data) -> tuple[Normal, torch.Tensor]:
        """Return (action_dist over delta_c at mesh nodes, value scalar)."""
        num_nodes = data.num_nodes
        device = data.x.device if data.x is not None else data.pos.device
        if data.x is None or data.x.shape[1] < 2:
            x = torch.zeros(num_nodes, 6, device=device)
        else:
            x = data.x if data.x.shape[1] >= 6 else torch.cat(
                [data.x, torch.zeros(num_nodes, 6 - data.x.shape[1], device=data.x.device)], dim=-1
            )
        state = SoSState(c_values=torch.ones(num_nodes, device=device) * 1.5)
        obs = Observation(tof_observed=torch.zeros(num_nodes, device=device))
        tof_error = torch.zeros(num_nodes, device=device)
        data2 = Data(x=x, edge_index=data.edge_index, pos=data.pos)
        delta = self.gnn_policy(state, obs, tof_error, data2)
        mesh_start = self.num_sensor_nodes
        delta_mesh = delta[mesh_start:]
        action_std = torch.exp(self.action_log_std).expand_as(delta_mesh)
        action_dist = Normal(delta_mesh, action_std)
        value = self.value_scale * delta_mesh.mean()
        return action_dist, value


class TomoEnv:
    """Env that uses TomographySystem. reset() -> graph; step(action_delta) -> next_graph, reward, done, info."""

    def __init__(
        self,
        system: TomographySystem,
        observation: Observation,
        observation_graph: Data,
        device: torch.device,
        reward_from_objective: bool = True,
    ) -> None:
        self.system = system
        self.observation = observation
        self.observation_graph = observation_graph
        self.device = device
        self.reward_from_objective = reward_from_objective
        self._state: SoSState | None = None

    def reset(self) -> Data:
        self._state, _ = self.system.rollout(self.observation, self.observation_graph, initial_state=None)
        return self._obs_from_state()

    def _obs_from_state(self) -> Data:
        g = self.observation_graph
        x = g.x
        if x is None:
            x = torch.zeros(g.num_nodes, 6, device=self.device)
            if self._state is not None:
                x[:, 0] = self._state.c_values
        return Data(x=x, edge_index=g.edge_index, pos=g.pos, edge_attr=getattr(g, "edge_attr", None))

    def step(self, action_delta: torch.Tensor) -> tuple[Data, float, bool, dict[str, Any]]:
        if self._state is None:
            self.reset()
        delta_full = torch.zeros_like(self._state.c_values, device=self.device)
        mesh_start = getattr(self.system.policy, "num_sensor_nodes", 0)
        delta_full[mesh_start:] = action_delta.to(self.device)
        self._state = self._state.apply(1.0 * delta_full)
        tof_pred = self.system.operator(self._state, self.observation_graph)
        tof_obs = self.observation.tof_observed
        if tof_obs.numel() == tof_pred.numel():
            reward = -torch.nn.functional.mse_loss(tof_pred, tof_obs.to(self.device)).item()
        else:
            reward = -float(tof_pred.var().item())
        return self._obs_from_state(), reward, False, {}
