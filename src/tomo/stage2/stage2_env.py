"""Stage 2 environment: per-source episode, ROI-only corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from tomo.operators.matlab_fmm_wrapper import forward_tof
from tomo.stage2.grid_mapper import map_roi_corrections_to_grid
from tomo.stage2.roi_builder import build_roi_and_support
from tomo.stage2.stage2_graph import Stage2GraphInput, build_stage2_graph
from tomo.stage2.stage2_objective import stage2_reward


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class Stage2Sample:
    """Per-sample inputs for Stage 2."""

    c_stage1: np.ndarray  # (128, 128)
    raw_tof: np.ndarray  # (S, R)
    x_s: np.ndarray  # (S, 2)
    x_r: np.ndarray  # (R, 2)
    tof_pred_stage1: np.ndarray  # (S, R)
    c_base: float = 1.5
    c_min: float = 1.45
    c_max: float = 1.8
    scale_factor: float = 1.0


class Stage2Env:
    """Environment for Stage 2: one source per episode, ROI-only corrections.

    State: c_current (starts as c_stage1), graph for active source.
    Action: delta_update (N_roi,) from policy.
    """

    def __init__(
        self,
        sample: Stage2Sample,
        *,
        device: torch.device | None = None,
        tau_delta: float = 0.02,
        tau_residual: float = 0.01,
        dilation_iterations: int = 2,
        support_width: int = 1,
        roi_k: int = 20,
        rcv_k: int = 15,
        spatial_k: int = 9,
        sigma: float = 0.5,
        max_delta: float = 0.02,
        eta: float = 0.5,
        loss_type: str = "huber",
        delta: float = 0.01,
        lambda_smooth: float = 1.0,
        lambda_bounds: float = 10.0,
        lambda_step: float = 0.1,
        max_steps: int = 5,
        improvement_threshold: float = 1e-5,
    ):
        self.sample = sample
        self.device = device or torch.device("cpu")

        self.tau_delta = tau_delta
        self.tau_residual = tau_residual
        self.dilation_iterations = dilation_iterations
        self.support_width = support_width
        self.roi_k = roi_k
        self.rcv_k = rcv_k
        self.spatial_k = spatial_k
        self.sigma = sigma
        self.max_delta = max_delta
        self.eta = eta
        self.loss_type = loss_type
        self.delta = delta
        self.lambda_smooth = lambda_smooth
        self.lambda_bounds = lambda_bounds
        self.lambda_step = lambda_step
        self.max_steps = max_steps
        self.improvement_threshold = improvement_threshold

        self._c_current: np.ndarray | None = None
        self._roi_masks = None
        self._graph_input: Stage2GraphInput | None = None
        self._observation: Data | None = None
        self._active_source: int = 0
        self._step_count: int = 0
        self._last_loss: float = float("inf")

    def reset(self, active_source: int = 0) -> Data:
        """Reset for new episode: one source, build ROI and graph."""
        self._active_source = active_source
        self._step_count = 0
        self._c_current = self.sample.c_stage1.copy()
        self._last_loss = float("inf")

        s = active_source
        delta_stage1 = self.sample.c_stage1 - self.sample.c_base
        tof_residual = self.sample.tof_pred_stage1 - self.sample.raw_tof

        x_s_single = self.sample.x_s[s : s + 1]
        x_r = self.sample.x_r

        self._roi_masks = build_roi_and_support(
            delta_stage1,
            tof_residual,
            self.sample.x_s,
            self.sample.x_r,
            tau_delta=self.tau_delta,
            tau_residual=self.tau_residual,
            dilation_iterations=self.dilation_iterations,
            support_width=self.support_width,
        )

        roi_coords = self._roi_masks.roi_coords
        if roi_coords.shape[0] == 0:
            return self._build_empty_observation(active_source)

        tof_obs = self.sample.raw_tof[s, :]
        tof_pred = self.sample.tof_pred_stage1[s, :]
        tof_error = tof_pred - tof_obs

        self._graph_input = Stage2GraphInput(
            c_stage1=self.sample.c_stage1,
            delta_stage1=delta_stage1,
            backproj_map=self._roi_masks.backproj_map,
            x_s=x_s_single,
            x_r=x_r,
            tof_obs=tof_obs,
            tof_pred=tof_pred,
            tof_error=tof_error,
            roi_masks=self._roi_masks,
        )

        self._observation = build_stage2_graph(
            self._graph_input,
            device=self.device,
            roi_k=self.roi_k,
            rcv_k=self.rcv_k,
            spatial_k=self.spatial_k,
        )
        return self._observation

    def _build_empty_observation(self, active_source: int) -> Data:
        """Return minimal valid Data when ROI is empty (no refinement possible)."""
        n_roi = 0
        n_support = self._roi_masks.support_coords.shape[0] if self._roi_masks else 0
        x_s = self.sample.x_s[active_source : active_source + 1]
        x_r = self.sample.x_r
        n_src = 1
        n_rcv = x_r.shape[0]

        pos_list = [x_s, x_r]
        if n_support > 0:
            support_coords = self._roi_masks.support_coords
            support_pos = np.column_stack([support_coords[:, 1], support_coords[:, 0]])
            pos_list.append(support_pos)
        pos = np.vstack(pos_list).astype(np.float32)
        n_total = pos.shape[0]

        x = torch.zeros((n_total, 10), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(pos, dtype=torch.float32),
            num_nodes=n_total,
            n_src=n_src,
            n_rcv=n_rcv,
            n_roi=0,
            n_support=n_support,
            roi_node_start=n_src + n_rcv,
            roi_node_end=n_src + n_rcv,
        )
        if self.device is not None:
            data = data.to(self.device)
        return data

    def step(self, action: np.ndarray | torch.Tensor) -> tuple[Data | None, float, bool, dict[str, Any]]:
        """Step environment with policy action (N_roi,) delta_update."""
        roi_coords = self._roi_masks.roi_coords
        if roi_coords.shape[0] == 0:
            return self._observation, 0.0, True, {"done": True, "reason": "empty_roi"}

        action = _to_numpy(action)
        if action.size != roi_coords.shape[0]:
            return self._observation, -1e6, True, {"done": True, "reason": "action_size_mismatch"}

        delta_c = map_roi_corrections_to_grid(
            roi_coords,
            action,
            self.sample.c_stage1.shape,
            sigma=self.sigma,
            max_delta=self.max_delta,
            eta=self.eta,
        )

        c_new = np.clip(
            self._c_current + delta_c,
            self.sample.c_min,
            self.sample.c_max,
        )

        s = self._active_source
        x_s_single = self.sample.x_s[s : s + 1]
        x_r = self.sample.x_r
        tof_pred_new = forward_tof(
            c_new,
            x_s_single,
            x_r,
            scale_factor=self.sample.scale_factor,
        )
        tof_pred_new = tof_pred_new[0, :]
        raw_tof_obs = self.sample.raw_tof[s, :]

        reward = stage2_reward(
            tof_pred_new,
            raw_tof_obs,
            c_new,
            delta_c,
            c_min=self.sample.c_min,
            c_max=self.sample.c_max,
            loss_type=self.loss_type,
            delta=self.delta,
            lambda_smooth=self.lambda_smooth,
            lambda_bounds=self.lambda_bounds,
            lambda_step=self.lambda_step,
        )

        self._c_current = c_new
        self._step_count += 1

        residuals = tof_pred_new - raw_tof_obs
        if self.loss_type == "huber":
            from tomo.stage2.stage2_objective import _huber_loss
            current_loss = _huber_loss(residuals, self.delta)
        else:
            from tomo.stage2.stage2_objective import _charbonnier_loss
            current_loss = _charbonnier_loss(residuals, self.delta)

        done = False
        reason = ""
        if self._step_count >= self.max_steps:
            done = True
            reason = "max_steps"
        elif current_loss > self._last_loss:
            done = True
            reason = "loss_increased"
        elif self._last_loss - current_loss < self.improvement_threshold:
            done = True
            reason = "no_improvement"

        self._last_loss = current_loss

        info: dict[str, Any] = {
            "step": self._step_count,
            "reward": reward,
            "done": done,
            "reason": reason,
            "current_loss": current_loss,
            "source": s,
        }

        return self._observation, reward, done, info
