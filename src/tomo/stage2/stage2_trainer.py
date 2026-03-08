"""Stage 2 trainer: PPO loop over per-source episodes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tomo.stage2.stage2_env import Stage2Env, Stage2Sample
from tomo.stage2.stage2_policy import Stage2Policy


class RolloutBuffer:
    """Buffer for PPO rollout."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations: list = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[torch.Tensor] = []


class Stage2Trainer:
    """PPO trainer for Stage 2 local refinement policy."""

    def __init__(
        self,
        policy: Stage2Policy,
        env: Stage2Env,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        rollout_length: int = 10,
        device: torch.device | None = None,
    ):
        self.policy = policy.to(device or torch.device("cpu"))
        self.env = env
        self.device = device or torch.device("cpu")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_length = rollout_length
        self.buffer = RolloutBuffer()

    def select_action(self, obs) -> tuple[torch.Tensor | None, bool]:
        """Run policy, sample action. Returns (action, skip) where skip=True if no ROI."""
        if obs.n_roi == 0:
            return None, True

        with torch.no_grad():
            action_dist, value = self.policy(obs)
            action = action_dist.sample()
            logprob = action_dist.log_prob(action).sum(dim=-1)

        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.values.append(value)

        return action.cpu().numpy(), False

    def store_reward(self, reward: float, done: bool):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def _compute_returns_and_advantages(self, next_value: torch.Tensor):
        returns = []
        advantages = []
        gae = 0.0
        next_val = next_value

        for step in reversed(range(len(self.buffer.rewards))):
            mask = 1.0 - float(self.buffer.dones[step])
            delta = (
                self.buffer.rewards[step]
                + self.gamma * next_val * mask
                - self.buffer.values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_val = self.buffer.values[step]
            returns.insert(0, gae + self.buffer.values[step])

        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if adv.numel() > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return ret, adv

    def update_policy(self, next_obs):
        """PPO update from buffer."""
        if len(self.buffer.observations) == 0:
            return

        with torch.no_grad():
            if next_obs is not None and next_obs.n_roi > 0:
                _, next_value = self.policy(next_obs)
            else:
                next_value = torch.tensor(0.0, device=self.device)

        returns, advantages = self._compute_returns_and_advantages(next_value)
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        old_values = torch.stack(self.buffer.values).to(self.device)

        for epoch in range(self.update_epochs):
            for i, obs in enumerate(self.buffer.observations):
                if obs.n_roi == 0:
                    continue
                action_dist, value = self.policy(obs)
                logprob = action_dist.log_prob(old_actions[i]).sum(dim=-1)
                entropy = action_dist.entropy().sum(dim=-1)

                ratio = torch.exp(logprob - old_logprobs[i].detach())
                adv = advantages[i]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value.squeeze(), returns[i])
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.clear()

    def train(
        self,
        samples: list[Stage2Sample],
        *,
        max_episodes: int | None = None,
        sources_per_sample: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Train over samples and sources.

        Args:
            samples: List of Stage2Sample from Stage 1 outputs.
            max_episodes: Max total episodes (None = all samples * all sources).
            sources_per_sample: List of source counts per sample (default: all sources).

        Returns:
            List of episode info dicts.
        """
        S = len(samples[0].x_s) if samples else 0
        if sources_per_sample is None:
            sources_per_sample = [S] * len(samples)

        episode_infos: list[dict[str, Any]] = []
        ep_count = 0
        max_ep = max_episodes or (len(samples) * S)

        for sample_idx, sample in enumerate(samples):
            n_sources = min(sources_per_sample[sample_idx], len(sample.x_s))
            for src_idx in range(n_sources):
                if ep_count >= max_ep:
                    break

                self.env.sample = sample
                obs = self.env.reset(active_source=src_idx)
                obs = obs.to(self.device)

                episode_reward = 0.0
                step_count = 0
                info: dict[str, Any] = {"done": False, "reason": "skip"}

                while step_count < self.rollout_length:
                    action, skip = self.select_action(obs)
                    if skip:
                        break

                    next_obs, reward, done, info = self.env.step(action)
                    self.store_reward(reward, done)
                    episode_reward += reward
                    step_count += 1

                    if next_obs is not None:
                        next_obs = next_obs.to(self.device)
                    obs = next_obs

                    if done:
                        break

                if len(self.buffer.observations) > 0:
                    self.update_policy(obs)

                episode_infos.append({
                    "sample_idx": sample_idx,
                    "source_idx": src_idx,
                    "reward": episode_reward,
                    "steps": step_count,
                    **info,
                })
                ep_count += 1

            if ep_count >= max_ep:
                break

        return episode_infos
