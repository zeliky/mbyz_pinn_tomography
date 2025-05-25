import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple


class RolloutBuffer:
    """A lightweight buffer for PPO rollouts."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations: List = []  # PyG Data objects
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []


class RLAgent:
    """
    PPO-style agent that works with a GNN-based policy.

    The policy network must return a tuple (action_dist, value) when called with an observation.
      * action_dist: torch.distributions distribution (e.g. Normal)
      * value: state-value estimate (Tensor, shape [])
    """

    def __init__(
        self,
        policy: nn.Module,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.env = env
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Interaction ------------------------------------------------------
    # ------------------------------------------------------------------

    def select_action(self, observation):
        """Run policy → sample action → store transition in buffer."""
        # policy must internally handle PyG Data object
        with torch.no_grad():
            dist, value = self.policy(observation)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)  # sum over action dims if multi-dim

        # store
        self.buffer.observations.append(observation)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.values.append(value)

        return action.cpu().numpy()  # env may expect numpy

    def store_reward(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    # ------------------------------------------------------------------
    # PPO Update -------------------------------------------------------
    # ------------------------------------------------------------------

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
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update_policy(self, next_observation):
        """Run PPO update using the collected rollout buffer."""
        with torch.no_grad():
            _, next_value = self.policy(next_observation)
        returns, advantages = self._compute_returns_and_advantages(next_value)

        # flatten stored tensors
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        old_values = torch.stack(self.buffer.values).to(self.device)

        # Because observations are PyG Data objects, we keep them as list
        dataset = list(zip(self.buffer.observations, old_actions, old_logprobs, returns, advantages))

        for _ in range(self.update_epochs):
            for obs, act, old_lp, ret, adv in dataset:
                dist, value = self.policy(obs)
                logprob = dist.log_prob(act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                ratio = torch.exp(logprob - old_lp.detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value.squeeze(-1), ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        # clear buffer
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Training Loop ----------------------------------------------------
    # ------------------------------------------------------------------

    def train(self, total_episodes: int, max_steps: int = 512):
        for ep in range(total_episodes):
            obs = self.env.reset().to(self.device)
            episode_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.store_reward(reward, done)
                episode_reward += reward

                if done or step == max_steps - 1:
                    # compute update & break
                    self.update_policy(next_obs.to(self.device))
                    print(f"Episode {ep} — reward {episode_reward:.3f}")
                    break
                obs = next_obs.to(self.device)
