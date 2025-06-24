import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple
from .constants import PHYSICS_LOSS_WEIGHT, EIKONAL_TOLERANCE
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        physics_weight: float = PHYSICS_LOSS_WEIGHT,
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
        self.physics_weight = physics_weight

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Interaction ------------------------------------------------------
    # ------------------------------------------------------------------

    def select_action(self, observation):
        """Run policy → sample action → store transition in buffer."""
        # policy must internally handle PyG Data object
        with torch.no_grad():
            action = self.policy(observation)
        # store
        self.buffer.observations.append(observation)
        self.buffer.actions.append(action)


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
                #- self.buffer.values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            #next_val = self.buffer.values[step]
            #returns.insert(0, gae + self.buffer.values[step])
            returns.insert(0, gae )

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Add value function baseline subtraction
        advantages = advantages - advantages.mean()
        # Normalize advantages
        advantages = advantages / (advantages.std() + 1e-8)
        return returns, advantages

    def update_policy(self, next_observation):
        """Run PPO update using the collected rollout buffer."""
        with torch.no_grad():
            next_value = self.policy(next_observation)
        returns, advantages = self._compute_returns_and_advantages(next_value)

        # flatten stored tensors
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        #old_values = torch.stack(self.buffer.values).to(self.device)

        dataset = list(zip(self.buffer.observations, old_actions, old_logprobs, returns, advantages))
        
        # Add early stopping
        best_loss = float('inf')
        patience = 3
        no_improve = 0

        for epoch in range(self.update_epochs):
            epoch_loss = 0.0
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

                # Compute physics loss
                T = obs.x[:, 0]  # Time of flight values
                c = obs.x[:, 1]  # Speed of sound values
                pos = obs.pos    # Node positions
                physics_loss = self.compute_physics_loss(T, c, pos)

                # Combine losses with physics weight
                loss = (policy_loss + 
                       self.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss +
                       self.physics_weight * physics_loss)
                
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.buffer.clear()

    # ------------------------------------------------------------------
    # Training Loop ----------------------------------------------------
    # ------------------------------------------------------------------

    def train(self, max_steps):
        optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        reward_history = []
        obs = self.env.reset().to(self.device)

        episode_reward = 0.0
        episode_length = 0

        r_weight = 0
        p_weight = 1
        mc_weight = 1
        mt_weight = 1e-6  # Increased from 1e-6 to 1.0
        
        # Initialize running statistics for normalization
        running_mean = torch.zeros(1, device=self.device)
        running_std = torch.ones(1, device=self.device)
        momentum = 0.99

        for step in range(max_steps):
            print(f"---- step {step}")
            # Get action (ΔSoS) from policy
            action = self.select_action(obs)
            
            # Step environment - update SoS and compute physics loss
            next_obs, reward, done, info = self.env.step(action)

            # Get losses from environment
            r_loss = info['avg_reward']  # Reward-based loss
            p_loss = info['avg_pde_loss']  # Physics (Eikonal) loss
            c_loss = info['avg_c_mse_loss']  # SoS prediction loss
            t_loss = info['avg_t_mse_loss']  # TOF prediction loss




            # Combine losses to guide policy
            loss = r_weight * r_loss + p_weight * p_loss + mc_weight * c_loss + mt_weight * t_loss
            print(f"---- loss {r_loss} {p_loss} {c_loss} {t_loss} ")
            print(f"---- loss {loss} ")
            
            # Backpropagate to improve policy's actions
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for all parameters
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Update learning rate based on loss
            scheduler.step(loss)

            # Store transition for PPO update
            self.store_reward(reward, done)
            episode_reward += reward
            episode_length += 1
            obs = next_obs.to(self.device)

            if done:
                self.update_policy(obs)
                reward_history.append(episode_reward)
                episode_reward = 0.0
                episode_length = 0
                obs = self.env.reset().to(self.device)
        
        return reward_history
