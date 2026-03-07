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
        with torch.no_grad():
            action_dist, value = self.policy(observation)
            action = action_dist.sample()
            logprob = action_dist.log_prob(action).sum(dim=-1)
        
        # Store complete transition data for PPO
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
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update_policy(self, next_observation, physics_losses=None):
        """Run PPO update using the collected rollout buffer."""
        with torch.no_grad():
            _, next_value = self.policy(next_observation)
        returns, advantages = self._compute_returns_and_advantages(next_value)

        # flatten stored tensors
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        old_values = torch.stack(self.buffer.values).to(self.device)

        dataset = list(zip(self.buffer.observations, old_actions, old_logprobs, old_values, returns, advantages))
        
        # Add early stopping
        best_loss = float('inf')
        patience = 3
        no_improve = 0

        for epoch in range(self.update_epochs):
            epoch_loss = 0.0
            for obs, act, old_lp, old_val, ret, adv in dataset:
                action_dist, value = self.policy(obs)
                logprob = action_dist.log_prob(act).sum(dim=-1)
                entropy = action_dist.entropy().sum(dim=-1)

                # PPO clipped policy loss
                ratio = torch.exp(logprob - old_lp.detach())
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = F.mse_loss(value.squeeze(), ret)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Physics-informed loss component
                physics_loss = torch.tensor(0.0, device=self.device)
                if physics_losses is not None:
                    # Incorporate PINN losses into policy training
                    pde_loss = physics_losses.get('avg_pde_loss', 0.0)
                    c_mse_loss = physics_losses.get('avg_c_mse_loss', 0.0) 
                    t_mse_loss = physics_losses.get('avg_t_mse_loss', 0.0)
                    
                    # Convert to tensors if they aren't already
                    if not isinstance(pde_loss, torch.Tensor):
                        pde_loss = torch.tensor(pde_loss, device=self.device)
                    if not isinstance(c_mse_loss, torch.Tensor):
                        c_mse_loss = torch.tensor(c_mse_loss, device=self.device)
                    if not isinstance(t_mse_loss, torch.Tensor):
                        t_mse_loss = torch.tensor(t_mse_loss, device=self.device)
                    
                    # Weighted combination of physics losses
                    physics_loss = (pde_loss + c_mse_loss + t_mse_loss)

                # Combined PPO + Physics loss
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

    def train(self, max_steps, rollout_length=10):
        """
        Physics-informed PPO training loop that incorporates PINN losses.
        
        Args:
            max_steps: Total number of environment steps
            rollout_length: Number of steps per rollout before PPO update
        """
        reward_history = []
        obs = self.env.reset().to(self.device)
        
        episode_reward = 0.0
        step_count = 0
        latest_physics_losses = None
        
        print(f"Starting Physics-Informed PPO training for {max_steps} steps...")
        
        while step_count < max_steps:
            # Collect rollout
            rollout_physics_losses = []
            for rollout_step in range(rollout_length):
                if step_count >= max_steps:
                    break
                    
                print(f"---- Step {step_count}/{max_steps}")
                
                # Get action from policy
                action = self.select_action(obs)
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Store reward and done flag
                self.store_reward(reward, done)
                episode_reward += reward
                step_count += 1
                
                # Collect physics losses for PPO update
                physics_losses = {
                    'avg_pde_loss': info.get('avg_pde_loss', 0.0),
                    'avg_c_mse_loss': info.get('avg_c_mse_loss', 0.0),
                    'avg_t_mse_loss': info.get('avg_t_mse_loss', 0.0)
                }
                rollout_physics_losses.append(physics_losses)
                latest_physics_losses = physics_losses
                
                # Log physics metrics from environment
                if step_count % 5 == 0:
                    print(f"Physics metrics - PDE: {physics_losses['avg_pde_loss']:.6f}, "
                          f"C MSE: {physics_losses['avg_c_mse_loss']:.6f}, "
                          f"T MSE: {physics_losses['avg_t_mse_loss']:.6f}, "
                          f"Reward: {reward:.4f}")
                
                obs = next_obs.to(self.device)
                
                if done:
                    print(f"Episode finished with reward: {episode_reward:.4f}")
                    reward_history.append(episode_reward)
                    episode_reward = 0.0
                    obs = self.env.reset().to(self.device)
                    break
            
            # PPO update after rollout with physics losses
            if len(self.buffer.observations) > 0:
                # Average physics losses over the rollout
                if rollout_physics_losses:
                    avg_physics_losses = {
                        'avg_pde_loss': sum(p['avg_pde_loss'] for p in rollout_physics_losses) / len(rollout_physics_losses),
                        'avg_c_mse_loss': sum(p['avg_c_mse_loss'] for p in rollout_physics_losses) / len(rollout_physics_losses),
                        'avg_t_mse_loss': sum(p['avg_t_mse_loss'] for p in rollout_physics_losses) / len(rollout_physics_losses)
                    }
                    print(f"PPO update with physics losses - PDE: {avg_physics_losses['avg_pde_loss']:.6f}, "
                          f"C MSE: {avg_physics_losses['avg_c_mse_loss']:.6f}, "
                          f"T MSE: {avg_physics_losses['avg_t_mse_loss']:.6f}")
                else:
                    avg_physics_losses = latest_physics_losses
                
                print(f"Performing Physics-Informed PPO update with {len(self.buffer.observations)} transitions")
                self.update_policy(obs, physics_losses=avg_physics_losses)
        
        return reward_history
