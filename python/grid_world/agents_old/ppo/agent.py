"""
Created on Sun Sep 13 19:08:16 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections import deque, Counter

import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from agents.ppo.model import ActorCritic
from agents.reward import reward_for

import ripper
from ripper import Action

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Compute GAE
###############################################################################

def compute_gae(
    rewards,
    values,
    dones,
    final_value,
    gamma,
    gae_lambda
):
    advantages = [0.0] * len(rewards)

    gae = 0.0
    next_value = float(final_value)

    for t in reversed(range(len(rewards))):

        nonterminal = 1.0 - float(dones[t])

        delta = (
            rewards[t]
            + gamma * next_value * nonterminal
            - values[t]
        )

        gae = (
            delta
            + gamma * gae_lambda * nonterminal * gae
        )

        advantages[t] = gae
        next_value = values[t]

    return advantages

###############################################################################
# Collect Samples 
###############################################################################

def collect_samples(agent, world, observation, rollout_steps):
    player_id = world.get_player_id()

    states, actions, rewards, log_probs, values, dones, reasons = [], [], [], [], [], [], []

    for i in range(rollout_steps):
        action_idx = agent.act(observation.grid)
        step_result = world.step({player_id: ACTIONS[action_idx]})
        done, reason = step_result.done, step_result.reason
        reward = reward_for(reason)

        states.append(observation.grid)
        actions.append(action_idx)
        log_probs.append(agent.last_log_prob)
        values.append(agent.last_value)
        rewards.append(reward)
        dones.append(done)

        if done:
            reasons.append(str(reason))
            world.reset()

        observation = world.observation()

    with torch.no_grad():
        final_state_tensor = torch.tensor(
            observation.grid,
            dtype=torch.float32,
            device=agent.device
        ).unsqueeze(0)
    
        _, final_value = agent.model(final_state_tensor)
    
    final_value = final_value.item()
    
    return states, actions, rewards, log_probs, values, dones, reasons, final_value, observation


###############################################################################
# PPO Agent
###############################################################################

class PPOAgent(BaseAgent):

    def __init__(
        self,
        model,
        optimizer,
        device,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        epochs_per_update=4,
        batch_size=64,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        self.eval_mode = False
        
        self.recent_reasons = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)

    def set_eval_mode(self):
        self.eval_mode = True
        self.model.eval()
    
    def set_train_mode(self):
        self.eval_mode = False
        self.model.train()

    def act(self, observation):
    
        state_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
    
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            dist = Categorical(logits=logits)
    
            if self.eval_mode:
                return logits.argmax(dim=1).item()
    
            action = dist.sample()
    
            self.last_log_prob = dist.log_prob(action).item()
            self.last_value = value.item()
    
        return action.item()

    def train(self, layout, episodes, max_tick=100, rollout_steps=512, verbose=False, log_every=50):
        world = ripper.PyWorld(layout, max_tick)
        world.reset()
        observation = world.observation()

        for iteration in range(episodes):
            states, actions, rewards, log_probs, values, dones, reasons, final_value, observation = collect_samples(
                self, world, observation, rollout_steps
            )
    
            self.recent_reasons.extend(reasons)   # multiple outcomes per rollout
    
            advantages = compute_gae(rewards, values, dones, final_value, self.gamma, self.gae_lambda)
            returns = [
                adv + val
                for adv, val in zip(advantages, values)
            ]
                
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
            old_log_probs_tensor = torch.tensor(
                log_probs,
                dtype=torch.float32,
                device=self.device
            )
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()
            ) / (advantages_tensor.std(unbiased=False) + 1e-8)
            
            returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
    
            num_samples = states_tensor.size(0)

            for epoch in range(self.epochs_per_update):
            
                indices = torch.randperm(num_samples, device=self.device)
            
                for start in range(0, num_samples, self.batch_size):
            
                    idx = indices[start:start + self.batch_size]
            
                    logits, values_pred = self.model(states_tensor[idx])
                    dist = Categorical(logits=logits)
            
                    new_log_probs = dist.log_prob(actions_tensor[idx])
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(
                        new_log_probs - old_log_probs_tensor[idx]
                    )
            
                    mb_advantages = advantages_tensor[idx]
            
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1 - self.clip_epsilon,
                            1 + self.clip_epsilon
                        )
                        * mb_advantages
                    )
            
                    policy_loss = -torch.min(surr1, surr2).mean()
            
                    critic_loss = nn.functional.mse_loss(
                        values_pred.squeeze(-1),
                        returns_tensor[idx]
                    )
            
                    loss = policy_loss + 0.5 * critic_loss - self.entropy_coef * entropy 
            
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    
                    self.recent_losses.append(loss.item())
    
            if verbose and iteration % log_every == 0 and len(self.recent_reasons) > 0:
                win_rate = self.recent_reasons.count("EndReason.GoalReached") / len(self.recent_reasons)
                avg_loss = sum(self.recent_losses) / len(self.recent_losses) if self.recent_losses else 0.0
                breakdown = Counter(self.recent_reasons)
                print(f"Iteration {iteration}, win_rate(last {len(self.recent_reasons)}): {win_rate:.2%}, "
                      f"avg_loss: {avg_loss:.4f}, outcomes: {dict(breakdown)}")

    def save(self, path):
      torch.save({
          "model": self.model.state_dict(),
          "optimizer": self.optimizer.state_dict(),
      }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

###############################################################################
# Build PPO agent
###############################################################################

def build_ppo_agent(
    width,
    height,
    n_actions=4,
    channels=5,
    lr=1e-4,
    device=None,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    epochs_per_update=4,
    batch_size=128,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(height, width, n_actions, channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return PPOAgent(
        model, optimizer, device,
        gamma=gamma, gae_lambda=gae_lambda, clip_epsilon=clip_epsilon, entropy_coef=entropy_coef,
        epochs_per_update=epochs_per_update, batch_size=batch_size,
    )