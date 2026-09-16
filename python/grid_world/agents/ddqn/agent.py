"""
Created on Sun Sep 13 16:39:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import random
from collections import deque, Counter

from agents.base_agent import BaseAgent
from agents.reward import reward_for
from agents.ddqn.replay_buffer import ReplayBuffer
from agents.ddqn.model import DQN

import ripper
from ripper import Action

import torch
import torch.nn as nn

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Tensor utils 
###############################################################################
    
def sample_to_tensor(sample, device):
    
    states, actions, rewards, next_states, dones = zip(*sample)
    
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_state_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
    
    return states_tensor, actions_tensor, rewards_tensor, next_state_tensor, dones_tensor
    
###############################################################################
# Train
###############################################################################

def train_step(agent, batch, gamma):
    
    # Sample a batch from buffer
    states, actions, rewards, next_states, dones = sample_to_tensor(batch, agent.device)

    # Get predicted actions from model for the current state [B, n_actions]
    q_values = agent.policy_net(states)
    
    # Select the chosen action [B, 1]
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        
        # Get the action with max value from the next state for policy model
        next_actions = agent.policy_net(next_states).argmax(dim=1)
        
        # Get the action with max value from the next state for policy model
        next_q_values = agent.target_net(next_states)
        
        max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * max_next_q * (1 - dones)

    # Calculate loss 
    loss = agent.loss_fn(q_taken, target_q)
    
    # Update weights
    agent.optimizer.zero_grad()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    loss.backward()
    agent.optimizer.step()

    return loss.item()

###############################################################################
# Agent
###############################################################################

class DDQNAgent(BaseAgent):

    def __init__(
        self,
        policy_net,
        target_net,
        optimizer,
        loss_fn,
        device,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9999,
        buffer_capacity=50000,
        batch_size=128,
        min_buffer_size=1000,
        target_update_freq=500,
        gamma=0.99,
        log_every=50,
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.log_every = log_every

        self.train_step_count = 0

    def set_eval_mode(self):
        self._saved_epsilon = self.epsilon
        self.epsilon = 0.0

    def set_train_mode(self):
        self.epsilon = getattr(self, '_saved_epsilon', self.epsilon)

    def act(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, layout, episodes, max_tick=100, verbose=False):
        
        world = ripper.PyWorld(layout, max_tick)
        player_id = world.get_player_id()

        recent_reasons = deque(maxlen=100)
        recent_losses = deque(maxlen=100)

        for episode in range(episodes):
            world.reset()
            observation = world.observation()
            done = False

            while not done:
                action_idx = self.act(observation.grid)
                step_result = world.step({player_id: ACTIONS[action_idx]})
                done, reason = step_result.done, step_result.reason
                reward = reward_for(reason)

                next_observation = world.observation()
                self.replay_buffer.push(observation.grid, action_idx, reward, next_observation.grid, done)
                observation = next_observation

                if len(self.replay_buffer) >= self.min_buffer_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = train_step(self, batch, self.gamma)
                    recent_losses.append(loss)
                    self.train_step_count += 1
                    self.decay_epsilon()

                    if self.train_step_count % self.target_update_freq == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

            recent_reasons.append(str(reason))

            if verbose and episode % self.log_every == 0 and len(recent_reasons) > 0:
                win_rate = recent_reasons.count("EndReason.GoalReached") / len(recent_reasons)
                avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                breakdown = Counter(recent_reasons)
                print(f"Episode {episode}, train_steps: {self.train_step_count}, epsilon: {self.epsilon:.3f}, "
                      f"win_rate: {win_rate:.2%}, avg_loss: {avg_loss:.4f}, outcomes: {dict(breakdown)}")
                
    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_step_count = checkpoint["train_step_count"]
                
###############################################################################
# Build agent
###############################################################################

def build_ddqn_agent(
    width,
    height,
    n_actions=4,
    channels=5,
    lr=1e-4,
    device=None,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9999,
    buffer_capacity=50000,
    batch_size=128,
    min_buffer_size=1000,
    target_update_freq=500,
    gamma=0.99,
    log_every=50,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(height, width, n_actions, channels).to(device)
    target_net = DQN(height, width, n_actions, channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    return DDQNAgent(
        policy_net, target_net, optimizer, loss_fn, device,
        epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity, batch_size=batch_size,
        min_buffer_size=min_buffer_size, target_update_freq=target_update_freq,
        gamma=gamma, log_every=log_every,
    )