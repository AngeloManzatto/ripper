"""
Created on Sun Sep 13 16:39:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy
import random

from agents.base_agent import BaseAgent
from agents.ddqn.replay_buffer import ReplayBuffer
from agents.ddqn.model import DQN

import torch
import torch.nn as nn

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
        self.epsilon_init = epsilon  
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
        self.epsilon = 0.05

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

    def clone(self):
        
        # Deep copy networks
        policy_net = copy.deepcopy(self.policy_net).to(self.device)
        target_net = copy.deepcopy(self.target_net).to(self.device)
        
        # Deep copy learning rate
        lr = self.optimizer.param_groups[0]["lr"]
        
        # Reset optimizer 
        optimizer = type(self.optimizer)(policy_net.parameters(), lr=lr)
        
        # Return a fresh copy
        return DDQNAgent(
            policy_net,
            target_net,
            optimizer,
            self.loss_fn,
            self.device,
            epsilon=self.epsilon_init,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            buffer_capacity=self.replay_buffer.buffer.maxlen,
            batch_size=self.batch_size,
            min_buffer_size=self.min_buffer_size,
            target_update_freq=self.target_update_freq,
            gamma=self.gamma,
            log_every=self.log_every,
        )


###############################################################################
# Build agent
###############################################################################

def build_ddqn_agent(
    width,
    height,
    n_actions=4,
    channels=7,
    lr=1e-4,
    device=None,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9999,
    buffer_capacity=10000,
    batch_size=128,
    min_buffer_size=1000,
    target_update_freq=500,
    gamma=0.99,
    log_every=50,
):
    
    # Initialize device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize policy and target models
    policy_net = DQN(height, width, n_actions, channels).to(device)
    target_net = DQN(height, width, n_actions, channels).to(device)
    
    # Copy policy weights into target net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Initialize optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    
    # Initialize loss
    loss_fn = nn.MSELoss()

    return DDQNAgent(
        policy_net, 
        target_net, 
        optimizer, 
        loss_fn, 
        device,
        epsilon=epsilon, 
        epsilon_min=epsilon_min, 
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity, 
        batch_size=batch_size,
        min_buffer_size=min_buffer_size, 
        target_update_freq=target_update_freq,
        gamma=gamma, 
        log_every=log_every,
    )