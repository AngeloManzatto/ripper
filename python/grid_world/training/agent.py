"""
Created on Thu Sep 10 12:59:56 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import torch
import torch.nn as nn

from grid_world.model import DQN

###############################################################################
# Agent container
###############################################################################

class Agent:
    def __init__(self, policy_net, target_net, optimizer, loss_fn, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

###############################################################################
# Build agent
###############################################################################

def build_agent(width, height, n_actions=4, channels=5, lr=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(height, width, n_actions, channels).to(device)
    target_net = DQN(height, width, n_actions, channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    return Agent(policy_net, target_net, optimizer, loss_fn, device)