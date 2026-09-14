"""
Created on Sun Sep 13 19:08:16 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from agents.ppo.model import ActorCritic

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
        epochs_per_update=4,
        batch_size=64,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        self.eval_mode = False

    def set_eval_mode(self):
        self.eval_mode = True

    def set_train_mode(self):
        self.eval_mode = False

    def act(self, observation):
        # next piece to build
        ...

    def train(self, layout, episodes, max_tick=100, verbose=False):
        # to build after act
        ...

    def save(self, path):
        # to build
        ...

    def load(self, path):
        # to build
        ...