"""
Created on Thu Sep 10 12:51:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import random
from collections import deque

###############################################################################
# Replay buffer
###############################################################################

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
