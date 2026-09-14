"""
Created on Sun Sep 13 17:49:39 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import torch.nn as nn

###############################################################################
# Actor-Critic Model
###############################################################################

class ActorCritic(nn.Module):
    def __init__(self, height, width, n_actions=4, channels=5):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Shared layer
        self.shared_fc = nn.Sequential(
            nn.Linear(32 * height * width, 128),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor_head = nn.Linear(128, n_actions)
        
        # Critic head
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        
        logits = self.actor_head(x)
        value  = self.critic_head(x)

        return logits, value