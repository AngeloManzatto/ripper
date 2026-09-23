"""
Created on Thu Sep 10 12:54:09 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import torch.nn as nn

###############################################################################
# Model
###############################################################################

class DQN(nn.Module):
    def __init__(self, height, width, n_actions=4, channels=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * height * width, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x