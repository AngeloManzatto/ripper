"""
Created on Thu Sep 24 21:53:31 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np
from collections import deque

###############################################################################
# Trail Traker
###############################################################################

class TrailTracker:
    """
    Tracks an agent's own recent positions and paints them onto a grid
    channel. One instance per agent (player or enemy) -- role-agnostic,
    since it only ever knows about the entity that owns it.
    """

    def __init__(self, grid_shape, maxlen=4, decay=0.5):
        self.grid_shape = grid_shape
        self.maxlen = maxlen
        self.decay = decay
        self.history = deque(maxlen=maxlen)

    def reset(self):
        self.history.clear()

    def update(self, position):
        self.history.append(position)

    def get_grid(self):
        grid = np.zeros(self.grid_shape, dtype=np.float32)
        for i, pos in enumerate(reversed(self.history)):
            value = self.decay ** i  # most recent = 1.0, fading backward
            grid[pos[0], pos[1]] = max(grid[pos[0], pos[1]], value)
        return grid
    
 