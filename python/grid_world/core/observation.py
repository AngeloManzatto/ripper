"""
Created on Sun Sep 27 09:09:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

###############################################################################
# Constants
###############################################################################


SYMBOLS = {
    -1: "?",   # Unknown
     0: ".",   # Free
     1: "#",   # Wall
     2: "P",   # Player
     3: "E",   # Enemy
     4: "T",   # Trap
     5: "G",   # Goal
}

N_CHANNELS = 7  # Unknown, Free, Wall, Player, Enemy, Trap, Goal

###############################################################################
# Render Grid
###############################################################################

def render_grid(grid: list[list[int]]) -> str:
    """Pretty-print an entity's grid from its own perspective (fog-of-war included)."""
    return "\n".join(
        "".join(SYMBOLS.get(cell, "?") for cell in row)
        for row in grid
    )

###############################################################################
# One Hot Encode Grid
###############################################################################

def one_hot_encode_grid(grid: list[list[int]]) -> np.ndarray:
    """
    Encode a raw entity grid (as returned by the engine) into a
    (channels, height, width) one-hot tensor for network input.
    """
    arr = np.asarray(grid)
    one_hot = np.zeros((*arr.shape, N_CHANNELS), dtype=np.float32)
    one_hot[arr == -1, 0] = 1  # Unknown
    one_hot[arr ==  0, 1] = 1  # Free
    one_hot[arr ==  1, 2] = 1  # Wall
    one_hot[arr ==  2, 3] = 1  # Player
    one_hot[arr ==  3, 4] = 1  # Enemy
    one_hot[arr ==  4, 5] = 1  # Trap
    one_hot[arr ==  5, 6] = 1  # Goal
    
    return np.transpose(one_hot, (2, 0, 1))