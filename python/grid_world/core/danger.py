"""
Created on Thu Oct 15 08:25:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from core.actions import DIRECTIONS

N_PROXIMITY = len(DIRECTIONS)

###############################################################################
# Hostile proximity flags
###############################################################################

def hostile_proximity(found, drow, dcol, radius=5.0) -> list[float]:
    """up, down, left, right closeness of the nearest visible hostile, in [0, 1].
    drow, dcol in cells (not scaled), relative to the player. 0.0 everywhere if not found."""
    
    flags = []

    for (d_row, d_col) in DIRECTIONS:

        along = d_row * drow + d_col * dcol

        if found and along > 0:
            flags.append(max(0.0, 1.0 - along / radius))
        else:
            flags.append(0.0)

    return flags