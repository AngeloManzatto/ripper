"""
Created on Thu Oct 15 22:54:51 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections.abc import Sequence

from core.actions import DIRECTIONS, DIRECTION_NAMES
from engine import EntityObservation

###############################################################################
# Globals
###############################################################################

N_MEMORY = len(DIRECTIONS)

MEMORY_NAMES = tuple(f"visited_{d}" for d in DIRECTION_NAMES)

###############################################################################
# Recent visit flags
###############################################################################

def recent_visit_flags(position, recent_positions):

    row, col = position
    visited = {tuple(p) for p in recent_positions}

    flags = []

    for (d_row, d_col) in DIRECTIONS:

        if (row + d_row, col + d_col) in visited:
            flags.append(1.0)
        else:
            flags.append(0.0)

    return flags

###############################################################################
# Memory flags
###############################################################################

def memory_flags(history: Sequence[EntityObservation]) -> list[float]:
    """Recent-visit flags for the current cell, from a history of observations.

    history: observations, oldest first, current one last (at least one).
    Returns N_MEMORY floats in DIRECTIONS order (up, down, left, right): 1.0
    when the neighbor in that direction is a position held in the history.
    """
    if len(history) == 0:
        raise ValueError("history must contain at least the current observation")

    positions = [o.position for o in history]

    return recent_visit_flags(history[-1].position, positions)