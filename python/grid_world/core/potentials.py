"""
Created on Wed Oct 14 07:23:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from core.cells import UNKNOWN, GOAL
from core.features import locate, manhattan_distance

###############################################################################
# Explore potencial
###############################################################################

def explore_potential(grid, unknown_id=UNKNOWN) -> float:
    """Fraction of cells that are not unknown."""

    if not grid or not grid[0]:
        raise ValueError("Grid cannot be empty")

    cells = np.asarray(grid)

    n_known = np.count_nonzero(cells != unknown_id)

    return float(n_known / cells.size)
        
###############################################################################
# Goal potencial
###############################################################################   
    
def goal_potential(grid, origin, goal_id=GOAL) -> float:
    
    # Find the goals
    positions = locate(grid, goal_id)
    
    # No goal found on the current view return 0
    if not positions:
        return 0.0
    
    rows = len(grid)
    cols = len(grid[0])
    max_distance = (rows - 1) + (cols - 1)
    
    min_distance = float("inf")
    potential = 0.0
    

    for position in positions:
        dist = manhattan_distance(origin, position)

        # strict "<" keeps the first of several equally near targets
        if dist < min_distance:
            min_distance = dist
            potential = 1.0 - min_distance / max(max_distance, 1)
            
    return potential
            