"""
Created on Wed Oct 14 15:41:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from core.actions import DIRECTIONS

N_MEMORY = len(DIRECTIONS)

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