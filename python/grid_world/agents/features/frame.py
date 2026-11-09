"""
Created on Thu Oct 15 22:08:55 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np 

from core.actions import DIRECTION_NAMES

from agents.features.grid import UNKNOWN, WALL, TRAP, GOAL, ENEMY, PLAYER
from agents.features.grid import offset_to_nearest, neighbors, slice_2d_list

###############################################################################
# Globals
###############################################################################

KIND_PLAYER, KIND_ENEMY = "player", "enemy"

NEIGHBOR_SYMBOLS = {"wall": WALL, "trap": TRAP, "hostile": None} 

TARGETS = {"goal": GOAL, "hostile": None, "trap": TRAP}

FEATURE_NAMES = (
    [f"{d}_{s}" for d in DIRECTION_NAMES for s in NEIGHBOR_SYMBOLS]          # 12
    + [f"{t}_{part}" for t in TARGETS for part in ("found", "drow", "dcol")]  # 9
    + [f"unknown_{d}" for d in DIRECTION_NAMES]                               # 4
)

N_FEATURES = len(FEATURE_NAMES)

###############################################################################
# Hostile code
###############################################################################

def hostile_code(kind):
    """Cell code of the entities this observer should treat as hostile."""
    if kind == KIND_PLAYER:
        return ENEMY
    if kind == KIND_ENEMY:
        return PLAYER
    raise ValueError(f"Unknown entity kind: {kind!r}")

###############################################################################
# Unknown fraction
###############################################################################

def unknown_fractions(grid, origin, unknown_id=UNKNOWN):
    """Fraction of still-unknown cells in each half of the map, as seen from origin.

    Order matches DIRECTIONS: [Up, Down, Left, Right]
      Up    -> cells with row <  origin_row   (all columns)
      Down  -> cells with row >  origin_row
      Left  -> cells with col <  origin_col   (all rows)
      Right -> cells with col >  origin_col
    The observer's own row/column is excluded. An empty half returns 0.0.
    """
    
    unknowns = []
    
    halves = [
        slice_2d_list(grid, row_end=origin[0]),          # Up
        slice_2d_list(grid, row_start=origin[0] + 1),    # Down
        slice_2d_list(grid, col_end=origin[1]),          # Left
        slice_2d_list(grid, col_start=origin[1] + 1),    # Right
    ]
    
    for half in halves:
            
        # Flat list of cells
        cells = [item for sublist in half for item in sublist]
        
        unknown_fraction = 0.0
        
        if  cells:
            
            n_cells = len(cells)
            unk_cells = cells.count(unknown_id)
            
            # Calcualte fraction n_unk_cells / total_cells
            unknown_fraction = unk_cells / n_cells
        
        unknowns.append(unknown_fraction)
                
    return unknowns

###############################################################################
# Features
###############################################################################

def features(entity_obs):
    """Fixed-length float32 feature vector for one entity (layout: FEATURE_NAMES).

    Everything is computed from the entity's own grid (current view + remembered
    terrain), so it never uses information the entity doesn't have.

    Layout (25 values):
        [ 0..11]  neighbor flags: for up, down, left, right -> (wall, trap, hostile)
        [12..14]  nearest goal:    (found, drow, dcol)
        [15..17]  nearest hostile: (found, drow, dcol)  -- only while visible
        [18..20]  nearest trap:    (found, drow, dcol)
        [21..24]  fraction of unexplored cells up / down / left / right
    Offsets are scaled by the map size, so they lie in [-1, 1].
    """
    grid = entity_obs.grid
    origin = entity_obs.position                       # (row, col)

    # Cell code of whoever is dangerous to this observer:
    # enemies for the player, the player for an enemy
    hostile_id = hostile_code(entity_obs.kind)

    # What lies in each of the 4 adjacent cells (the map edge counts as wall)
    neighbor_flags = neighbors(grid, origin, [WALL, TRAP, hostile_id])

    # Nearest known goal / hostile / trap: (found, drow, dcol).
    # Goal and trap are terrain, so they stay known after leaving the view;
    # hostiles are entities, so they only appear while in sight.
    goal_target    = offset_to_nearest(grid, origin, GOAL)
    hostile_target = offset_to_nearest(grid, origin, hostile_id)
    trap_target    = offset_to_nearest(grid, origin, TRAP)

    # How much of each half of the map is still unexplored
    unknown_flags = unknown_fractions(grid, origin)

    values = (
        neighbor_flags
        + list(goal_target)
        + list(hostile_target)
        + list(trap_target)
        + list(unknown_flags)
    )
    
    assert len(values) == N_FEATURES

    return np.asarray(values, dtype=np.float32)