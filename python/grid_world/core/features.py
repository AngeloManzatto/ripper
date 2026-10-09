"""
Created on Wed Oct  7 09:11:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

###############################################################################
# Globals
###############################################################################

UNKNOWN, FREE, WALL, PLAYER, ENEMY, TRAP, GOAL = -1, 0, 1, 2, 3, 4, 5

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

DIRECTION_NAMES = ["up", "down", "left", "right"]          # same order as DIRECTIONS

NEIGHBOR_SYMBOLS = {"wall": WALL, "trap": TRAP, "hostile": None}   # hostile resolved per observer

TARGETS = {"goal": GOAL, "hostile": None, "trap": TRAP}

TARGETS = {"goal": GOAL, "enemy": ENEMY, "trap": TRAP}

FEATURE_NAMES = (
    [f"{d}_{s}" for d in DIRECTION_NAMES for s in NEIGHBOR_SYMBOLS]          # 12
    + [f"{t}_{part}" for t in TARGETS for part in ("found", "drow", "dcol")]  # 9
    + [f"unknown_{d}" for d in DIRECTION_NAMES]                               # 4
)

N_FEATURES = len(FEATURE_NAMES)   # 25

KIND_PLAYER, KIND_ENEMY = "player", "enemy"

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
# Locate
###############################################################################

def locate(grid, symbol_id):
    
    positions = []
    
    rows = len(grid)
    cols = len(grid[0])
    
    for row in range(rows):
        for col in range(cols):
            
            if grid[row][col] == symbol_id:
                positions.append((row,col))
                
    return positions

###############################################################################
# Manhattan distance
###############################################################################

def manhattan_distance(current_position, target_position):
    
    a = abs(current_position[0] - target_position[0])
    b = abs(current_position[1] - target_position[1])
    
    dist = a + b
    
    return dist
    
###############################################################################
# Offset to nearest
###############################################################################

def offset_to_nearest(grid, origin, symbol_id, scale=None):
    """Offset from `origin` to the nearest cell holding `symbol_id`.

    "Nearest" is by Manhattan distance, searched over the WHOLE grid, so with
    memory it also finds remembered cells that are no longer in view.
    Ties go to the first match found (order given by `locate`).

    Args:
        grid:      2D list of cell codes, indexed grid[row][col].
        origin:    (row, col) of the observer.
        symbol_id: cell code to look for (e.g. goal, trap, enemy).
        scale:     divisor that normalizes the offsets. Default (None) is the
                   largest possible offset on this map, max(rows, cols) - 1,
                   so every offset lies in [-1, 1].

    Returns:
        (found, drow, dcol), all floats:
          found: 1.0 if `symbol_id` exists in the grid, else 0.0
          drow:  (target_row - origin_row) / scale   (positive = down)
          dcol:  (target_col - origin_col) / scale   (positive = right)
        When nothing is found, returns (0.0, 0.0, 0.0).
    """
    if scale is None:
        # max(..., 1) avoids dividing by zero on a degenerate 1x1 grid
        scale = max(max(len(grid), len(grid[0])) - 1, 1)

    positions = locate(grid, symbol_id)

    min_distance = float("inf")
    found = 0.0
    drow = 0.0
    dcol = 0.0

    for position in positions:
        dist = manhattan_distance(origin, position)

        # strict "<" keeps the first of several equally near targets
        if dist < min_distance:
            min_distance = dist
            found = 1.0
            drow = (position[0] - origin[0]) / scale
            dcol = (position[1] - origin[1]) / scale

    return found, drow, dcol

###############################################################################
# Neighbors
###############################################################################

def neighbors(grid, origin, symbol_ids):
    """
    One 0/1 flag per symbol id for each neighbor, in this order:
    DIRECTIONS (Up, Down, Left, Right) x symbol_ids.
    A neighbor outside the map counts as a WALL.
    """

    rows = len(grid)
    cols = len(grid[0])

    out = []

    for d_row, d_col in DIRECTIONS:

        row = origin[0] + d_row
        col = origin[1] + d_col

        inside = 0 <= row < rows and 0 <= col < cols
        cell = grid[row][col] if inside else WALL

        out += [1 if cell == symbol_id else 0 for symbol_id in symbol_ids]

    return out

###############################################################################
# Slice 2D list
###############################################################################

def slice_2d_list(grid, row_start=None, row_end=None, col_start=None, col_end=None):
    """
    Safely slices a 2D list by rows and columns.
    
    Parameters:
        grid (list of lists): The matrix to slice.
        row_start, row_end: Row slice indices (like list slicing).
        col_start, col_end: Column slice indices.
    
    Returns:
        list of lists: The sliced submatrix.
    """
    
    # Slice rows first, then columns
    return [row[col_start:col_end] for row in grid[row_start:row_end]]

###############################################################################
# Unknown fraction
###############################################################################

def unknown_fractions(grid, origin, unknown_id=UNKNOWN):
    """Fraction of still-unknown cells in each half of the map, as seen from origin.

    Order matches DIRECTIONS: [Up, Down, Left, Right]
      Up    -> cells with row <  origin_row   (all columns)
      Down  -> cells with row >  origin_row
      Left  -> cells with col <  origin_col   (all rows)Globals
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

    # Fail loudly if the layout ever drifts from FEATURE_NAMES
    assert len(values) == N_FEATURES, f"expected {N_FEATURES} features, got {len(values)}"

    return np.asarray(values, dtype=np.float32)