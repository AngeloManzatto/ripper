"""
Created on Thu Oct 15 21:47:44 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from core.actions import DIRECTIONS
from core.cells import WALL

###############################################################################
# Locate symbols
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
