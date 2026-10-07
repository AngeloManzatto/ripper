"""
Created on Wed Oct  7 09:11:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################


###############################################################################
# Globals
###############################################################################

WALL = 1

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def locate(grid, symbol_id):
    
    positions = []
    
    rows = len(grid)
    cols = len(grid[0])
    
    for x in range(rows):
        for y in range(cols):
            
            if grid[x][y] == symbol_id:
                positions.append((x,y))
                
    return positions
    
def manhattan_distance(current_postion, target_position):
    
    a = abs(current_postion[0] - target_position[0])
    b = abs(current_postion[1] - target_position[1])
    
    dist = a + b
    
    return dist
    
def offset_to_nearest(grid, origin, symbol_id, view_range=5):
    
    positions = locate(grid, symbol_id)
    
    min_distance = float('inf')
    visible = 0
    drow = 0
    dcol = 0
    
    for position in positions:
        
        dist = manhattan_distance(origin, position)

        if dist < min_distance:
            min_distance = dist
            visible = 1.0
            drow = (position[0] - origin[0])  / view_range
            dcol = (position[1] - origin[1])  / view_range
            
    return visible, drow, dcol

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