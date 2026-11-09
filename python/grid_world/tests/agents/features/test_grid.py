"""
Created on Thu Oct 15 21:38:58 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from agents.features.grid import (
    UNKNOWN, FREE, WALL, PLAYER, ENEMY, TRAP, GOAL,
    SYMBOLS, N_CHANNELS,
    one_hot_encode_grid, locate, manhattan_distance, offset_to_nearest,
    neighbors, slice_2d_list, unknown_fractions,
)

###############################################################################
# Fixture: a 3x3 grid with one unknown cell, one wall and one goal
###############################################################################
#
#        col0   col1   col2
# row0    .      .      .
# row1    #      ?      .
# row2    .      G      .

GRID = [
    [FREE, FREE,    FREE],
    [WALL, UNKNOWN, FREE],
    [FREE, GOAL,    FREE],
]


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def all_close(xs, ys):
    return len(xs) == len(ys) and all(close(x, y) for x, y in zip(xs, ys))

###############################################################################
# Cell codes
###############################################################################

def test_symbols_cover_every_code():
    codes = [UNKNOWN, FREE, WALL, PLAYER, ENEMY, TRAP, GOAL]
    assert sorted(SYMBOLS) == sorted(codes)
    assert len(set(SYMBOLS.values())) == len(codes)     # one distinct character each
    assert N_CHANNELS == len(codes) == 7

###############################################################################
# One hot encode
###############################################################################

def test_one_hot_shape_and_dtype():
    out = one_hot_encode_grid(GRID)
    assert out.shape == (N_CHANNELS, 3, 3)
    assert out.dtype == np.float32


def test_one_hot_every_cell_has_exactly_one_channel():
    out = one_hot_encode_grid(GRID)
    assert np.allclose(out.sum(axis=0), 1.0)


def test_one_hot_places_known_cells():
    out = one_hot_encode_grid(GRID)
    assert out[0, 1, 1] == 1        # unknown channel at (1,1)
    assert out[2, 1, 0] == 1        # wall channel at (1,0)
    assert out[6, 2, 1] == 1        # goal channel at (2,1)


def test_one_hot_channel_order_and_non_square():
    grid = [
        [UNKNOWN, FREE, WALL, PLAYER],
        [ENEMY,   TRAP, GOAL, FREE],
    ]
    out = one_hot_encode_grid(grid)
    assert out.shape == (7, 2, 4)
    # channel order: unknown, free, wall, player, enemy, trap, goal
    assert out.sum(axis=(1, 2)).tolist() == [1, 2, 1, 1, 1, 1, 1]
    assert out[3, 0, 3] == 1        # player
    assert out[4, 1, 0] == 1        # enemy
    assert out[5, 1, 1] == 1        # trap

###############################################################################
# Locate
###############################################################################

def test_locate_single_and_none():
    assert locate(GRID, GOAL) == [(2, 1)]
    assert locate(GRID, TRAP) == []


def test_locate_is_row_major():
    assert locate(GRID, FREE) == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 2)]


def test_locate_non_square_grid():
    grid = [[FREE, GOAL, FREE, GOAL],
            [GOAL, FREE, FREE, FREE]]
    assert locate(grid, GOAL) == [(0, 1), (0, 3), (1, 0)]

###############################################################################
# Manhattan distance
###############################################################################

def test_manhattan_distance():
    assert manhattan_distance((0, 0), (2, 1)) == 3
    assert manhattan_distance((2, 1), (0, 0)) == 3          # symmetric
    assert manhattan_distance((1, 1), (1, 1)) == 0
    assert manhattan_distance((0, 3), (4, 0)) == 7

###############################################################################
# Offset to nearest
###############################################################################

def test_offset_to_nearest_positive_direction():
    # goal is 2 rows down, 1 col right; default scale = max(3, 3) - 1 = 2
    assert all_close(offset_to_nearest(GRID, (0, 0), GOAL), (1.0, 1.0, 0.5))


def test_offset_to_nearest_negative_direction():
    # goal is on the same row, 1 col to the left
    assert all_close(offset_to_nearest(GRID, (2, 2), GOAL), (1.0, 0.0, -0.5))


def test_offset_to_nearest_not_found_is_all_zeros():
    assert offset_to_nearest(GRID, (0, 0), TRAP) == (0.0, 0.0, 0.0)


def test_offset_to_nearest_explicit_scale():
    assert all_close(offset_to_nearest(GRID, (0, 0), GOAL, scale=4), (1.0, 0.5, 0.25))


def test_offset_to_nearest_picks_the_closest():
    grid = [[GOAL, FREE, FREE, FREE, GOAL]]                  # goals at distance 1 and 3 from col 1
    found, drow, dcol = offset_to_nearest(grid, (0, 1), GOAL)
    assert found == 1.0 and drow == 0.0
    assert close(dcol, -1 / 4)                               # scale = 5 - 1 = 4, nearest is col 0


def test_offset_to_nearest_tie_goes_to_first_in_row_major_order():
    grid = [[GOAL, FREE, GOAL]]                              # both at distance 1 from col 1
    assert all_close(offset_to_nearest(grid, (0, 1), GOAL), (1.0, 0.0, -0.5))


def test_offset_to_nearest_one_by_one_grid_does_not_divide_by_zero():
    assert all_close(offset_to_nearest([[GOAL]], (0, 0), GOAL), (1.0, 0.0, 0.0))


def test_offset_to_nearest_stays_in_unit_range():
    rng = np.random.default_rng(0)
    grid = rng.integers(-1, 6, size=(6, 6)).tolist()
    for row in range(6):
        for col in range(6):
            found, drow, dcol = offset_to_nearest(grid, (row, col), GOAL)
            assert -1.0 <= drow <= 1.0 and -1.0 <= dcol <= 1.0

###############################################################################
# Neighbors
###############################################################################

def test_neighbors_order_is_up_down_left_right():
    grid = [[FREE, WALL,  FREE],
            [GOAL, FREE,  TRAP],
            [FREE, ENEMY, FREE]]
    # origin (1,1): up = wall, down = enemy, left = goal, right = trap
    assert neighbors(grid, (1, 1), [WALL]) == [1, 0, 0, 0]
    assert neighbors(grid, (1, 1), [ENEMY]) == [0, 1, 0, 0]
    assert neighbors(grid, (1, 1), [GOAL]) == [0, 0, 1, 0]
    assert neighbors(grid, (1, 1), [TRAP]) == [0, 0, 0, 1]


def test_neighbors_map_edge_counts_as_wall():
    assert neighbors(GRID, (0, 0), [WALL]) == [1, 1, 1, 0]   # up/left outside, down is a wall
    assert neighbors(GRID, (2, 2), [WALL]) == [0, 1, 0, 1]   # down/right outside


def test_neighbors_several_symbols_are_grouped_per_direction():
    # per direction: (wall, goal). Origin (1,1): up free, down goal, left wall, right free
    assert neighbors(GRID, (1, 1), [WALL, GOAL]) == [0, 0, 0, 1, 1, 0, 0, 0]


def test_neighbors_length_and_values():
    out = neighbors(GRID, (1, 1), [WALL, TRAP, ENEMY])
    assert len(out) == 4 * 3
    assert set(out) <= {0, 1}

###############################################################################
# Slice 2D list
###############################################################################

def test_slice_rows_only():
    assert slice_2d_list(GRID, row_end=1) == [[FREE, FREE, FREE]]
    assert slice_2d_list(GRID, row_start=2) == [[FREE, GOAL, FREE]]


def test_slice_cols_only():
    assert slice_2d_list(GRID, col_end=1) == [[FREE], [WALL], [FREE]]
    assert slice_2d_list(GRID, col_start=2) == [[FREE], [FREE], [FREE]]


def test_slice_rows_and_cols():
    assert slice_2d_list(GRID, 1, 3, 1, 3) == [[UNKNOWN, FREE], [GOAL, FREE]]


def test_slice_without_bounds_is_an_equal_copy():
    out = slice_2d_list(GRID)
    assert out == GRID
    out[0][0] = WALL                                         # rows are new lists
    assert GRID[0][0] == FREE


def test_slice_empty_range():
    assert slice_2d_list(GRID, row_end=0) == []

###############################################################################
# Unknown fractions
###############################################################################

def test_unknown_fractions_from_a_corner():
    # up and left are empty halves -> 0.0; down and right each hold 6 cells, 1 unknown
    assert all_close(unknown_fractions(GRID, (0, 0)), [0.0, 1 / 6, 0.0, 1 / 6])


def test_unknown_fractions_excludes_the_origin_cell():
    # the only unknown cell is the origin itself
    assert all_close(unknown_fractions(GRID, (1, 1)), [0.0, 0.0, 0.0, 0.0])


def test_unknown_fractions_all_unknown():
    grid = [[UNKNOWN] * 3 for _ in range(3)]
    assert all_close(unknown_fractions(grid, (1, 1)), [1.0, 1.0, 1.0, 1.0])


def test_unknown_fractions_order_is_up_down_left_right():
    grid = [[FREE,    FREE, FREE],
            [UNKNOWN, FREE, FREE],                           # unknown only in the left column
            [FREE,    FREE, FREE]]
    assert all_close(unknown_fractions(grid, (1, 1)), [0.0, 0.0, 1 / 3, 0.0])

###############################################################################
# Purity
###############################################################################

def test_helpers_do_not_modify_their_input():
    before = copy.deepcopy(GRID)
    locate(GRID, FREE)
    offset_to_nearest(GRID, (0, 0), GOAL)
    neighbors(GRID, (1, 1), [WALL, GOAL])
    slice_2d_list(GRID, 1, 3, 1, 3)
    unknown_fractions(GRID, (0, 0))
    one_hot_encode_grid(GRID)
    assert GRID == before

###############################################################################
# Direct run
###############################################################################

def run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
    print(f"grid: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()