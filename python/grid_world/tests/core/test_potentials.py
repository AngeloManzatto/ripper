"""
Created on Wed Oct 14 07:23:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from core.cells import GOAL, UNKNOWN, WALL
from core.potentials import explore_potential, goal_potential


def raises(error, fn, *args):
    try:
        fn(*args)
    except error:
        return True
    return False


def blank(rows, cols, fill=WALL):
    return [[fill for _ in range(cols)] for _ in range(rows)]


def test_explore_potential_is_the_known_fraction():
    assert np.isclose(explore_potential(blank(2, 2, UNKNOWN)), 0.0)
    assert np.isclose(explore_potential(blank(2, 2, WALL)), 1.0)
    assert np.isclose(explore_potential([[UNKNOWN, UNKNOWN], [WALL, WALL]]), 0.5)


def test_explore_potential_empty_grid_raises():
    assert raises(ValueError, explore_potential, [])


def test_no_known_goal_gives_zero():
    assert goal_potential(blank(5, 5), (2, 2)) == 0.0


def test_standing_on_the_goal_gives_one():
    grid = blank(5, 5)
    grid[2][3] = GOAL
    assert np.isclose(goal_potential(grid, (2, 3)), 1.0)


def test_distance_scaling_square_map():
    grid = blank(5, 5)
    grid[4][4] = GOAL                                   # max distance on 5x5 is 8
    assert np.isclose(goal_potential(grid, (0, 0)), 0.0)
    assert np.isclose(goal_potential(grid, (2, 2)), 0.5)
    assert np.isclose(goal_potential(grid, (4, 3)), 1 - 1 / 8)


def test_distance_scaling_rectangular_map():
    grid = blank(3, 6)
    grid[2][5] = GOAL                                   # max distance is 2 + 5 = 7
    assert np.isclose(goal_potential(grid, (0, 0)), 0.0)
    assert np.isclose(goal_potential(grid, (2, 4)), 1 - 1 / 7)


def test_nearest_goal_wins():
    grid = blank(5, 5)
    grid[0][1] = GOAL
    grid[4][4] = GOAL
    assert np.isclose(goal_potential(grid, (0, 0)), 1 - 1 / 8)


def test_potential_is_always_in_unit_interval():
    for goal_row in range(5):
        for goal_col in range(5):
            grid = blank(5, 5)
            grid[goal_row][goal_col] = GOAL
            for row in range(5):
                for col in range(5):
                    value = goal_potential(grid, (row, col))
                    assert 0.0 <= value <= 1.0


def test_moving_closer_raises_the_potential():
    grid = blank(5, 5)
    grid[0][4] = GOAL
    values = [goal_potential(grid, (0, col)) for col in range(5)]
    assert all(a < b for a, b in zip(values, values[1:]))


def run():
    test_explore_potential_is_the_known_fraction()
    test_explore_potential_empty_grid_raises()
    test_no_known_goal_gives_zero()
    test_standing_on_the_goal_gives_one()
    test_distance_scaling_square_map()
    test_distance_scaling_rectangular_map()
    test_nearest_goal_wins()
    test_potential_is_always_in_unit_interval()
    test_moving_closer_raises_the_potential()
    print("potentials: all checks passed")


if __name__ == "__main__":
    run()