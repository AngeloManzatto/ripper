"""
Created on Wed Oct 14 10:16:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from types import SimpleNamespace

import numpy as np

from core.cells import GOAL, UNKNOWN, WALL
from core.rewards import ShapedGoalReward, SparseGoalReward


def blank(rows, cols, fill=WALL):
    return [[fill for _ in range(cols)] for _ in range(rows)]


def with_goal(rows, cols, goal):
    grid = blank(rows, cols)
    grid[goal[0]][goal[1]] = GOAL
    return grid


def transition(grid, position, next_grid, next_position, status="Playing", terminated=None):
    if terminated is None:
        terminated = status != "Playing"
    obs = SimpleNamespace(grid=grid, position=position, status="Playing")
    next_obs = SimpleNamespace(grid=next_grid, position=next_position, status=status)
    return SimpleNamespace(obs=obs, next_obs=next_obs, terminated=terminated)


def test_zero_weights_equal_the_sparse_reward():
    shaped = ShapedGoalReward(gamma=0.95, w_goal=0.0, w_explore=0.0)
    sparse = SparseGoalReward()
    grid = with_goal(5, 5, (2, 2))
    for status in ("Playing", "GoalReached"):
        t = transition(grid, (0, 0), grid, (0, 1), status)
        assert np.isclose(shaped(t), sparse(t))


def test_step_toward_the_goal_pays_and_step_away_costs():
    reward = ShapedGoalReward(gamma=1.0, w_goal=1.0, w_explore=0.0)
    grid = with_goal(5, 5, (0, 4))                       # max distance on 5x5 is 8
    toward = transition(grid, (0, 1), grid, (0, 2))
    away = transition(grid, (0, 2), grid, (0, 1))
    assert np.isclose(reward(toward), 1 / 8)
    assert np.isclose(reward(away), -1 / 8)


def test_discovering_the_goal_is_a_positive_jump():
    reward = ShapedGoalReward(gamma=1.0, w_goal=1.0, w_explore=0.0, w_bump=0.0)
    before = blank(5, 5)
    before[2][2] = UNKNOWN                               # goal not seen yet
    after = with_goal(5, 5, (2, 2))                      # goal now known
    t = transition(before, (2, 0), after, (2, 0))
    assert np.isclose(reward(t), 1 - 2 / 8)              # potential 0 -> 0.75


def test_revealing_cells_pays_the_explored_fraction():
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=1.0, w_bump=0.0)
    before = [[WALL, UNKNOWN], [UNKNOWN, WALL]]          # 0.50 known
    after = [[WALL, WALL], [UNKNOWN, WALL]]              # 0.75 known
    t = transition(before, (0, 0), after, (0, 0))
    assert np.isclose(reward(t), 0.25)


def test_gamma_discounts_the_next_potential_only():
    reward = ShapedGoalReward(gamma=0.9, w_goal=1.0, w_explore=0.0, w_bump=0.0)
    grid = with_goal(5, 5, (2, 2))                       # standing at (2,0): potential 0.75
    t = transition(grid, (2, 0), grid, (2, 0))
    assert np.isclose(reward(t), (0.9 - 1.0) * 0.75)


def test_reaching_the_goal_pays_bonus_plus_last_progress():
    reward = ShapedGoalReward(gamma=1.0, w_goal=1.0, w_explore=0.0, w_bump=0.0)
    grid = with_goal(5, 5, (0, 4))
    t = transition(grid, (0, 3), grid, (0, 4), status="GoalReached")
    assert np.isclose(reward(t), 1.0 + (1.0 - 7 / 8))


def test_shaping_telescopes_along_a_path():
    reward = ShapedGoalReward(gamma=1.0, w_goal=1.0, w_explore=0.0)
    grid = with_goal(5, 5, (0, 4))
    path = [(0, 0), (0, 1), (0, 2), (0, 3)]
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += reward(transition(grid, a, grid, b))
    assert np.isclose(total, (1 - 1 / 8) - (1 - 4 / 8))  # potential(end) - potential(start)
    
def test_a_bump_costs_exactly_the_penalty():
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0, w_bump=0.2)
    grid = with_goal(5, 5, (2, 2))
    t = transition(grid, (0, 0), grid, (0, 0))               # did not move
    assert np.isclose(reward(t), -0.2)


def test_a_real_move_has_no_bump_penalty():
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0, w_bump=0.2)
    grid = with_goal(5, 5, (2, 2))
    t = transition(grid, (0, 0), grid, (0, 1))
    assert np.isclose(reward(t), 0.0)


def test_bump_penalty_is_off_by_default():
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0, w_bump=0.0)
    grid = with_goal(5, 5, (2, 2))
    t = transition(grid, (0, 0), grid, (0, 0))
    assert np.isclose(reward(t), 0.0)


def test_being_caught_in_place_is_not_a_bump():
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0, w_bump=0.2)
    grid = with_goal(5, 5, (2, 2))
    t = transition(grid, (0, 0), grid, (0, 0), status="Caught")
    assert np.isclose(reward(t), 0.0)


def run():
    test_zero_weights_equal_the_sparse_reward()
    test_step_toward_the_goal_pays_and_step_away_costs()
    test_discovering_the_goal_is_a_positive_jump()
    test_revealing_cells_pays_the_explored_fraction()
    test_gamma_discounts_the_next_potential_only()
    test_reaching_the_goal_pays_bonus_plus_last_progress()
    test_shaping_telescopes_along_a_path()
    test_a_bump_costs_exactly_the_penalty()
    test_a_real_move_has_no_bump_penalty()
    test_bump_penalty_is_off_by_default()
    test_being_caught_in_place_is_not_a_bump()
    print("shaped reward: all checks passed")


if __name__ == "__main__":
    run()