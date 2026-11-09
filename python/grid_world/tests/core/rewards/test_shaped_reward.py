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
    
    
# 

def revisit_reward(window=8):
    return ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0,
                            w_bump=0.0, w_revisit=0.2, window=window)


def go(reward, grid, a, b):
    return reward(transition(grid, a, grid, b))


def test_going_back_is_penalized():
    reward, grid = revisit_reward(), with_goal(5, 5, (4, 4))
    assert np.isclose(go(reward, grid, (0, 0), (0, 1)), 0.0)
    assert np.isclose(go(reward, grid, (0, 1), (0, 0)), -0.2)


def test_new_cells_are_not_penalized():
    reward, grid = revisit_reward(), with_goal(5, 5, (4, 4))
    assert np.isclose(go(reward, grid, (0, 0), (0, 1)), 0.0)
    assert np.isclose(go(reward, grid, (0, 1), (0, 2)), 0.0)


def test_a_bump_is_not_a_revisit():
    reward, grid = revisit_reward(), with_goal(5, 5, (4, 4))
    go(reward, grid, (0, 0), (0, 1))
    assert np.isclose(go(reward, grid, (0, 1), (0, 1)), 0.0)


def test_reset_forgets_the_episode():
    reward, grid = revisit_reward(), with_goal(5, 5, (4, 4))
    go(reward, grid, (0, 0), (0, 1))
    reward.reset()
    assert np.isclose(go(reward, grid, (0, 1), (0, 0)), 0.0)


def test_old_cells_are_forgotten_after_the_window():
    path = [(0, 0), (0, 1), (0, 2), (0, 3)]
    grid = with_goal(5, 5, (4, 4))

    short = revisit_reward(window=3)                  # remembers 2 cells
    for a, b in zip(path, path[1:]):
        go(short, grid, a, b)
    assert np.isclose(go(short, grid, (0, 3), (0, 0)), 0.0)

    long = revisit_reward(window=8)
    for a, b in zip(path, path[1:]):
        go(long, grid, a, b)
    assert np.isclose(go(long, grid, (0, 3), (0, 0)), -0.2)


def test_revisit_penalty_is_off_by_default():
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0, w_bump=0.0)
    grid = with_goal(5, 5, (4, 4))
    go(reward, grid, (0, 0), (0, 1))
    assert np.isclose(go(reward, grid, (0, 1), (0, 0)), 0.0)


def test_window_must_be_at_least_one():
    for bad in (0, -3):
        try:
            ShapedGoalReward(window=bad)
        except ValueError:
            continue
        raise AssertionError(f"window={bad} should have raised")


def test_window_one_never_penalizes_revisits():
    reward = revisit_reward(window=1)
    grid = with_goal(5, 5, (4, 4))
    go(reward, grid, (0, 0), (0, 1))
    assert np.isclose(go(reward, grid, (0, 1), (0, 0)), 0.0)
    
# 

def danger_reward(**kw):
    return ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0,
                            w_bump=0.0, w_revisit=0.0, w_danger=0.5, **kw)


def test_being_caught_costs_the_danger_penalty():
    grid = with_goal(5, 5, (4, 4))
    t = transition(grid, (0, 0), grid, (0, 1), status="Caught", terminated=True)
    assert np.isclose(danger_reward()(t), -0.5)


def test_being_trapped_costs_the_danger_penalty():
    grid = with_goal(5, 5, (4, 4))
    t = transition(grid, (0, 0), grid, (0, 1), status="Trapped", terminated=True)
    assert np.isclose(danger_reward()(t), -0.5)


def test_caught_in_place_pays_danger_not_bump():
    grid = with_goal(5, 5, (4, 4))
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0,
                              w_bump=0.1, w_danger=0.5)
    t = transition(grid, (0, 0), grid, (0, 0), status="Caught", terminated=True)
    assert np.isclose(reward(t), -0.5)


def test_goal_and_timeout_pay_no_danger():
    grid = with_goal(5, 5, (4, 4))
    won = transition(grid, (0, 0), grid, (0, 1), status="GoalReached", terminated=True)
    assert np.isclose(danger_reward()(won), 1.0)
    timeout = transition(grid, (0, 0), grid, (0, 1), status="Playing", terminated=False)
    assert np.isclose(danger_reward()(timeout), 0.0)


def test_danger_penalty_is_off_by_default():
    grid = with_goal(5, 5, (4, 4))
    reward = ShapedGoalReward(gamma=1.0, w_goal=0.0, w_explore=0.0)
    t = transition(grid, (0, 0), grid, (0, 1), status="Caught", terminated=True)
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
    test_going_back_is_penalized()
    test_new_cells_are_not_penalized()
    test_a_bump_is_not_a_revisit()
    test_reset_forgets_the_episode()
    test_old_cells_are_forgotten_after_the_window()
    test_revisit_penalty_is_off_by_default()
    test_window_must_be_at_least_one()
    test_window_one_never_penalizes_revisits()
    test_being_caught_costs_the_danger_penalty()
    test_being_trapped_costs_the_danger_penalty()
    test_caught_in_place_pays_danger_not_bump()
    test_goal_and_timeout_pay_no_danger()
    test_danger_penalty_is_off_by_default()
    print("shaped reward: all checks passed")


if __name__ == "__main__":
    run()