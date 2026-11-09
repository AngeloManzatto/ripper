#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:15:33 2026

@author: root
"""

"""
Tests for agents/features/frame.py: the per-frame feature vector.

Run with pytest, or run this file directly to get a single pass/fail line.
No engine is needed: a feature function only reads .grid, .position and .kind,
so a small stand-in object is enough.
"""

###############################################################################
# libraries
###############################################################################

import copy
from types import SimpleNamespace

import numpy as np

from agents.features.grid import UNKNOWN, FREE, WALL, PLAYER, ENEMY, TRAP, GOAL
from agents.features.frame import (
    FEATURE_NAMES, N_FEATURES, KIND_PLAYER, KIND_ENEMY, hostile_code, features,
)

###############################################################################
# Fixture: a 4x4 map with every kind of cell the features look at
###############################################################################
#
#        col0   col1   col2   col3
# row0    .      #      .      ?
# row1    .      P      E      ?
# row2    T      .      .      ?
# row3    .      G      .      ?

GRID = [
    [FREE, WALL,   FREE,  UNKNOWN],
    [FREE, PLAYER, ENEMY, UNKNOWN],
    [TRAP, FREE,   FREE,  UNKNOWN],
    [FREE, GOAL,   FREE,  UNKNOWN],
]


def obs(grid, position, kind=KIND_PLAYER):
    return SimpleNamespace(grid=grid, position=position, kind=kind)


def as_dict(vector):
    return dict(zip(FEATURE_NAMES, vector.tolist()))


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol

###############################################################################
# Names and layout
###############################################################################

def test_feature_names_layout():
    assert N_FEATURES == len(FEATURE_NAMES) == 25
    assert len(set(FEATURE_NAMES)) == 25                       # no duplicates
    assert FEATURE_NAMES[:3] == ["up_wall", "up_trap", "up_hostile"]
    assert FEATURE_NAMES[12:15] == ["goal_found", "goal_drow", "goal_dcol"]
    assert FEATURE_NAMES[15:18] == ["hostile_found", "hostile_drow", "hostile_dcol"]
    assert FEATURE_NAMES[18:21] == ["trap_found", "trap_drow", "trap_dcol"]
    assert FEATURE_NAMES[21:] == ["unknown_up", "unknown_down", "unknown_left", "unknown_right"]


def test_output_shape_and_dtype():
    out = features(obs(GRID, (1, 1)))
    assert out.shape == (N_FEATURES,)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()

###############################################################################
# Hostile code
###############################################################################

def test_hostile_code():
    assert hostile_code(KIND_PLAYER) == ENEMY
    assert hostile_code(KIND_ENEMY) == PLAYER


def test_hostile_code_rejects_unknown_kind():
    try:
        hostile_code("ghost")
    except ValueError:
        return
    raise AssertionError("expected ValueError")

###############################################################################
# Values on the fixture, checked by hand
###############################################################################

def test_player_view_values():
    f = as_dict(features(obs(GRID, (1, 1), KIND_PLAYER)))

    # neighbors of (1,1): up = wall, down = free, left = free, right = enemy
    assert f["up_wall"] == 1 and f["up_trap"] == 0 and f["up_hostile"] == 0
    assert f["down_wall"] == 0 and f["down_trap"] == 0 and f["down_hostile"] == 0
    assert f["left_wall"] == 0 and f["left_trap"] == 0 and f["left_hostile"] == 0
    assert f["right_wall"] == 0 and f["right_trap"] == 0 and f["right_hostile"] == 1

    # offsets are divided by max(rows, cols) - 1 = 3
    assert f["goal_found"] == 1 and close(f["goal_drow"], 2 / 3) and close(f["goal_dcol"], 0.0)
    assert f["hostile_found"] == 1 and close(f["hostile_drow"], 0.0) and close(f["hostile_dcol"], 1 / 3)
    assert f["trap_found"] == 1 and close(f["trap_drow"], 1 / 3) and close(f["trap_dcol"], -1 / 3)

    # unexplored fraction of each half; column 3 is the only unknown column
    assert close(f["unknown_up"], 1 / 4)        # row 0
    assert close(f["unknown_down"], 2 / 8)      # rows 2-3
    assert close(f["unknown_left"], 0.0)        # column 0
    assert close(f["unknown_right"], 4 / 8)     # columns 2-3


def test_enemy_view_treats_the_player_as_hostile():
    f = as_dict(features(obs(GRID, (1, 2), KIND_ENEMY)))

    # the player sits in the cell to the left of (1,2)
    assert f["left_hostile"] == 1
    assert f["right_hostile"] == 0 and f["up_hostile"] == 0 and f["down_hostile"] == 0
    assert f["hostile_found"] == 1 and close(f["hostile_drow"], 0.0) and close(f["hostile_dcol"], -1 / 3)


def test_missing_targets_give_zeros():
    grid = [[FREE, FREE], [FREE, PLAYER]]
    f = as_dict(features(obs(grid, (1, 1))))
    for name in ("goal", "hostile", "trap"):
        assert (f[f"{name}_found"], f[f"{name}_drow"], f[f"{name}_dcol"]) == (0.0, 0.0, 0.0)


def test_map_edge_counts_as_wall():
    f = as_dict(features(obs(GRID, (0, 0))))
    assert f["up_wall"] == 1 and f["left_wall"] == 1
    assert f["right_wall"] == 1 and f["down_wall"] == 0      # (0,1) is a wall, (1,0) is free

###############################################################################
# Properties on random maps
###############################################################################

def random_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        rows, cols = rng.integers(2, 9, size=2)
        grid = rng.integers(-1, 6, size=(rows, cols)).tolist()
        position = (int(rng.integers(rows)), int(rng.integers(cols)))
        grid[position[0]][position[1]] = PLAYER
        kind = KIND_PLAYER if rng.random() < 0.5 else KIND_ENEMY
        cases.append(obs(grid, position, kind))
    return cases


def test_random_maps_stay_finite_and_in_range():
    for o in random_cases(200):
        out = features(o)
        assert out.shape == (N_FEATURES,) and np.isfinite(out).all()
        f = as_dict(out)
        for name in ("goal", "hostile", "trap"):
            assert -1.0 <= f[f"{name}_drow"] <= 1.0
            assert -1.0 <= f[f"{name}_dcol"] <= 1.0
        for name in FEATURE_NAMES[21:]:
            assert 0.0 <= f[name] <= 1.0


def test_features_do_not_modify_the_observation():
    for o in random_cases(20, seed=1):
        before = copy.deepcopy(o.grid)
        features(o)
        assert o.grid == before

###############################################################################
# Equivalence with the old module (skipped once core/features.py is gone)
###############################################################################

def test_matches_old_core_features():
    try:
        from core import features as old
    except ImportError:
        print("  (old core.features not found: equivalence check skipped)")
        return
    assert old.FEATURE_NAMES == FEATURE_NAMES
    for o in random_cases(300, seed=2):
        assert np.array_equal(old.features(o), features(o))

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"frame: all {len(tests)} checks passed")

if __name__ == "__main__":
    run()