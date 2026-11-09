"""
Created on Fri Oct 16 09:55:03 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from types import SimpleNamespace

import numpy as np

from agents.features.temporal import (
    N_MEMORY, MEMORY_NAMES, recent_visit_flags, memory_flags,
)


def obs(position):
    return SimpleNamespace(position=position)


def history_of(*positions):
    return [obs(p) for p in positions]

###############################################################################
# Names
###############################################################################

def test_names_and_width():
    assert N_MEMORY == 4
    assert MEMORY_NAMES == ("visited_up", "visited_down", "visited_left", "visited_right")

###############################################################################
# recent_visit_flags
###############################################################################

def test_no_recent_positions_gives_zeros():
    assert recent_visit_flags((3, 3), []) == [0.0, 0.0, 0.0, 0.0]


def test_each_neighbor_sets_exactly_its_own_slot():
    # order is up, down, left, right
    assert recent_visit_flags((3, 3), [(2, 3)]) == [1.0, 0.0, 0.0, 0.0]
    assert recent_visit_flags((3, 3), [(4, 3)]) == [0.0, 1.0, 0.0, 0.0]
    assert recent_visit_flags((3, 3), [(3, 2)]) == [0.0, 0.0, 1.0, 0.0]
    assert recent_visit_flags((3, 3), [(3, 4)]) == [0.0, 0.0, 0.0, 1.0]


def test_several_neighbors_at_once():
    assert recent_visit_flags((3, 3), [(2, 3), (3, 4)]) == [1.0, 0.0, 0.0, 1.0]
    assert recent_visit_flags((3, 3), [(2, 3), (4, 3), (3, 2), (3, 4)]) == [1.0] * 4


def test_only_the_four_neighbors_count():
    # diagonal, two cells away, and the current cell itself
    assert recent_visit_flags((3, 3), [(2, 2), (1, 3), (3, 5), (3, 3)]) == [0.0] * 4


def test_positions_may_be_lists_and_repeated():
    assert recent_visit_flags((3, 3), [[2, 3], (2, 3), [2, 3]]) == [1.0, 0.0, 0.0, 0.0]


def test_values_are_floats():
    flags = recent_visit_flags((3, 3), [(2, 3)])
    assert len(flags) == 4 and all(isinstance(v, float) for v in flags)

###############################################################################
# memory_flags (history of observations)
###############################################################################

def test_one_observation_history_gives_zeros():
    assert memory_flags(history_of((3, 3))) == [0.0, 0.0, 0.0, 0.0]


def test_empty_history_is_rejected():
    try:
        memory_flags([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_revisiting_a_neighbor_sets_that_direction():
    # the agent walked (3,2) -> (3,3) -> (3,4) -> (3,3): now standing on (3,3)
    flags = memory_flags(history_of((3, 2), (3, 3), (3, 4), (3, 3)))
    assert flags == [0.0, 0.0, 1.0, 1.0]            # left and right were visited


def test_the_current_observation_is_the_last_one():
    # standing on (5,5); the flags are about (5,5)'s neighbors, not the first entry's
    flags = memory_flags(history_of((0, 0), (5, 4), (5, 5)))
    assert flags == [0.0, 0.0, 1.0, 0.0]


def test_dropping_the_oldest_observation_forgets_it():
    full = history_of((2, 3), (3, 4), (3, 3))        # up and right visited
    assert memory_flags(full) == [1.0, 0.0, 0.0, 1.0]
    assert memory_flags(full[1:]) == [0.0, 0.0, 0.0, 1.0]     # (2,3) forgotten


def test_next_state_window_as_the_agent_builds_it():
    # (history + [next_obs]) trimmed to the history length, as the old agent did
    maxlen = 3
    history = history_of((3, 1), (3, 2), (3, 3))
    next_obs = obs((3, 4))
    window = (history + [next_obs])[-maxlen:]
    assert [o.position for o in window] == [(3, 2), (3, 3), (3, 4)]
    assert memory_flags(window) == [0.0, 0.0, 1.0, 0.0]       # came from the left


def test_random_histories_match_recent_visit_flags():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        positions = [(int(r), int(c)) for r, c in rng.integers(0, 5, size=(n, 2))]
        expected = recent_visit_flags(positions[-1], positions)
        assert memory_flags(history_of(*positions)) == expected


def test_history_is_not_modified():
    history = history_of((1, 1), (1, 2))
    before = [o.position for o in history]
    memory_flags(history)
    assert [o.position for o in history] == before

###############################################################################
# Equivalence with the old module (skipped once core/memory.py is gone)
###############################################################################

def test_matches_old_core_memory():
    try:
        from core import memory as old
    except ImportError:
        print("  (old core.memory not found: equivalence check skipped)")
        return
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(0, 9))
        positions = [(int(r), int(c)) for r, c in rng.integers(0, 5, size=(n, 2))]
        pos = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
        assert old.recent_visit_flags(pos, positions) == recent_visit_flags(pos, positions)

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"temporal: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()