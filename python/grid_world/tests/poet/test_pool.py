"""
Created on Wed Nov  4 08:00:12 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Tests of poet/pool.py. Entries are plain numbers and the distance is |a - b|: the pool does not care.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import importlib
import math
import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import pool as pool_module

importlib.reload(pool_module)      # pick up edits without restarting the console
from poet.pool import admit, select_novel


@contextmanager
def raises(exception, match=None):
    """The block must raise `exception` (with `match` inside the message when given)."""
    try:
        yield
    except exception as error:
        if match is not None and match not in str(error):
            raise AssertionError(f"{exception.__name__} message {str(error)!r} does not contain {match!r}")
    else:
        raise AssertionError(f"{exception.__name__} was not raised")


def gap(a, b):
    return abs(a - b)


###############################################################################
# select_novel
###############################################################################

def test_the_farthest_candidate_comes_first_and_copies_never_get_in():
    # 0 is a copy of the pool's entry (distance 0)
    assert select_novel([1, 10, 0], [0], n_keep=3, distance=gap) == [10, 1]


def test_two_similar_siblings_do_not_both_get_in():
    # 10 first. Then 9 is only 1 away from it, while 5 is 5 away from both the pool and 10.
    assert select_novel([10, 9, 5], [0], n_keep=2, distance=gap) == [10, 5]


def test_novelty_is_the_distance_to_the_NEAREST_entry():
    # 7 is far from 0 but sits right next to the pool's second entry, 8
    assert select_novel([7, 3], [0, 8], n_keep=1, distance=gap) == [3]


def test_ties_go_to_the_first_candidate():
    assert select_novel([5, -5, 5], [0], n_keep=1, distance=gap) == [5]
    assert select_novel([-5, 5], [0], n_keep=1, distance=gap) == [-5]


def test_with_an_empty_pool_the_first_is_picked_and_the_rest_differ_from_it():
    assert select_novel([3, 4, 9], [], n_keep=2, distance=gap) == [3, 9]
    assert select_novel([3], [], n_keep=5, distance=gap) == [3]


def test_it_stops_when_only_copies_are_left():
    assert select_novel([0, 0, 6, 6], [0], n_keep=4, distance=gap) == [6]


def test_nothing_to_pick():
    assert select_novel([], [0], n_keep=2, distance=gap) == []
    assert select_novel([5], [0], n_keep=0, distance=gap) == []


def test_it_returns_the_same_objects_and_leaves_the_arguments_alone():
    a, b = {"id": "a"}, {"id": "b"}
    candidates, pool = [a, b], [{"id": "p"}]
    dist = lambda x, y: {"a": 1, "b": 2, "p": 0}[x["id"]] if y["id"] == "p" else 1
    picked = select_novel(candidates, pool, 2, dist)
    assert picked[0] is b and picked[1] is a
    assert candidates == [a, b] and pool == [{"id": "p"}]


def test_it_does_not_recompute_distances():
    calls = []

    def counting(a, b):
        calls.append((a, b))
        return abs(a - b)

    select_novel([10, 9, 8, 7, 6], [0, 1], n_keep=2, distance=counting)
    # 5 candidates x 2 pool entries, then 4 + 3 remaining candidates against each pick
    assert len(calls) == 5 * 2 + 4 + 3


def test_a_distance_that_is_not_a_distance_raises():
    for bad in (-1, math.nan):
        with raises(ValueError, match="distance must be"):
            select_novel([1, 2], [0], 1, distance=lambda a, b: bad)
        # empty pool: no distance is asked until a candidate is picked, then 2 is compared with the pick 1
        with raises(ValueError, match="distance must be"):
            select_novel([1, 2], [], 2, distance=lambda a, b: bad)


def test_a_negative_n_keep_raises():
    with raises(ValueError, match="n_keep"):
        select_novel([1], [0], -1, gap)


###############################################################################
# admit
###############################################################################

def test_the_oldest_leave_first():
    assert admit([1, 2, 3], [4], 3) == [2, 3, 4]
    assert admit([1, 2, 3], [4, 5], 4) == [2, 3, 4, 5]
    assert admit([1, 2, 3], [], 5) == [1, 2, 3]
    assert admit([], [1, 2, 3], 2) == [2, 3]


def test_admit_does_not_modify_its_arguments():
    pool, children = [1, 2, 3], [4]
    admit(pool, children, 2)
    assert pool == [1, 2, 3] and children == [4]


def test_a_pool_needs_room_for_at_least_one_entry():
    with raises(ValueError, match="max_size"):
        admit([1], [], 0)


###############################################################################
# Runner
###############################################################################

def run_all():
    # Only the tests defined in THIS file: a console that stays open still holds the tests of earlier runs
    tests = [(name, test) for name, test in globals().items()
             if name.startswith("test_") and callable(test)]
    passed, skipped, failed = 0, 0, []

    for name, test in tests:
        try:
            test()
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"skip  {name}  ({reason})")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")

    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed" + (f": {failed}" if failed else ""))

    try:
        get_ipython()                    # inside Spyder / IPython: just report
    except NameError:
        if failed:
            sys.exit(1)                  # from a plain terminal: a failing exit code


if __name__ == "__main__":
    run_all()