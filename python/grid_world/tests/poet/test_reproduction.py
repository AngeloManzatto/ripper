"""
Created on Wed Nov  4 16:05:08 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Tests of poet/reproduction.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import copy

import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import reproduction


from poet.entry import Entry
from poet.reproduction import reproduce, entry_distance


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


@contextmanager
def patched(module, **replacements):
    """Replace attributes of `module` and ALWAYS put the originals back, even when a test fails."""
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


class FakeLearner:
    def agent(self):
        return self

    def train(self, layout, opponent, rng):
        pass

    def clone(self):
        return copy.deepcopy(self)


def entry(layout):
    """An entry whose layout is a plain string, so a distance is easy to write."""
    return Entry(layout, {"player": FakeLearner(), "enemy_0": FakeLearner()})


def string_distance(a, b):
    """Number of different characters (the strings have the same length)."""
    return sum(x != y for x, y in zip(a, b))


@contextmanager
def stand_ins(eligible=(), accepts=None):
    """is_eligible, make_child and the layout distance replaced.

    eligible : layouts of the entries that are eligible.
    accepts  : function(parent, mutation_seed) -> the layout of the child, or None when rejected.
               By default every attempt makes a child that differs from its parent in the last
               character, a different one for every seed (seeds that are multiples of 26 would
               copy the parent).
    """
    calls = {"eligible": [], "child": []}

    def fake_eligible(candidate, n_games, first_seed, **limits):
        calls["eligible"].append((candidate, n_games, first_seed, limits))
        return candidate.layout in eligible, {}

    def fake_child(parent, mutation_config, mutation_seed, n_games, first_seed, **limits):
        calls["child"].append((parent, mutation_config, mutation_seed, n_games, first_seed, limits))
        layout = accepts(parent, mutation_seed) if accepts else parent.layout[:-1] + chr(97 + mutation_seed % 26)
        return (entry(layout) if layout is not None else None), {}

    with patched(reproduction, is_eligible=fake_eligible, make_child=fake_child,
                 layout_distance=string_distance):
        yield calls


def run(pool, **overrides):
    arguments = dict(mutation_config={"m": 1}, mutation_seed=100, n_children=2, max_new=10,
                     n_games=50, first_seed=230_000)
    arguments.update(overrides)
    return reproduce(pool, **arguments)


###############################################################################
# Who reproduces
###############################################################################

def test_every_entry_of_the_pool_is_checked_and_only_the_eligible_reproduce():
    pool = [entry("aaaa"), entry("bbbb"), entry("cccc")]
    with stand_ins(eligible=("aaaa", "cccc")) as calls:
        run(pool, n_children=3)
    assert [c[0] for c in calls["eligible"]] == pool
    parents = [c[0] for c in calls["child"]]
    assert parents == [pool[0]] * 3 + [pool[2]] * 3             # parent after parent, n_children each


def test_the_eligibility_and_the_children_are_measured_on_the_same_games_and_limits():
    pool = [entry("aaaa")]
    with stand_ins(eligible=("aaaa",)) as calls:
        run(pool, n_games=40, first_seed=7, n_children=1, low=0.1, timeout_cap=0.5)
    assert calls["eligible"][0][1:] == (40, 7, {"low": 0.1, "timeout_cap": 0.5})
    _, config, _, n_games, first_seed, limits = calls["child"][0]
    assert (config, n_games, first_seed, limits) == ({"m": 1}, 40, 7, {"low": 0.1, "timeout_cap": 0.5})


def test_without_eligible_parents_nothing_is_tried_and_the_report_says_so():
    pool = [entry("aaaa"), entry("bbbb")]
    with stand_ins(eligible=()) as calls:
        children, report = run(pool)
    assert children == [] and calls["child"] == []
    assert report == {"eligible": 0, "attempts": 0, "accepted": 0, "picked": 0}


def test_an_empty_pool_has_no_children():
    with stand_ins() as calls:
        children, report = run([])
    assert children == [] and report["eligible"] == 0 and calls["child"] == []


###############################################################################
# The mutation seeds
###############################################################################

def test_the_mutation_seeds_start_at_the_given_one_and_never_repeat():
    pool = [entry("aaaa"), entry("bbbb")]
    with stand_ins(eligible=("aaaa", "bbbb")) as calls:
        run(pool, mutation_seed=100, n_children=3)
    assert [c[2] for c in calls["child"]] == [100, 101, 102, 103, 104, 105]


def test_the_same_arguments_give_the_same_children():
    pool = [entry("aaaa")]
    with stand_ins(eligible=("aaaa",)):
        one, _ = run(pool)
        two, _ = run(pool)
    assert [c.layout for c in one] == [c.layout for c in two]


def test_another_first_seed_gives_other_attempts():
    pool = [entry("aaaa")]
    with stand_ins(eligible=("aaaa",)):
        one, _ = run(pool, mutation_seed=1, n_children=1)
        two, _ = run(pool, mutation_seed=500, n_children=1)
    assert [c.layout for c in one] != [c.layout for c in two]


###############################################################################
# Accepted children and the choice
###############################################################################

def test_rejected_children_are_not_candidates_and_the_report_counts_everything():
    pool = [entry("aaaa")]
    accepts = lambda parent, seed: None if seed % 2 else "aaa" + chr(98 + (seed - 100) // 2)   # odd seeds are rejected
    with stand_ins(eligible=("aaaa",), accepts=accepts):
        children, report = run(pool, n_children=4, mutation_seed=100)
    assert report == {"eligible": 1, "attempts": 4, "accepted": 2, "picked": 2}
    assert sorted(c.layout for c in children) == ["aaab", "aaac"]


def test_the_most_novel_children_are_returned_first_and_at_most_max_new():
    pool = [entry("aaaa")]
    layouts = {100: "aaab", 101: "bbbb", 102: "aabb"}                                   # distances to the parent: 1, 4, 2
    with stand_ins(eligible=("aaaa",), accepts=lambda parent, seed: layouts[seed]):
        children, report = run(pool, n_children=3, mutation_seed=100, max_new=2)
    assert [c.layout for c in children] == ["bbbb", "aabb"]
    assert report["accepted"] == 3 and report["picked"] == 2


def test_a_child_that_copies_an_entry_of_the_pool_is_never_returned():
    pool = [entry("aaaa"), entry("bbbb")]
    with stand_ins(eligible=("aaaa",), accepts=lambda parent, seed: "bbbb"):             # copies the OTHER entry
        children, report = run(pool, n_children=2)
    assert children == [] and report["accepted"] == 2 and report["picked"] == 0


def test_siblings_that_are_close_to_each_other_do_not_all_get_in():
    pool = [entry("aaaa")]
    layouts = {100: "bbbb", 101: "bbbc", 102: "cccc"}                                    # 100 and 101 are siblings
    with stand_ins(eligible=("aaaa",), accepts=lambda parent, seed: layouts[seed]):
        children, _ = run(pool, n_children=3, mutation_seed=100, max_new=2)
    assert [c.layout for c in children] == ["bbbb", "cccc"]


def test_max_new_zero_returns_nothing():
    pool = [entry("aaaa")]
    with stand_ins(eligible=("aaaa",)):
        children, report = run(pool, max_new=0)
    assert children == [] and report["picked"] == 0 and report["accepted"] == 2


###############################################################################
# Arguments
###############################################################################

def test_n_children_must_be_at_least_one():
    with stand_ins():
        with raises(ValueError, match="n_children"):
            run([entry("aaaa")], n_children=0)


def test_the_pool_is_not_modified_and_the_children_are_new_entries():
    pool = [entry("aaaa"), entry("bbbb")]
    before = list(pool)
    with stand_ins(eligible=("aaaa",)):
        children, _ = run(pool)
    assert pool == before and [e.layout for e in pool] == ["aaaa", "bbbb"]
    assert all(not any(child is member for member in pool) for child in children)


def test_the_distance_between_entries_is_the_distance_between_their_layouts():
    with patched(reproduction, layout_distance=string_distance):
        assert entry_distance(entry("aaaa"), entry("aabb")) == 2
        assert entry_distance(entry("aaaa"), entry("aaaa")) == 0


###############################################################################
# The real engine (not run in my sandbox: it needs your engine)
###############################################################################

def test_real_reproduction_gives_new_entries_on_other_layouts():
    try:
        from engine import GenerationConfig, generate_valid_layout
        from agents.heuristic_agent.agent import HeuristicAgent
    except ImportError:
        raise unittest.SkipTest("the real engine is not available")

    class Fixed:
        def __init__(self, agent):
            self._agent = agent

        def agent(self):
            return self._agent

        def train(self, layout, opponent, rng):
            pass

        def clone(self):
            return Fixed(self._agent.clone())

    layout = generate_valid_layout(GenerationConfig(width=16, height=16, wall_density=0.1, num_enemies=1,
                                                    num_traps=0, min_player_goal_distance=10))
    mutation = {"n_wall_mutations": 1, "max_wall_density": 0.3, "max_enemies": 1,
                "enemy_add_weight": 0.0, "enemy_remove_weight": 0.0, "enemy_nothing_weight": 1.0,
                "max_traps": 3, "trap_add_weight": 0.0, "trap_remove_weight": 0.0,
                "trap_nothing_weight": 1.0, "n_repositions": 1, "min_player_goal_distance": 10}
    pool = [Entry(layout, {"player": Fixed(HeuristicAgent()), "enemy_0": Fixed(HeuristicAgent())})]

    # limits that accept everything: the test is about the round, not about the criterion
    children, report = reproduce(pool, mutation, mutation_seed=0, n_children=4, max_new=2, n_games=10,
                                 first_seed=230_000, low=0.0, high=1.0, timeout_cap=1.0)
    assert len(children) == 2 and report == {"eligible": 1, "attempts": 4, "accepted": 4, "picked": 2}
    assert all(entry_distance(child, pool[0]) > 0 for child in children)


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