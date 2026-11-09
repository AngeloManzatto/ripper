#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:20:38 2026

@author: root
"""

"""Tests of poet/loop.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import copy
import random
import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import loop

from poet.entry import Entry
from poet.loop import run_poet


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


def entry(name):
    return Entry(name, {"player": FakeLearner(), "enemy_0": FakeLearner()})


def nothing_adopted(pool):
    """The report transfer_round gives when no entry adopts anything."""
    row = {"own": 0.0, "best": 0.0, "from": 0, "adopted": False}
    return [{"player": dict(row), "enemy_0": dict(row)} for _ in pool]


@contextmanager
def stand_ins(script=None, transfers=None):
    """reproduce, train_entry and transfer_round replaced by recorders.

    script    : list with one (children_layouts, report_numbers) for every generation. By default no
                children and 0 attempts.
    transfers : list with the reports transfer_round gives, one for every round that happens (by
                default nobody adopts anything).
    Everything that happens goes into calls["log"], in order, so the order of the loop can be tested.
    """
    calls = {"reproduce": [], "train": [], "transfer": [], "log": []}
    script = list(script or [])
    transfers = list(transfers or [])

    def fake_reproduce(pool, mutation_config, mutation_seed, n_children, max_new, n_games, first_seed, **limits):
        calls["reproduce"].append(dict(pool=list(pool), mutation_config=mutation_config, mutation_seed=mutation_seed,
                                       n_children=n_children, max_new=max_new, n_games=n_games,
                                       first_seed=first_seed, limits=limits))
        calls["log"].append("reproduce")
        layouts, numbers = script.pop(0) if script else ([], {})
        report = {"eligible": 1, "attempts": 0, "accepted": len(layouts), "picked": len(layouts)}
        report.update(numbers)
        return [entry(layout) for layout in layouts], report

    def fake_train(e, rng):
        calls["train"].append((e, rng))
        calls["log"].append(f"train {e.layout}")

    def fake_transfer(pool, n_games, first_seed, margin):
        calls["transfer"].append(dict(pool=list(pool), n_games=n_games, first_seed=first_seed, margin=margin))
        calls["log"].append("transfer")
        return transfers.pop(0) if transfers else nothing_adopted(pool)

    with patched(loop, reproduce=fake_reproduce, train_entry=fake_train, transfer_round=fake_transfer):
        yield calls


def run(pool, **overrides):
    arguments = dict(mutation_config={"m": 1}, n_generations=1, max_pool=10, n_children=3, max_new=2,
                     n_games=50, first_seed=230_000)
    arguments.update(overrides)
    return run_poet(pool, **arguments)


###############################################################################
# One generation
###############################################################################

def test_a_generation_is_reproduce_then_admit_then_train_everything_in_the_pool_order():
    pool = [entry("a"), entry("b")]
    with stand_ins([(["c", "d"], {})]) as calls:
        run(pool)
    assert calls["log"] == ["reproduce", "train a", "train b", "train c", "train d"]   # the new entries train too


def test_the_children_join_the_pool_at_the_end():
    pool = [entry("a")]
    with stand_ins([(["b", "c"], {})]):
        final, _ = run(pool)
    assert [e.layout for e in final] == ["a", "b", "c"]


def test_reproduce_gets_the_settings_of_the_run():
    pool = [entry("a")]
    with stand_ins() as calls:
        run(pool, mutation_config={"walls": 2}, n_children=7, max_new=4, n_games=33, first_seed=5,
            mutation_seed=1000, low=0.1, timeout_cap=0.6)
    call = calls["reproduce"][0]
    assert call["mutation_config"] == {"walls": 2}
    assert (call["n_children"], call["max_new"], call["n_games"], call["first_seed"], call["mutation_seed"]) \
        == (7, 4, 33, 5, 1000)
    assert call["limits"] == {"low": 0.1, "timeout_cap": 0.6}
    assert call["pool"] == pool


###############################################################################
# The pool over the generations
###############################################################################

def test_every_generation_reproduces_from_the_pool_the_last_one_left():
    pool = [entry("a")]
    with stand_ins([(["b"], {}), (["c"], {}), ([], {})]) as calls:
        run(pool, n_generations=3)
    assert [[e.layout for e in call["pool"]] for call in calls["reproduce"]] == [["a"], ["a", "b"], ["a", "b", "c"]]


def test_the_oldest_entries_leave_when_the_pool_is_full():
    pool = [entry("a"), entry("b")]
    with stand_ins([(["c", "d"], {}), (["e"], {})]):
        final, history = run(pool, n_generations=2, max_pool=3)
    assert [e.layout for e in final] == ["c", "d", "e"]
    assert [h["dropped"] for h in history] == [1, 1]
    assert [h["pool"] for h in history] == [3, 3]


def test_the_starting_list_is_not_modified():
    pool = [entry("a")]
    with stand_ins([(["b"], {})]):
        run(pool)
    assert [e.layout for e in pool] == ["a"]


def test_zero_generations_change_nothing():
    pool = [entry("a")]
    with stand_ins() as calls:
        final, history = run(pool, n_generations=0)
    assert history == [] and calls["log"] == []
    assert final == pool and final is not pool


###############################################################################
# Seeds
###############################################################################

def test_the_mutation_seed_moves_forward_by_the_attempts_made():
    pool = [entry("a")]
    with stand_ins([([], {"attempts": 4}), ([], {"attempts": 6}), ([], {"attempts": 0})]) as calls:
        run(pool, n_generations=3, mutation_seed=100)
    assert [c["mutation_seed"] for c in calls["reproduce"]] == [100, 104, 110]


def test_every_generation_trains_with_its_own_generator_shared_by_its_entries():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        run(pool, n_generations=2, train_seed=50)
    (_, rng_0a), (_, rng_0b), (_, rng_1a), (_, rng_1b) = calls["train"]
    assert rng_0a is rng_0b and rng_1a is rng_1b and rng_0a is not rng_1a
    assert rng_0a.random() == random.Random(50).random()           # generation 0 -> seed 50
    assert rng_1a.random() == random.Random(51).random()           # generation 1 -> seed 51


###############################################################################
# The report
###############################################################################

def test_the_history_has_one_report_per_generation():
    pool = [entry("a")]
    numbers = {"eligible": 1, "attempts": 4, "accepted": 3, "picked": 2}
    with stand_ins([(["b", "c"], numbers), ([], {"eligible": 0})]):
        _, history = run(pool, n_generations=2)
    assert history[0] == {"generation": 0, "eligible": 1, "attempts": 4, "accepted": 3, "picked": 2,
                          "dropped": 0, "pool": 3, "transfers": []}
    assert history[1]["generation"] == 1 and history[1]["eligible"] == 0 and history[1]["pool"] == 3


def test_without_eligible_parents_the_pool_still_trains():
    pool = [entry("a")]
    with stand_ins([([], {"eligible": 0})]) as calls:
        final, history = run(pool)
    assert history[0]["eligible"] == 0 and calls["log"] == ["reproduce", "train a"]


def test_on_generation_is_called_at_the_end_of_every_generation_after_training():
    pool = [entry("a")]
    seen = []
    with stand_ins([(["b"], {}), ([], {})]) as calls:
        def on_generation(generation, current_pool, report):
            calls["log"].append(f"callback {generation}")
            seen.append((generation, [e.layout for e in current_pool], report["generation"]))
        run(pool, n_generations=2, on_generation=on_generation)
    assert seen == [(0, ["a", "b"], 0), (1, ["a", "b"], 1)]
    assert calls["log"] == ["reproduce", "train a", "train b", "callback 0",
                            "reproduce", "train a", "train b", "callback 1"]


def test_an_error_in_on_generation_stops_the_run():
    pool = [entry("a")]

    def stop(generation, current_pool, report):
        raise KeyboardInterrupt

    with stand_ins() as calls:
        try:
            run(pool, n_generations=5, on_generation=stop)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the run did not stop")
    assert len(calls["reproduce"]) == 1


###############################################################################
# Transfer
###############################################################################

def test_without_n_transfer_there_is_never_a_transfer():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        _, history = run(pool, n_generations=3)
    assert calls["transfer"] == []
    assert all(h["transfers"] == [] for h in history)


def test_a_transfer_round_happens_after_every_n_transfer_generations():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        run(pool, n_generations=6, n_transfer=3)
    assert [step for step in calls["log"] if step == "transfer" or step == "reproduce"] == \
        ["reproduce", "reproduce", "reproduce", "transfer", "reproduce", "reproduce", "reproduce", "transfer"]


def test_every_generation_when_n_transfer_is_one():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        run(pool, n_generations=3, n_transfer=1)
    assert len(calls["transfer"]) == 3


def test_transfer_comes_after_training_and_before_on_generation():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        run(pool, n_transfer=1, on_generation=lambda g, p, r: calls["log"].append("callback"))
    assert calls["log"] == ["reproduce", "train a", "train b", "transfer", "callback"]


def test_transfer_gets_the_pool_after_admitting_and_the_settings_of_the_run():
    pool = [entry("a")]
    with stand_ins([(["b"], {})]) as calls:
        run(pool, n_transfer=1, transfer_seed=900, n_transfer_games=40, transfer_margin=0.1)
    call = calls["transfer"][0]
    assert [e.layout for e in call["pool"]] == ["a", "b"]               # the new entry takes part
    assert (call["n_games"], call["first_seed"], call["margin"]) == (40, 900, 0.1)


def test_the_transfer_games_move_forward_with_the_generation():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        run(pool, n_generations=4, n_transfer=2, transfer_seed=1000, n_transfer_games=30)
    # rounds after generations 1 and 3
    assert [c["first_seed"] for c in calls["transfer"]] == [1000 + 1 * 30, 1000 + 3 * 30]


def test_a_pool_of_one_entry_never_transfers():
    pool = [entry("a")]
    with stand_ins() as calls:
        _, history = run(pool, n_generations=3, n_transfer=1)
    assert calls["transfer"] == [] and all(h["transfers"] == [] for h in history)


def test_the_first_round_with_more_than_one_entry_is_when_the_children_arrive():
    pool = [entry("a")]
    with stand_ins([(["b"], {}), ([], {})]) as calls:
        _, history = run(pool, n_generations=2, n_transfer=1)
    assert len(calls["transfer"]) == 2                                  # generation 0 already has 2 entries after admit


def test_the_report_lists_the_adoptions_with_positions_in_the_pool():
    pool = [entry("a"), entry("b"), entry("c")]

    def row(adopted, source=0):
        return {"own": 0.0, "best": 0.0, "from": source, "adopted": adopted}

    # entry 0: adopts the enemy from its candidate 0 (= entry 1)
    # entry 1: adopts the player from its candidate 1 (its candidates are entries 0 and 2, so = entry 2)
    # entry 2: nothing
    reports = [
        {"player": row(False), "enemy_0": row(True, 0)},
        {"player": row(True, 1), "enemy_0": row(False)},
        {"player": row(False), "enemy_0": row(False)},
    ]
    with stand_ins(transfers=[reports]):
        _, history = run(pool, n_transfer=1)
    assert history[0]["transfers"] == [(0, "enemy_0", 1), (1, "player", 2)]


def test_adoptions_counts_the_candidates_without_the_entry_itself():
    pool = [entry("a"), entry("b"), entry("c")]
    row = lambda adopted, source: {"own": 0.0, "best": 0.0, "from": source, "adopted": adopted}
    nobody = {"player": row(False, 0), "enemy_0": row(False, 0)}

    # the last entry: its candidates are entries 0 and 1; candidate 1 is entry 1
    reports = [nobody, nobody, {"player": row(True, 1), "enemy_0": row(True, 0)}]
    assert loop.adoptions(pool, reports) == [(2, "player", 1), (2, "enemy_0", 0)]


def test_a_generation_without_a_round_reports_no_transfers():
    pool = [entry("a"), entry("b")]
    with stand_ins() as calls:
        _, history = run(pool, n_generations=2, n_transfer=2)
    assert [h["transfers"] for h in history] == [[], []]
    assert len(calls["transfer"]) == 1


###############################################################################
# Arguments
###############################################################################

def test_bad_arguments_are_errors_before_anything_runs():
    with stand_ins() as calls:
        with raises(ValueError, match="at least one entry"):
            run([])
        with raises(ValueError, match="n_generations"):
            run([entry("a")], n_generations=-1)
        with raises(ValueError, match="max_pool"):
            run([entry("a"), entry("b")], max_pool=1)
        with raises(ValueError, match="n_transfer"):
            run([entry("a")], n_transfer=0)
        with raises(ValueError, match="n_transfer_games"):
            run([entry("a")], n_transfer=1, n_transfer_games=0)
        with raises(ValueError, match="transfer_margin"):
            run([entry("a")], n_transfer=1, transfer_margin=-0.1)
    assert calls["log"] == []


###############################################################################
# The real engine (not run in my sandbox: it needs your engine)
###############################################################################

def test_real_run_grows_the_pool_up_to_its_limit():
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
    start = [Entry(layout, {"player": Fixed(HeuristicAgent()), "enemy_0": Fixed(HeuristicAgent())})]

    # limits that accept everything: the test is about the loop, not about the criterion
    pool, history = run_poet(start, mutation, n_generations=3, max_pool=4, n_children=3, max_new=2, n_games=10,
                             first_seed=230_000, low=0.0, high=1.0, timeout_cap=1.0)
    assert len(history) == 3 and [h["generation"] for h in history] == [0, 1, 2]
    assert 1 < len(pool) <= 4 and all(h["pool"] <= 4 for h in history)


def test_real_run_with_transfer_every_generation():
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

    def start():
        return [Entry(layout, {"player": Fixed(HeuristicAgent()), "enemy_0": Fixed(HeuristicAgent())})]

    arguments = dict(mutation_config=mutation, n_generations=3, max_pool=4, n_children=3, max_new=2, n_games=10,
                     first_seed=230_000, low=0.0, high=1.0, timeout_cap=1.0,
                     n_transfer=1, n_transfer_games=10, transfer_margin=0.1)
    pool, history = run_poet(start(), **arguments)
    again_pool, again = run_poet(start(), **arguments)

    assert len(history) == 3 and all("transfers" in h for h in history)
    assert history == again                       # the same run twice gives the same report, transfers included


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