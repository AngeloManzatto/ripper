#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 06:34:18 2026

@author: root
"""

"""Tests of experiments/poet/learner_config.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import dataclasses

import sys
import traceback
import unittest
from contextlib import contextmanager


from experiments.poet.learner_config import TrainConfig, ChampionConfig, check_seed_ranges


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


def train(**overrides):
    values = dict(population_size=50, n_generations=5, n_games=10, n_valid_games=30)
    values.update(overrides)
    return TrainConfig(**values)


###############################################################################
# TrainConfig
###############################################################################

def test_a_train_config_keeps_its_values_and_has_the_planned_defaults():
    config = train()
    assert (config.population_size, config.n_generations, config.n_games, config.n_valid_games) == (50, 5, 10, 30)
    assert (config.train_seed, config.valid_seed) == (200_000, 210_000)
    assert (config.novelty_weight, config.k_neighbours, config.report_every) == (0.2, 10, 10)


def test_the_four_sizes_have_no_default():
    for name in ("population_size", "n_generations", "n_games", "n_valid_games"):
        values = dict(population_size=50, n_generations=5, n_games=10, n_valid_games=30)
        del values[name]
        with raises(TypeError):
            TrainConfig(**values)


def test_a_config_cannot_be_changed_after_it_is_made():
    config = train()
    with raises(dataclasses.FrozenInstanceError):
        config.n_games = 99
    assert config.n_games == 10


def test_sizes_must_be_whole_numbers_of_at_least_one():
    for name, bad in (("n_generations", 0), ("n_games", 0), ("n_valid_games", 0), ("k_neighbours", 0),
                      ("n_games", -3), ("n_games", 2.5), ("n_games", True), ("n_games", "10")):
        with raises(ValueError, match=name):
            train(**{name: bad})


def test_a_population_needs_at_least_two_genomes():
    with raises(ValueError, match="population_size"):
        train(population_size=1, k_neighbours=1)
    assert train(population_size=2, k_neighbours=1).population_size == 2


def test_seeds_cannot_be_negative():
    with raises(ValueError, match="train_seed"):
        train(train_seed=-1)
    with raises(ValueError, match="valid_seed"):
        train(valid_seed=-1)
    assert train(train_seed=0, valid_seed=10).train_seed == 0


def test_the_novelty_weight_is_between_zero_and_one_both_included():
    assert train(novelty_weight=0).novelty_weight == 0 and train(novelty_weight=1).novelty_weight == 1
    for bad in (-0.01, 1.01):
        with raises(ValueError, match="novelty_weight"):
            train(novelty_weight=bad)


def test_k_neighbours_must_be_smaller_than_the_population():
    assert train(population_size=11, k_neighbours=10).k_neighbours == 10
    with raises(ValueError, match="k_neighbours"):
        train(population_size=10, k_neighbours=10)
    with raises(ValueError, match="k_neighbours"):
        train(population_size=5)                                   # the default k = 10


def test_report_every_is_a_positive_number_or_none_for_never():
    assert train(report_every=None).report_every is None
    assert train(report_every=1).report_every == 1
    with raises(ValueError, match="report_every"):
        train(report_every=0)
    with raises(ValueError, match="report_every"):
        train(report_every=-5)


def test_the_validation_games_start_after_the_first_generation_of_training():
    assert train(train_seed=100, n_games=10, valid_seed=110).valid_seed == 110          # right after
    with raises(ValueError, match="valid_seed"):
        train(train_seed=100, n_games=10, valid_seed=109)
    with raises(ValueError, match="valid_seed"):
        train(train_seed=200_000, valid_seed=200_000)


###############################################################################
# last_generation: how far the training seeds may go
###############################################################################

def test_last_generation_is_the_last_one_whose_games_stay_below_the_validation_seed():
    config = train(train_seed=0, n_games=10, valid_seed=100)
    # generation 9 plays the seeds 90 ... 99, generation 10 would play 100 ... 109
    assert config.last_generation() == 9


def test_last_generation_when_the_range_is_not_a_multiple():
    config = train(train_seed=0, n_games=30, valid_seed=100)
    # generation 2 plays 60 ... 89, generation 3 would play 90 ... 119: it crosses 100
    assert config.last_generation() == 2


def test_last_generation_with_the_planned_seeds():
    config = train(n_games=100)                                    # 200_000 ... 210_000: 100 generations
    assert config.last_generation() == 99
    assert train(n_games=20).last_generation() == 499              # the "generation 500" of the old runs


###############################################################################
# ChampionConfig and check_seed_ranges
###############################################################################

def test_a_champion_config_has_the_planned_defaults():
    config = ChampionConfig()
    assert (config.champion_seed, config.n_finalists, config.n_score_games, config.n_final_games) \
        == (240_000, 5, 20, 100)


def test_the_champion_sizes_must_be_whole_numbers_of_at_least_one():
    for name, bad in (("n_finalists", 0), ("n_score_games", 0), ("n_final_games", 0), ("champion_seed", -1),
                      ("n_finalists", 1.5), ("n_final_games", True)):
        with raises(ValueError, match=name):
            ChampionConfig(**{name: bad})


def test_games_per_round_adds_both_stages():
    assert ChampionConfig(n_score_games=20, n_final_games=100).games_per_round == 120
    assert ChampionConfig(n_score_games=3, n_final_games=4).games_per_round == 7


def test_a_champion_config_cannot_be_changed_either():
    config = ChampionConfig()
    with raises(dataclasses.FrozenInstanceError):
        config.n_finalists = 9


def test_the_champion_games_must_start_after_the_validation_games():
    config = train(valid_seed=210_000, n_valid_games=100)
    check_seed_ranges(config, ChampionConfig(champion_seed=210_100))                       # right after: fine
    check_seed_ranges(config, ChampionConfig(champion_seed=240_000))
    with raises(ValueError, match="champion_seed"):
        check_seed_ranges(config, ChampionConfig(champion_seed=210_099))                   # the last validation game
    with raises(ValueError, match="champion_seed"):
        check_seed_ranges(config, ChampionConfig(champion_seed=0))


###############################################################################
# Runner
###############################################################################

def run_all():
    # Only the tests defined in THIS file: a console that stays open still holds the tests of earlier runs
    tests = [(name, test) for name, test in globals().items()
             if name.startswith("test_") and callable(test) ]
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