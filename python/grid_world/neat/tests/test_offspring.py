"""
Created on Mon Oct 19 09:25:56 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/offspring.py: compute_offspring(species_set, fitnesses, cfg=None).

compute_offspring decides how many children each species gets in the next
generation. It does NOT build the children (selection and crossover do that).

How the counts are decided
--------------------------
1. Every species gets a SCORE. Its fitness is the MEAN fitness of its members
   (this is fitness sharing: a big species does not win just by being big),
   shifted so that the lowest member fitness among the species in the species
   set becomes 0:

       score(species) = mean_fitness(species) - lowest_member_fitness

   Only genomes that belong to a species in the species set count; fitnesses
   of genomes that are in no species (for example removed as stagnant) are
   ignored. Because of the shift, adding a constant to every fitness, or
   multiplying all of them by a positive constant, changes nothing.
2. The population (cfg["population_size"]) is divided in proportion to the
   scores. If every score is 0 (all fitnesses equal) the population is divided
   equally.
3. The division is exact: the counts always add up to population_size. Each
   species gets the whole part of its share, and the children that are left
   go one by one to the species with the biggest fractional part (on a tie the
   lower species id first).
4. A species with score 0 gets no children (unless everything is equal).
   There is no minimum; protecting the best genome is a job for reproduction.

Returns {species_id: count} with Python ints, ascending species ids.
Raises ValueError for an empty species set or a NaN fitness, and IndexError or
ValueError if a member has no fitness. Nothing is modified; no randomness.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.offspring import compute_offspring

###############################################################################
# helpers
###############################################################################

def make_set(members_by_id):
    """A species set built by hand: {species_id: [member indices]}."""
    species = {}
    for species_id, members in members_by_id.items():
        species[species_id] = {"representative": {}, "members": list(members)}
    return {"species": species, "next_id": max(members_by_id) + 1}


def cfg_with(population_size):
    cfg = default_config()
    cfg["population_size"] = population_size
    return cfg


def two_members_each(*values):
    """Species 0, 1, 2... each with two members that both have the given fitness.

    Returns (species_set, fitnesses).
    """
    members = {}
    fitnesses = []
    for species_id, value in enumerate(values):
        members[species_id] = [len(fitnesses), len(fitnesses) + 1]
        fitnesses.extend([value, value])
    return make_set(members), fitnesses

###############################################################################
# proportional shares
###############################################################################

def test_shares_follow_the_shifted_mean_fitness():
    species_set, fitnesses = two_members_each(0.0, 10.0, 30.0)
    # scores 0, 10, 30 -> shares 0, 1/4, 3/4
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 0, 1: 25, 2: 75}


def test_equal_species_split_the_population_equally():
    species_set, fitnesses = two_members_each(5.0, 5.0, 5.0)
    assert compute_offspring(species_set, fitnesses, cfg_with(150)) == {0: 50, 1: 50, 2: 50}


def test_a_single_species_gets_everyone():
    species_set, fitnesses = two_members_each(3.0)
    assert compute_offspring(species_set, fitnesses, cfg_with(150)) == {0: 150}


def test_better_species_never_get_fewer_children():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n_species = int(rng.integers(2, 8))
        values = [float(v) for v in rng.uniform(-10, 10, size=n_species)]
        species_set, fitnesses = two_members_each(*values)
        counts = compute_offspring(species_set, fitnesses, cfg_with(150))
        for a in range(n_species):
            for b in range(n_species):
                if values[a] > values[b]:
                    assert counts[a] >= counts[b]

###############################################################################
# the mean, not the size and not the maximum
###############################################################################

def test_a_big_species_does_not_win_by_size():
    # species 0: 4 members with fitness 10; species 1: 1 member with fitness 10;
    # species 2: 1 member with fitness 0.  Means 10, 10, 0 -> 50%, 50%, 0%
    species_set = make_set({0: [0, 1, 2, 3], 1: [4], 2: [5]})
    fitnesses = [10.0, 10.0, 10.0, 10.0, 10.0, 0.0]
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 50, 1: 50, 2: 0}


def test_the_mean_is_used_not_the_maximum():
    # species 0 has fitnesses 0 and 20 (mean 10, max 20); species 1 has 10 and 10
    # (mean 10, max 10); species 2 has 0 -> same mean, same share
    species_set = make_set({0: [0, 1], 1: [2, 3], 2: [4]})
    fitnesses = [0.0, 20.0, 10.0, 10.0, 0.0]
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 50, 1: 50, 2: 0}


def test_member_positions_pick_the_right_fitnesses():
    species_set = make_set({0: [3, 1], 1: [0, 2]})
    fitnesses = [20.0, 0.0, 20.0, 0.0]
    # species 0 -> fitnesses[3], fitnesses[1] = 0, 0.  species 1 -> 20, 20
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 0, 1: 100}

###############################################################################
# shift and scale
###############################################################################

def test_adding_a_constant_changes_nothing():
    species_set, fitnesses = two_members_each(0.0, 10.0, 30.0)
    base = compute_offspring(species_set, fitnesses, cfg_with(100))
    for shift in (-1000.0, -5.0, 7.0, 1000.0):
        shifted = [f + shift for f in fitnesses]
        assert compute_offspring(species_set, shifted, cfg_with(100)) == base


def test_negative_fitness_works():
    species_set, fitnesses = two_members_each(-30.0, -20.0, 0.0)
    # lowest -30: scores 0, 10, 30
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 0, 1: 25, 2: 75}


def test_multiplying_by_a_constant_changes_nothing():
    species_set, fitnesses = two_members_each(0.0, 10.0, 30.0)
    base = compute_offspring(species_set, fitnesses, cfg_with(100))
    for factor in (0.25, 2.0, 8.0):
        scaled = [f * factor for f in fitnesses]
        assert compute_offspring(species_set, scaled, cfg_with(100)) == base


def test_genomes_in_no_species_are_ignored():
    # genome 2 belongs to no species (for example a removed stagnant one)
    species_set = make_set({0: [0], 1: [1]})
    fitnesses = [10.0, 20.0, -100.0]
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 0, 1: 100}

###############################################################################
# every score is zero
###############################################################################

def test_all_equal_fitness_splits_equally():
    species_set, fitnesses = two_members_each(4.0, 4.0, 4.0, 4.0)
    assert compute_offspring(species_set, fitnesses, cfg_with(100)) == {0: 25, 1: 25, 2: 25, 3: 25}


def test_all_equal_fitness_leftover_goes_to_the_lowest_ids():
    species_set, fitnesses = two_members_each(4.0, 4.0, 4.0)
    assert compute_offspring(species_set, fitnesses, cfg_with(10)) == {0: 4, 1: 3, 2: 3}


def test_all_zero_fitness_is_fine():
    species_set, fitnesses = two_members_each(0.0, 0.0)
    assert compute_offspring(species_set, fitnesses, cfg_with(10)) == {0: 5, 1: 5}

###############################################################################
# exact division
###############################################################################

def test_counts_always_add_up_to_the_population_size():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n_species = int(rng.integers(1, 12))
        values = [float(v) for v in rng.uniform(-50, 50, size=n_species)]
        species_set, fitnesses = two_members_each(*values)
        population_size = int(rng.integers(1, 400))
        counts = compute_offspring(species_set, fitnesses, cfg_with(population_size))
        assert sum(counts.values()) == population_size
        assert all(count >= 0 for count in counts.values())


def test_leftover_children_go_to_the_biggest_fractions():
    # scores 1, 2, 4 (total 7), population 10 -> quotas 1.43, 2.86, 5.71
    species_set = make_set({0: [0, 1], 1: [2, 3], 2: [4, 5]})
    fitnesses = [0.0, 2.0, 2.0, 2.0, 4.0, 4.0]       # means 1, 2, 4 and lowest 0
    # floors 1, 2, 5 (sum 8); fractions .43, .86, .71 -> species 1 and 2 get one more
    assert compute_offspring(species_set, fitnesses, cfg_with(10)) == {0: 1, 1: 3, 2: 6}


def test_tie_between_fractions_goes_to_the_lower_id():
    # shares 1/4, 1/4, 1/2 of 10 -> quotas 2.5, 2.5, 5.0 -> one left over, tie on .5
    species_set = make_set({0: [0, 1], 1: [2, 3], 2: [4, 5]})
    fitnesses = [0.0, 2.0, 1.0, 1.0, 2.0, 2.0]       # means 1, 1, 2; lowest 0
    assert compute_offspring(species_set, fitnesses, cfg_with(10)) == {0: 3, 1: 2, 2: 5}


def test_population_smaller_than_the_number_of_species():
    species_set, fitnesses = two_members_each(0.0, 5.0, 10.0, 20.0)
    counts = compute_offspring(species_set, fitnesses, cfg_with(1))
    assert counts == {0: 0, 1: 0, 2: 0, 3: 1}


def test_population_size_comes_from_cfg_and_defaults_to_150():
    species_set, fitnesses = two_members_each(1.0, 2.0, 3.0)
    assert sum(compute_offspring(species_set, fitnesses, cfg_with(37)).values()) == 37
    assert sum(compute_offspring(species_set, fitnesses).values()) == 150

###############################################################################
# output format
###############################################################################

def test_counts_are_python_ints_with_ascending_ids():
    species_set = make_set({5: [0, 1], 2: [2, 3], 9: [4, 5]})
    fitnesses = [1.0, 1.0, 5.0, 5.0, 9.0, 9.0]
    counts = compute_offspring(species_set, fitnesses, cfg_with(60))
    assert list(counts.keys()) == [2, 5, 9]
    for species_id, count in counts.items():
        assert type(species_id) is int
        assert type(count) is int


def test_works_with_numpy_fitness_values():
    species_set, fitnesses = two_members_each(0.0, 10.0, 30.0)
    counts = compute_offspring(species_set, np.array(fitnesses), cfg_with(100))
    assert counts == {0: 0, 1: 25, 2: 75}
    for count in counts.values():
        assert type(count) is int

###############################################################################
# errors and purity
###############################################################################

def test_empty_species_set_raises():
    try:
        compute_offspring({"species": {}, "next_id": 0}, [], cfg_with(10))
    except ValueError:
        return
    raise AssertionError("an empty species set must raise ValueError")


def test_nan_fitness_raises():
    species_set = make_set({0: [0, 1]})
    try:
        compute_offspring(species_set, [1.0, float("nan")], cfg_with(10))
    except ValueError:
        return
    raise AssertionError("NaN fitness must raise ValueError")


def test_missing_fitness_raises():
    species_set = make_set({0: [0, 1, 2]})
    try:
        compute_offspring(species_set, [1.0, 2.0], cfg_with(10))
    except (IndexError, ValueError):
        return
    raise AssertionError("a member without fitness must raise")


def test_inputs_are_not_modified():
    species_set, fitnesses = two_members_each(0.0, 10.0, 30.0)
    cfg = cfg_with(100)
    set_snapshot, fit_snapshot, cfg_snapshot = (copy.deepcopy(species_set),
                                                list(fitnesses), copy.deepcopy(cfg))
    compute_offspring(species_set, fitnesses, cfg)
    assert species_set == set_snapshot
    assert fitnesses == fit_snapshot
    assert cfg == cfg_snapshot


def test_same_input_gives_the_same_result():
    species_set, fitnesses = two_members_each(1.0, 2.5, 7.0, 7.0)
    first = compute_offspring(species_set, fitnesses, cfg_with(97))
    second = compute_offspring(species_set, fitnesses, cfg_with(97))
    assert first == second

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"offspring: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()