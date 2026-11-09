"""
Created on Mon Oct 19 09:10:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/stagnation.py: update_stagnation, find_stagnant, remove_species.

A species is STAGNANT when its best fitness has not improved for too long.
Stagnant species are removed, so the population does not keep spending
offspring on a dead end.

The stagnation record is plain data, separate from the species set:

    {species_id: {"best_fitness": float, "last_improved": generation}}

best_fitness is the best value EVER seen by that species (a record that only
goes up), last_improved is the generation where the record was last beaten.
The library always MAXIMISES fitness (to minimise something, negate it).

update_stagnation(stagnation, species_set, fitnesses, generation)
    - fitnesses[i] is the fitness of genome i (same list given to speciate);
      the best of a species is the maximum over its members;
    - a species not yet in the record starts with best_fitness = its best and
      last_improved = generation;
    - a known species improves only if its best is STRICTLY above the record:
      then both fields are updated; otherwise the old record is kept;
    - species that are not in the species set any more are dropped;
    - NaN fitness raises ValueError; a member index missing from fitnesses
      raises IndexError or ValueError;
    - returns a NEW dict (best_fitness is a Python float); nothing is modified.

find_stagnant(stagnation, generation, cfg=None)
    - a species is stagnant when generation - last_improved >=
      cfg["max_stagnation"];
    - but at least cfg["min_species_to_keep"] species are always kept: if
      removing every stagnant species would leave fewer, the stagnant species
      with the best record are spared (the worst records go first; on a tie
      the higher id, the newer species, goes first);
    - returns the ids to remove, sorted ascending; nothing is modified.

remove_species(species_set, species_ids)
    - returns a new species set (deep copies) without those species; next_id
      is unchanged; an id that is not in the set raises ValueError.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.stagnation import update_stagnation, find_stagnant, remove_species

###############################################################################
# helpers
###############################################################################

def make_set(members_by_id, next_id=None):
    """A species set built by hand: {species_id: [member indices]}."""
    species = {}
    for species_id, members in members_by_id.items():
        species[species_id] = {"representative": {"tag": species_id}, "members": list(members)}
    if next_id is None:
        next_id = max(members_by_id) + 1 if members_by_id else 0
    return {"species": species, "next_id": next_id}


def record(best, last):
    return {"best_fitness": best, "last_improved": last}


def cfg_with(max_stagnation=15, keep=2):
    cfg = default_config()
    cfg["max_stagnation"] = max_stagnation
    cfg["min_species_to_keep"] = keep
    return cfg

###############################################################################
# update_stagnation: first generation and new species
###############################################################################

def test_first_generation_starts_every_species():
    species_set = make_set({0: [0, 1], 1: [2, 3]})
    result = update_stagnation({}, species_set, [1.0, 3.0, 2.0, 0.5], 0)
    assert result == {0: record(3.0, 0), 1: record(2.0, 0)}


def test_the_best_of_a_species_uses_only_its_own_members():
    species_set = make_set({0: [0, 2], 1: [1, 3]})
    result = update_stagnation({}, species_set, [1.0, 100.0, 2.0, 50.0], 4)
    assert result[0]["best_fitness"] == 2.0
    assert result[1]["best_fitness"] == 100.0


def test_negative_fitness_is_fine():
    species_set = make_set({0: [0, 1]})
    result = update_stagnation({}, species_set, [-5.0, -2.0], 0)
    assert result[0]["best_fitness"] == -2.0


def test_new_species_in_a_later_generation_starts_at_that_generation():
    species_set = make_set({0: [0], 1: [1]})
    stagnation = {0: record(1.0, 3)}
    result = update_stagnation(stagnation, species_set, [0.5, 0.2], 9)
    assert result[1] == record(0.2, 9)
    assert result[0] == record(1.0, 3)

###############################################################################
# update_stagnation: improvement
###############################################################################

def test_a_better_best_updates_both_fields():
    species_set = make_set({0: [0, 1]})
    result = update_stagnation({0: record(1.0, 0)}, species_set, [2.0, 0.0], 5)
    assert result == {0: record(2.0, 5)}


def test_an_equal_best_is_not_an_improvement():
    species_set = make_set({0: [0, 1]})
    result = update_stagnation({0: record(2.0, 0)}, species_set, [2.0, 0.0], 5)
    assert result == {0: record(2.0, 0)}


def test_a_worse_best_keeps_the_old_record():
    species_set = make_set({0: [0, 1]})
    result = update_stagnation({0: record(5.0, 2)}, species_set, [1.0, 0.0], 9)
    assert result == {0: record(5.0, 2)}


def test_the_record_only_goes_up():
    species_set = make_set({0: [0]})
    stagnation = {}
    for generation, fitness in enumerate([1.0, 3.0, 2.0, 2.5, 4.0, 1.0]):
        stagnation = update_stagnation(stagnation, species_set, [fitness], generation)
    assert stagnation == {0: record(4.0, 4)}

###############################################################################
# update_stagnation: species that disappear, edge cases
###############################################################################

def test_species_that_are_gone_are_dropped():
    species_set = make_set({1: [0, 1]})
    stagnation = {0: record(1.0, 0), 1: record(1.0, 0)}
    result = update_stagnation(stagnation, species_set, [0.5, 0.5], 3)
    assert set(result) == {1}


def test_empty_species_set_gives_an_empty_record():
    assert update_stagnation({0: record(1.0, 0)}, make_set({}), [], 3) == {}


def test_best_fitness_is_a_python_float():
    species_set = make_set({0: [0, 1], 1: [2]})
    result = update_stagnation({}, species_set, [1, 3, np.float64(2.5)], 0)
    for entry in result.values():
        assert type(entry["best_fitness"]) is float


def test_nan_fitness_raises():
    species_set = make_set({0: [0, 1]})
    try:
        update_stagnation({}, species_set, [1.0, float("nan")], 0)
    except ValueError:
        return
    raise AssertionError("NaN fitness must raise ValueError")


def test_missing_fitness_raises():
    species_set = make_set({0: [0, 1, 2]})
    try:
        update_stagnation({}, species_set, [1.0, 2.0], 0)
    except (IndexError, ValueError):
        return
    raise AssertionError("a member without fitness must raise")


def test_update_does_not_modify_its_inputs_and_returns_new_dicts():
    species_set = make_set({0: [0, 1]})
    stagnation = {0: record(5.0, 2)}
    set_snapshot, stag_snapshot = copy.deepcopy(species_set), copy.deepcopy(stagnation)
    fitnesses = [1.0, 0.0]
    result = update_stagnation(stagnation, species_set, fitnesses, 9)
    assert species_set == set_snapshot
    assert stagnation == stag_snapshot
    assert fitnesses == [1.0, 0.0]
    assert result is not stagnation
    assert result[0] is not stagnation[0]

###############################################################################
# find_stagnant
###############################################################################

def test_nothing_is_stagnant_when_everybody_improved_recently():
    stagnation = {0: record(1.0, 10), 1: record(2.0, 12), 2: record(3.0, 14)}
    assert find_stagnant(stagnation, 15, cfg_with(max_stagnation=15, keep=0)) == []


def test_stagnation_boundary():
    stagnation = {0: record(1.0, 5)}
    cfg = cfg_with(max_stagnation=10, keep=0)
    assert find_stagnant(stagnation, 14, cfg) == []        # waited 9
    assert find_stagnant(stagnation, 15, cfg) == [0]       # waited 10
    assert find_stagnant(stagnation, 16, cfg) == [0]       # waited 11


def test_only_the_stagnant_species_are_listed():
    stagnation = {0: record(1.0, 0), 1: record(2.0, 18), 2: record(3.0, 0)}
    assert find_stagnant(stagnation, 20, cfg_with(max_stagnation=15, keep=0)) == [0, 2]


def test_all_stagnant_keeps_the_best_records():
    stagnation = {0: record(5.0, 0), 1: record(1.0, 0), 2: record(4.0, 0),
                  3: record(2.0, 0), 4: record(3.0, 0)}
    # keep 2 species: the two best records (ids 0 and 2) survive
    assert find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=2)) == [1, 3, 4]


def test_fresh_species_count_towards_the_minimum():
    # 2 fresh species already satisfy keep = 2, so every stagnant one goes
    stagnation = {0: record(9.0, 0), 1: record(8.0, 0), 2: record(1.0, 98), 3: record(2.0, 98)}
    assert find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=2)) == [0, 1]


def test_one_fresh_species_and_keep_two_spares_the_best_stagnant_one():
    stagnation = {0: record(9.0, 0), 1: record(8.0, 0), 2: record(1.0, 98)}
    # 3 species, keep 2: only one may go, the worst record among the stagnant
    assert find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=2)) == [1]


def test_minimum_bigger_than_the_number_of_species_removes_nothing():
    stagnation = {0: record(1.0, 0), 1: record(2.0, 0)}
    assert find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=5)) == []


def test_a_single_stagnant_species_is_kept_when_keep_is_two():
    assert find_stagnant({0: record(1.0, 0)}, 99, cfg_with(max_stagnation=15, keep=2)) == []


def test_keep_zero_allows_removing_everything():
    stagnation = {0: record(1.0, 0), 1: record(2.0, 0)}
    assert find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=0)) == [0, 1]


def test_tie_on_the_record_removes_the_newer_species_first():
    stagnation = {0: record(3.0, 0), 1: record(3.0, 0), 2: record(3.0, 0)}
    # keep 2 -> only one may go: the highest id
    assert find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=2)) == [2]


def test_result_is_sorted_python_ints():
    stagnation = {7: record(1.0, 0), 3: record(2.0, 0), 5: record(3.0, 0)}
    result = find_stagnant(stagnation, 99, cfg_with(max_stagnation=15, keep=0))
    assert result == [3, 5, 7]
    for species_id in result:
        assert type(species_id) is int


def test_empty_record_gives_an_empty_list():
    assert find_stagnant({}, 10, cfg_with()) == []


def test_cfg_values_are_read_from_cfg():
    stagnation = {0: record(1.0, 0), 1: record(2.0, 0), 2: record(3.0, 0)}
    assert find_stagnant(stagnation, 5, cfg_with(max_stagnation=5, keep=2)) == [0]
    assert find_stagnant(stagnation, 5, cfg_with(max_stagnation=6, keep=2)) == []
    assert find_stagnant(stagnation, 5, cfg_with(max_stagnation=5, keep=1)) == [0, 1]


def test_default_cfg_is_used_when_none():
    # defaults: max_stagnation 15, min_species_to_keep 2
    stagnation = {0: record(1.0, 0), 1: record(2.0, 0), 2: record(3.0, 0)}
    assert find_stagnant(stagnation, 14) == []
    assert find_stagnant(stagnation, 15) == [0]


def test_find_stagnant_does_not_modify_its_inputs():
    stagnation = {0: record(1.0, 0), 1: record(2.0, 0), 2: record(3.0, 0)}
    cfg = cfg_with()
    stag_snapshot, cfg_snapshot = copy.deepcopy(stagnation), copy.deepcopy(cfg)
    find_stagnant(stagnation, 99, cfg)
    assert stagnation == stag_snapshot
    assert cfg == cfg_snapshot

###############################################################################
# remove_species
###############################################################################

def test_remove_species_drops_only_the_given_ids():
    species_set = make_set({0: [0, 1], 1: [2], 2: [3, 4]})
    result = remove_species(species_set, [1])
    assert set(result["species"]) == {0, 2}
    assert result["species"][0] == species_set["species"][0]
    assert result["species"][2] == species_set["species"][2]


def test_remove_species_keeps_next_id():
    species_set = make_set({0: [0], 1: [1]}, next_id=7)
    assert remove_species(species_set, [0, 1])["next_id"] == 7


def test_remove_nothing_gives_an_equal_but_new_set():
    species_set = make_set({0: [0], 1: [1]})
    result = remove_species(species_set, [])
    assert result == species_set
    assert result is not species_set
    assert result["species"] is not species_set["species"]


def test_remove_every_species():
    species_set = make_set({0: [0], 1: [1]})
    assert remove_species(species_set, [0, 1])["species"] == {}


def test_remove_unknown_id_raises():
    species_set = make_set({0: [0]})
    try:
        remove_species(species_set, [5])
    except ValueError:
        return
    raise AssertionError("unknown species id must raise ValueError")


def test_remove_species_does_not_modify_the_input_and_copies_the_rest():
    species_set = make_set({0: [0, 1], 1: [2]})
    snapshot = copy.deepcopy(species_set)
    result = remove_species(species_set, [1])
    assert species_set == snapshot
    assert result["species"][0] is not species_set["species"][0]
    assert result["species"][0]["members"] is not species_set["species"][0]["members"]
    assert result["species"][0]["representative"] is not species_set["species"][0]["representative"]

###############################################################################
# the three together
###############################################################################

def test_a_flat_species_is_removed_after_max_stagnation_generations():
    # species 0 keeps improving, species 1 is flat, species 2 improves slowly
    species_set = make_set({0: [0], 1: [1], 2: [2]})
    cfg = cfg_with(max_stagnation=5, keep=1)
    stagnation = {}
    removed_at = None
    for generation in range(20):
        fitnesses = [float(generation), 10.0, generation * 0.5]
        stagnation = update_stagnation(stagnation, species_set, fitnesses, generation)
        stagnant = find_stagnant(stagnation, generation, cfg)
        if stagnant and removed_at is None:
            removed_at = (generation, stagnant)
            species_set = remove_species(species_set, stagnant)
        if removed_at is not None:
            break
    # species 1 reached its record at generation 0 and waited 5 generations
    assert removed_at == (5, [1])
    assert set(species_set["species"]) == {0, 2}

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"stagnation: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()