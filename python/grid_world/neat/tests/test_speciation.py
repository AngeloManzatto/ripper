"""
Created on Mon Oct 19 08:55:25 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/speciation.py: init_species and speciate.

The species set is plain data:

    {"species": {species_id: {"representative": genome, "members": [index, ...]}},
     "next_id": next free species id}

`members` are positions in the list of genomes given to speciate (this
generation only). `representative` is a stored copy of one genome; the next
generation compares its genomes to it.

What speciate(genomes, species_set, cfg=None) must do
-----------------------------------------------------
1. Go through the genomes in list order. Each genome is compared with the
   representatives, in increasing species-id order, and joins the FIRST
   species whose representative is at distance STRICTLY LESS than
   cfg["compatibility_threshold"] (compatibility_distance with the same cfg).
2. A genome that fits no species founds a new one: id = next_id (then
   next_id grows by one), and the founder is its representative right away,
   so later genomes of the same call can join it.
3. After everybody is placed:
     - species without members are dropped (ids are never reused);
     - each surviving species gets a new representative: the member closest
       to the OLD representative (the first one in case of a tie), stored as
       a deep copy.
4. Every genome ends in exactly one species. Members are Python ints in
   ascending order. Species ids are Python ints.
5. Nothing is modified: the genomes, the old species set and cfg stay the
   same; a NEW species set is returned.

Test genomes have one connection {0: weight}; with c3 = 1.0 the distance
between two of them is simply |weight_a - weight_b|.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.genome import init_genome
from neat.innovations import init_innovations
from neat.mutation import mutate
from neat.speciation import init_species, speciate

###############################################################################
# helpers
###############################################################################

def genome_with(weight):
    """A genome with one connection: distance = |weight difference| when c3 = 1."""
    return {"n_inputs": 1, "n_outputs": 1, "nodes": {},
            "connections": {0: {"src": 0, "dst": 1, "weight": weight, "enabled": True}}}


def population(*weights):
    return [genome_with(w) for w in weights]


def cfg_with(threshold=1.0, c3=1.0):
    cfg = default_config()
    cfg["compatibility_threshold"] = threshold
    cfg["compatibility_c3"] = c3
    return cfg


def members_of(species_set):
    return {sid: s["members"] for sid, s in species_set["species"].items()}


def rep_weights(species_set):
    return {sid: s["representative"]["connections"][0]["weight"]
            for sid, s in species_set["species"].items()}

###############################################################################
# init_species
###############################################################################

def test_init_species_is_empty():
    assert init_species() == {"species": {}, "next_id": 0}


def test_init_species_returns_a_new_dict_each_time():
    a = init_species()
    b = init_species()
    assert a is not b
    assert a["species"] is not b["species"]
    a["species"][0] = "x"
    assert init_species()["species"] == {}

###############################################################################
# the first generation
###############################################################################

def test_empty_population():
    result = speciate([], init_species(), cfg_with())
    assert result == {"species": {}, "next_id": 0}


def test_empty_population_keeps_the_counter():
    species_set = {"species": {}, "next_id": 7}
    result = speciate([], species_set, cfg_with())
    assert result == {"species": {}, "next_id": 7}


def test_a_single_genome_founds_species_zero():
    genomes = population(2.0)
    result = speciate(genomes, init_species(), cfg_with())
    assert members_of(result) == {0: [0]}
    assert result["species"][0]["representative"] == genomes[0]
    assert result["next_id"] == 1


def test_close_genomes_share_a_species_and_far_ones_split():
    genomes = population(0.0, 0.5, 5.0, 5.2)
    result = speciate(genomes, init_species(), cfg_with(threshold=1.0))
    assert members_of(result) == {0: [0, 1], 1: [2, 3]}
    assert result["next_id"] == 2


def test_identical_genomes_are_one_species():
    genomes = population(1.0, 1.0, 1.0, 1.0)
    result = speciate(genomes, init_species(), cfg_with())
    assert members_of(result) == {0: [0, 1, 2, 3]}


def test_threshold_is_strict():
    # distance exactly equal to the threshold is NOT compatible
    result = speciate(population(0.0, 1.0), init_species(), cfg_with(threshold=1.0))
    assert members_of(result) == {0: [0], 1: [1]}
    result = speciate(population(0.0, 0.999), init_species(), cfg_with(threshold=1.0))
    assert members_of(result) == {0: [0, 1]}


def test_a_genome_joins_the_founder_of_a_species_created_in_the_same_call():
    # 5.0 founds species 1 (too far from 0.0); 5.4 must join it, not found a third
    result = speciate(population(0.0, 5.0, 5.4), init_species(), cfg_with(threshold=1.0))
    assert members_of(result) == {0: [0], 1: [1, 2]}


def test_the_first_compatible_species_wins_not_the_closest():
    # species 0 (rep 0.0) and species 1 (rep 3.0) are 3.0 apart (> threshold 2.0)
    first = speciate(population(0.0, 3.0), init_species(), cfg_with(threshold=2.0))
    assert members_of(first) == {0: [0], 1: [1]}
    # 1.7 is compatible with both (1.7 and 1.3), closer to species 1, but
    # species 0 comes first
    second = speciate(population(1.7), first, cfg_with(threshold=2.0))
    assert members_of(second) == {0: [0]}


def test_every_genome_is_in_exactly_one_species():
    rng = np.random.default_rng(1)
    weights = [float(w) for w in rng.uniform(0, 20, size=40)]
    genomes = population(*weights)
    result = speciate(genomes, init_species(), cfg_with(threshold=1.5))
    everyone = []
    for members in members_of(result).values():
        everyone.extend(members)
    assert sorted(everyone) == list(range(40))
    assert len(result["species"]) > 1


def test_members_are_python_ints_in_ascending_order():
    genomes = population(0.0, 5.0, 0.1, 5.1, 0.2)
    result = speciate(genomes, init_species(), cfg_with(threshold=1.0))
    assert members_of(result) == {0: [0, 2, 4], 1: [1, 3]}
    for sid, species in result["species"].items():
        assert type(sid) is int
        for index in species["members"]:
            assert type(index) is int
    assert type(result["next_id"]) is int

###############################################################################
# the next generations
###############################################################################

def test_same_genomes_again_give_the_same_species():
    genomes = population(0.0, 0.5, 5.0, 5.2)
    cfg = cfg_with(threshold=1.0)
    first = speciate(genomes, init_species(), cfg)
    second = speciate(genomes, first, cfg)
    assert members_of(second) == members_of(first)
    assert second["next_id"] == first["next_id"]


def test_a_far_genome_founds_a_new_species_and_old_ones_stay():
    cfg = cfg_with(threshold=1.0)
    first = speciate(population(0.0, 5.0), init_species(), cfg)
    second = speciate(population(0.0, 5.0, 20.0), first, cfg)
    assert members_of(second) == {0: [0], 1: [1], 2: [2]}
    assert second["next_id"] == 3


def test_extinct_species_are_dropped_and_ids_are_not_reused():
    cfg = cfg_with(threshold=1.0)
    first = speciate(population(0.0, 10.0), init_species(), cfg)
    assert set(first["species"]) == {0, 1}
    second = speciate(population(0.0, 20.0), first, cfg)
    assert members_of(second) == {0: [0], 2: [1]}
    assert second["next_id"] == 3


def test_no_species_without_members_is_kept():
    cfg = cfg_with(threshold=1.0)
    first = speciate(population(0.0, 10.0), init_species(), cfg)
    second = speciate(population(0.0, 0.1), first, cfg)
    for species in second["species"].values():
        assert len(species["members"]) > 0
    assert set(second["species"]) == {0}


def test_representative_is_the_member_closest_to_the_old_one():
    cfg = cfg_with(threshold=1.0)
    first = speciate(population(0.0), init_species(), cfg)       # rep 0.0
    second = speciate(population(0.9, 0.3, 0.6), first, cfg)
    assert members_of(second) == {0: [0, 1, 2]}
    assert rep_weights(second) == {0: 0.3}


def test_representative_tie_goes_to_the_first_member():
    cfg = cfg_with(threshold=1.0)
    first = speciate(population(0.0), init_species(), cfg)       # rep 0.0
    second = speciate(population(0.5, -0.5), first, cfg)
    assert rep_weights(second) == {0: 0.5}


def test_a_new_species_keeps_its_founder_as_representative():
    cfg = cfg_with(threshold=1.0)
    result = speciate(population(5.0, 5.4, 5.2), init_species(), cfg)
    assert rep_weights(result) == {0: 5.0}


def test_next_generation_compares_with_the_new_representative():
    cfg = cfg_with(threshold=1.0)
    first = speciate(population(0.0), init_species(), cfg)
    second = speciate(population(0.9, 0.3, 0.6), first, cfg)     # new rep: 0.3
    # 1.2 is 0.9 from the new rep (compatible) but 1.2 from the old one
    third = speciate(population(1.2), second, cfg)
    assert members_of(third) == {0: [0]}
    assert third["next_id"] == second["next_id"]

###############################################################################
# copies, not references
###############################################################################

def test_representative_is_a_deep_copy():
    genomes = population(0.0, 0.5)
    result = speciate(genomes, init_species(), cfg_with())
    rep = result["species"][0]["representative"]
    assert rep is not genomes[0] and rep is not genomes[1]
    snapshot = copy.deepcopy(rep)
    genomes[0]["connections"][0]["weight"] = 99.0
    genomes[1]["connections"][0]["weight"] = 99.0
    assert result["species"][0]["representative"] == snapshot


def test_genomes_are_not_modified():
    genomes = population(0.0, 0.5, 5.0)
    snapshot = copy.deepcopy(genomes)
    speciate(genomes, init_species(), cfg_with())
    assert genomes == snapshot


def test_old_species_set_is_not_modified():
    cfg = cfg_with()
    first = speciate(population(0.0, 5.0), init_species(), cfg)
    snapshot = copy.deepcopy(first)
    speciate(population(0.1, 5.1, 20.0), first, cfg)
    assert first == snapshot


def test_returns_a_new_species_set():
    cfg = cfg_with()
    first = speciate(population(0.0), init_species(), cfg)
    second = speciate(population(0.0), first, cfg)
    assert second is not first
    assert second["species"] is not first["species"]


def test_cfg_is_not_modified():
    cfg = cfg_with()
    snapshot = copy.deepcopy(cfg)
    speciate(population(0.0, 5.0), init_species(), cfg)
    assert cfg == snapshot

###############################################################################
# cfg
###############################################################################

def test_threshold_comes_from_cfg():
    genomes = population(0.0, 2.0)
    wide = speciate(genomes, init_species(), cfg_with(threshold=3.0))
    narrow = speciate(genomes, init_species(), cfg_with(threshold=1.0))
    assert len(wide["species"]) == 1
    assert len(narrow["species"]) == 2


def test_distance_coefficients_come_from_cfg():
    genomes = population(0.0, 2.0)
    # c3 = 0.1 -> distance 0.2 (compatible); c3 = 5.0 -> distance 10 (not)
    assert len(speciate(genomes, init_species(), cfg_with(threshold=1.0, c3=0.1))["species"]) == 1
    assert len(speciate(genomes, init_species(), cfg_with(threshold=1.0, c3=5.0))["species"]) == 2


def test_default_cfg_is_used_when_none():
    # defaults: threshold 3.0, c3 0.4 -> weights 0.0 and 5.0 are 2.0 apart
    result = speciate(population(0.0, 5.0), init_species())
    assert len(result["species"]) == 1
    result = speciate(population(0.0, 9.0), init_species())      # 3.6 apart
    assert len(result["species"]) == 2

###############################################################################
# real genomes
###############################################################################

def test_a_mutated_population_splits_into_several_species():
    innovations = init_innovations(3, 2)
    rng = np.random.default_rng(2)
    genomes = [init_genome(3, 2, rng) for _ in range(20)]
    cfg = default_config()
    cfg["node_add_prob"] = 1.0
    cfg["conn_add_prob"] = 1.0
    cfg["compatibility_threshold"] = 0.3
    for _ in range(6):
        genomes = [mutate(g, rng, innovations, cfg) for g in genomes]
    result = speciate(genomes, init_species(), cfg)
    everyone = []
    for species in result["species"].values():
        everyone.extend(species["members"])
    assert sorted(everyone) == list(range(20))
    assert len(result["species"]) > 1


def test_a_fresh_population_is_one_species_with_default_cfg():
    rng = np.random.default_rng(3)
    genomes = [init_genome(29, 4, rng) for _ in range(30)]
    result = speciate(genomes, init_species())
    assert len(result["species"]) == 1
    assert result["species"][0]["members"] == list(range(30))

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"speciation: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()