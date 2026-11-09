"""
Created on Mon Oct 19 11:16:51 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/reproduction.py: rank_members, parent_pool, find_champion,
ensure_champion_slot and reproduce.

Reproduction builds the next population. compute_offspring already decided HOW
MANY children each species gets (counts); reproduction decides WHICH children.

rank_members(members, fitnesses)
    Member indices sorted from best to worst fitness; equal fitness: lower
    index first. NaN raises ValueError. Returns Python ints; no modification.

parent_pool(ranked, cfg=None)
    The best members that may become parents: the first
    ceil(survival_threshold * size) of `ranked`, but never fewer than 2 (or
    the whole species if it has fewer), never more than the species.

find_champion(species_set, fitnesses)
    (index, species_id) of the genome with the best fitness among all the
    members of the species set (equal fitness: lowest index).

ensure_champion_slot(species_set, fitnesses, counts)
    Returns new counts in which the champion's species has at least 1 child:
    if it had 0, one child is taken from the species with the most children
    (lowest id on a tie). The total does not change; counts are not modified.

reproduce(genomes, fitnesses, species_set, counts, innovations, rng, cfg=None)
    Returns the new population (a list of new genomes, sum(counts) of them).
    For every species with at least one child, in ascending id order:
      - ELITES: the best elitism_count members are copied unchanged, if the
        species has at least elitism_min_species_size members (never more than
        the species' children). The champion's species always keeps its
        champion, even if it is small.  Elites count as children of the species.
      - The other children are bred from the parent pool of the species:
          * pool of 2 or more: two different parents are drawn; the fitter one
            is parent_a of crossover (equal fitness: random order); then, unless
            a roll below mate_only_prob says "mate only", the child is mutated;
          * pool of 1: the child is a mutated clone of that parent.
    The champion of the whole population therefore always survives unchanged.
    counts must have exactly the species ids of the species set and no negative
    value (ValueError otherwise). Nothing given is modified, children share
    nothing with the parents, the same seed gives the same population.

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
from neat.compatibility import compatibility_distance
from neat.speciation import init_species, speciate
from neat.offspring import compute_offspring
from neat.reproduction import (rank_members, parent_pool, find_champion,
                               ensure_champion_slot, reproduce)

###############################################################################
# helpers
###############################################################################

def tagged(weight, extra_keys=()):
    """A tiny genome: connection 0 has the given weight (a unique tag)."""
    connections = {0: {"src": 0, "dst": 1, "weight": weight, "enabled": True}}
    for key in extra_keys:
        connections[key] = {"src": 0, "dst": 1, "weight": weight, "enabled": True}
    return {"n_inputs": 1, "n_outputs": 1,
            "nodes": {0: {"type": "input"},
                      1: {"type": "output", "bias": 0.0, "activation": "sigmoid"}},
            "connections": connections}


def make_set(members_by_id):
    species = {}
    for species_id, members in members_by_id.items():
        species[species_id] = {"representative": {}, "members": list(members)}
    return {"species": species, "next_id": max(members_by_id) + 1}


def cfg_with(**overrides):
    cfg = default_config()
    cfg["population_size"] = 10
    cfg.update(overrides)
    return cfg


def no_mutation(**overrides):
    """cfg where mutate changes nothing."""
    cfg = cfg_with(weight_mutate_prob=0.0, bias_mutate_prob=0.0,
                   activation_mutate_prob=0.0, node_add_prob=0.0, conn_add_prob=0.0)
    cfg.update(overrides)
    return cfg


def always_mutates(**overrides):
    """cfg where mutate always changes every weight (a continuous nudge)."""
    cfg = cfg_with(weight_mutate_prob=1.0, weight_perturb_prob=1.0, weight_perturb_power=0.5,
                   bias_mutate_prob=0.0, activation_mutate_prob=0.0,
                   node_add_prob=0.0, conn_add_prob=0.0)
    cfg.update(overrides)
    return cfg


def weights(genomes):
    return [g["connections"][0]["weight"] for g in genomes]


def n_exact_copies(population, originals):
    """How many genomes of `population` are exactly equal to one of `originals`."""
    count = 0
    for genome in population:
        for original in originals:
            if genome == original:
                count += 1
                break
    return count


def is_acyclic(genome):
    incoming = {n: 0 for n in genome["nodes"]}
    for c in genome["connections"].values():
        incoming[c["dst"]] += 1
    ready = [n for n, count in incoming.items() if count == 0]
    removed = 0
    while ready:
        current = ready.pop()
        removed += 1
        for c in genome["connections"].values():
            if c["src"] == current:
                incoming[c["dst"]] -= 1
                if incoming[c["dst"]] == 0:
                    ready.append(c["dst"])
    return removed == len(genome["nodes"])

###############################################################################
# rank_members
###############################################################################

def test_rank_members_best_first():
    assert rank_members([0, 1, 2, 3], [1.0, 4.0, 2.0, 3.0]) == [1, 3, 2, 0]


def test_rank_members_only_ranks_the_given_members():
    assert rank_members([3, 1], [9.0, 2.0, 9.0, 5.0]) == [3, 1]


def test_rank_members_ties_go_to_the_lower_index():
    assert rank_members([2, 0, 1], [5.0, 5.0, 5.0]) == [0, 1, 2]


def test_rank_members_negative_fitness():
    assert rank_members([0, 1, 2], [-3.0, -1.0, -2.0]) == [1, 2, 0]


def test_rank_members_returns_python_ints_and_does_not_modify():
    members = [0, 1, 2]
    fitnesses = np.array([1.0, 3.0, 2.0])
    ranked = rank_members(members, fitnesses)
    assert members == [0, 1, 2]
    for index in ranked:
        assert type(index) is int


def test_rank_members_nan_raises():
    try:
        rank_members([0, 1], [1.0, float("nan")])
    except ValueError:
        return
    raise AssertionError("NaN fitness must raise ValueError")

###############################################################################
# parent_pool
###############################################################################

def test_parent_pool_is_the_top_fraction():
    ranked = list(range(10))
    cfg = default_config()
    cfg["survival_threshold"] = 0.5
    assert parent_pool(ranked, cfg) == [0, 1, 2, 3, 4]


def test_parent_pool_rounds_up():
    cfg = default_config()
    cfg["survival_threshold"] = 0.25
    assert parent_pool(list(range(10)), cfg) == [0, 1, 2]       # 2.5 -> 3


def test_parent_pool_has_at_least_two_parents():
    cfg = default_config()
    cfg["survival_threshold"] = 0.2
    assert parent_pool(list(range(3)), cfg) == [0, 1]           # ceil(0.6) = 1 -> 2
    cfg["survival_threshold"] = 0.0
    assert parent_pool(list(range(8)), cfg) == [0, 1]


def test_parent_pool_of_a_tiny_species():
    cfg = default_config()
    assert parent_pool([4], cfg) == [4]
    assert parent_pool([4, 7], cfg) == [4, 7]
    assert parent_pool([], cfg) == []


def test_parent_pool_never_exceeds_the_species():
    cfg = default_config()
    cfg["survival_threshold"] = 1.0
    assert parent_pool(list(range(6)), cfg) == list(range(6))
    cfg["survival_threshold"] = 5.0
    assert parent_pool(list(range(6)), cfg) == list(range(6))


def test_parent_pool_is_not_fooled_by_float_error():
    # 0.07 * 100 is 7.000000000000001 and 0.14 * 50 is 7.000000000000001:
    # a plain ceil would give 8, the right answer is 7
    cfg = default_config()
    cfg["survival_threshold"] = 0.07
    assert len(parent_pool(list(range(100)), cfg)) == 7
    cfg["survival_threshold"] = 0.14
    assert len(parent_pool(list(range(50)), cfg)) == 7
    assert len(parent_pool(list(range(150)), cfg)) == 21
    cfg["survival_threshold"] = 0.2
    assert len(parent_pool(list(range(150)), cfg)) == 30


def test_parent_pool_default_cfg_and_no_modification():
    ranked = list(range(20))
    pool = parent_pool(ranked)
    assert pool == [0, 1, 2, 3]                                  # default survival_threshold 0.2
    assert ranked == list(range(20))
    assert pool is not ranked

###############################################################################
# find_champion
###############################################################################

def test_champion_is_the_best_of_all_species():
    species_set = make_set({0: [0, 1], 1: [2, 3]})
    assert find_champion(species_set, [1.0, 2.0, 9.0, 3.0]) == (2, 1)


def test_champion_tie_goes_to_the_lowest_index():
    species_set = make_set({0: [3, 1], 1: [0, 2]})
    assert find_champion(species_set, [5.0, 5.0, 5.0, 5.0]) == (0, 1)


def test_champion_ignores_genomes_outside_the_species_set():
    species_set = make_set({0: [0], 1: [1]})
    assert find_champion(species_set, [1.0, 2.0, 100.0]) == (1, 1)


def test_champion_negative_fitness_and_nan():
    species_set = make_set({0: [0, 1]})
    assert find_champion(species_set, [-5.0, -2.0]) == (1, 0)
    try:
        find_champion(species_set, [1.0, float("nan")])
    except ValueError:
        return
    raise AssertionError("NaN fitness must raise ValueError")

###############################################################################
# ensure_champion_slot
###############################################################################

def test_counts_are_kept_when_the_champion_species_has_children():
    species_set = make_set({0: [0], 1: [1]})
    counts = {0: 3, 1: 7}
    assert ensure_champion_slot(species_set, [1.0, 9.0], counts) == counts


def test_a_slot_is_taken_from_the_biggest_species():
    species_set = make_set({0: [0], 1: [1], 2: [2]})
    counts = {0: 4, 1: 0, 2: 6}
    result = ensure_champion_slot(species_set, [1.0, 9.0, 2.0], counts)   # champion in species 1
    assert result == {0: 4, 1: 1, 2: 5}


def test_biggest_species_tie_goes_to_the_lowest_id():
    species_set = make_set({0: [0], 1: [1], 2: [2]})
    counts = {0: 5, 1: 5, 2: 0}
    result = ensure_champion_slot(species_set, [1.0, 2.0, 9.0], counts)
    assert result == {0: 4, 1: 5, 2: 1}


def test_the_total_does_not_change():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(2, 8))
        species_set = make_set({s: [s] for s in range(n)})
        fitnesses = [float(v) for v in rng.uniform(0, 10, size=n)]
        counts = {s: int(rng.integers(0, 6)) for s in range(n)}
        if sum(counts.values()) == 0:
            counts[0] = 1
        result = ensure_champion_slot(species_set, fitnesses, counts)
        assert sum(result.values()) == sum(counts.values())
        champion_species = find_champion(species_set, fitnesses)[1]
        assert result[champion_species] >= 1
        assert all(v >= 0 for v in result.values())


def test_ensure_champion_slot_does_not_modify_counts():
    species_set = make_set({0: [0], 1: [1]})
    counts = {0: 5, 1: 0}
    snapshot = dict(counts)
    result = ensure_champion_slot(species_set, [1.0, 9.0], counts)
    assert counts == snapshot
    assert result is not counts

###############################################################################
# reproduce: size and elites
###############################################################################

def small_world(sizes=(5, 5)):
    """Species 0.. with the given sizes, genome weights 1.0, 2.0, 3.0, ... and
    fitness equal to the weight, so the best genome of a species is its last."""
    members = {}
    genomes = []
    for species_id, size in enumerate(sizes):
        members[species_id] = list(range(len(genomes), len(genomes) + size))
        for _ in range(size):
            genomes.append(tagged(float(len(genomes) + 1)))
    fitnesses = [g["connections"][0]["weight"] for g in genomes]
    return genomes, fitnesses, make_set(members), init_innovations(1, 1)


def test_population_size_is_the_sum_of_the_counts():
    genomes, fitnesses, species_set, innovations = small_world((5, 5, 5))
    rng = np.random.default_rng(2)
    for counts in ({0: 3, 1: 3, 2: 4}, {0: 10, 1: 0, 2: 0}, {0: 1, 1: 1, 2: 1}):
        population = reproduce(genomes, fitnesses, species_set, counts, innovations, rng, no_mutation())
        assert len(population) == sum(counts.values())


def test_the_best_members_are_copied_unchanged_as_elites():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    cfg = always_mutates(elitism_count=2, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(3)
    population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, cfg)
    # every bred child is mutated, so the exact copies are the elites:
    # the 2 best of species 0 (weights 5, 4) and the 2 best of species 1 (10, 9)
    exact = [g for g in population if g in genomes]
    assert sorted(weights(exact)) == [4.0, 5.0, 9.0, 10.0]
    assert len(population) == 10


def test_elites_count_as_children_of_the_species():
    genomes, fitnesses, species_set, innovations = small_world((6,))
    cfg = always_mutates(elitism_count=1, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(4)
    population = reproduce(genomes, fitnesses, species_set, {0: 3}, innovations, rng, cfg)
    assert len(population) == 3
    assert n_exact_copies(population, genomes) == 1


def test_elites_are_never_more_than_the_children_of_the_species():
    genomes, fitnesses, species_set, innovations = small_world((6, 6))
    cfg = always_mutates(elitism_count=3, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(5)
    population = reproduce(genomes, fitnesses, species_set, {0: 2, 1: 8}, innovations, rng, cfg)
    assert len(population) == 10
    # species 0: only 2 children -> 2 elites; species 1: 3 elites
    assert n_exact_copies(population, genomes) == 5


def test_small_species_get_no_elites():
    # species 0 has 2 members, below elitism_min_species_size = 3; the champion
    # (weight 7) is in species 1, so species 0 has no reason to keep anyone
    genomes, fitnesses, species_set, innovations = small_world((2, 5))
    cfg = always_mutates(elitism_count=1, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(6)
    for _ in range(20):
        population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, cfg)
        assert n_exact_copies(population, [genomes[0], genomes[1]]) == 0
        assert n_exact_copies(population, genomes[2:]) == 1      # the elite of species 1


def test_elitism_count_zero_means_no_elites():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    cfg = always_mutates(elitism_count=0, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(7)
    population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, cfg)
    # only the champion (weight 10) is kept, because it always survives
    assert n_exact_copies(population, genomes) == 1
    assert genomes[9] in population

###############################################################################
# reproduce: the champion
###############################################################################

def test_the_champion_survives_even_when_its_species_gets_no_children():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    cfg = always_mutates(mate_only_prob=0.0)
    rng = np.random.default_rng(8)
    population = reproduce(genomes, fitnesses, species_set, {0: 10, 1: 0}, innovations, rng, cfg)
    assert len(population) == 10
    assert genomes[9] in population


def test_the_champion_survives_in_a_species_below_the_elitism_size():
    genomes, fitnesses, species_set, innovations = small_world((5, 2))
    cfg = always_mutates(elitism_count=1, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(9)
    fitnesses = list(fitnesses)
    fitnesses[6] = 100.0                      # champion is the best of the 2-member species
    population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, cfg)
    assert genomes[6] in population


def test_the_champion_is_not_duplicated_when_it_is_already_an_elite():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    cfg = always_mutates(elitism_count=1, elitism_min_species_size=3, mate_only_prob=0.0)
    rng = np.random.default_rng(10)
    population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, cfg)
    assert sum(1 for g in population if g == genomes[9]) == 1

###############################################################################
# reproduce: who becomes a parent
###############################################################################

def test_children_come_only_from_the_parent_pool():
    genomes, fitnesses, species_set, innovations = small_world((10,))
    # pool = top ceil(0.2 * 10) = 2 -> weights 10 and 9
    cfg = no_mutation(survival_threshold=0.2, elitism_count=0, mate_only_prob=1.0)
    rng = np.random.default_rng(11)
    population = reproduce(genomes, fitnesses, species_set, {0: 10}, innovations, rng, cfg)
    assert set(weights(population)) <= {10.0, 9.0}


def test_the_whole_pool_is_used():
    genomes, fitnesses, species_set, innovations = small_world((10,))
    cfg = no_mutation(survival_threshold=0.5, elitism_count=0, mate_only_prob=1.0)
    rng = np.random.default_rng(12)
    seen = set()
    for _ in range(40):
        population = reproduce(genomes, fitnesses, species_set, {0: 10}, innovations, rng, cfg)
        seen |= set(weights(population))
    assert seen == {6.0, 7.0, 8.0, 9.0, 10.0}


def test_the_two_parents_are_different_genomes():
    # with 2 pool members and mate-only, every child mixes one gene from each
    # parent or equals a parent; with identical parents it would always equal one
    genomes, fitnesses, species_set, innovations = small_world((4,))
    cfg = no_mutation(survival_threshold=0.5, elitism_count=0, mate_only_prob=1.0)
    rng = np.random.default_rng(13)
    for _ in range(30):
        population = reproduce(genomes, fitnesses, species_set, {0: 8}, innovations, rng, cfg)
        assert set(weights(population)) <= {3.0, 4.0}
    # both weights appear: the child gene is drawn from either parent
    all_weights = set()
    for _ in range(30):
        all_weights |= set(weights(reproduce(genomes, fitnesses, species_set, {0: 8},
                                            innovations, rng, cfg)))
    assert all_weights == {3.0, 4.0}


def test_the_fitter_parent_gives_the_structure():
    # genome 3 (best) has extra gene 1, genome 2 has extra gene 2
    genomes = [tagged(1.0), tagged(2.0), tagged(3.0, extra_keys=(2,)), tagged(4.0, extra_keys=(1,))]
    fitnesses = [1.0, 2.0, 3.0, 4.0]
    species_set = make_set({0: [0, 1, 2, 3]})
    cfg = no_mutation(survival_threshold=0.5, elitism_count=0, mate_only_prob=1.0)
    rng = np.random.default_rng(14)
    for _ in range(30):
        population = reproduce(genomes, fitnesses, species_set, {0: 8}, init_innovations(1, 1), rng, cfg)
        for child in population:
            # pool = genomes 3 and 2, so the child has the genes of genome 3: {0, 1}
            assert sorted(child["connections"]) == [0, 1]


def test_equal_fitness_picks_the_dominant_parent_at_random():
    genomes = [tagged(1.0, extra_keys=(1,)), tagged(2.0, extra_keys=(2,))]
    fitnesses = [5.0, 5.0]
    species_set = make_set({0: [0, 1]})
    cfg = no_mutation(survival_threshold=1.0, elitism_count=0, mate_only_prob=1.0)
    rng = np.random.default_rng(15)
    population = reproduce(genomes, fitnesses, species_set, {0: 200}, init_innovations(1, 1), rng, cfg)
    with_gene_1 = sum(1 for child in population if 1 in child["connections"])
    with_gene_2 = sum(1 for child in population if 2 in child["connections"])
    assert with_gene_1 + with_gene_2 == 200
    assert 60 < with_gene_1 < 140

###############################################################################
# reproduce: mutation
###############################################################################

def test_mate_only_children_are_not_mutated():
    genomes, fitnesses, species_set, innovations = small_world((6,))
    cfg = always_mutates(elitism_count=0, mate_only_prob=1.0, survival_threshold=1.0)
    rng = np.random.default_rng(16)
    population = reproduce(genomes, fitnesses, species_set, {0: 20}, innovations, rng, cfg)
    # without mutation every weight is still one of the parents' weights
    assert set(weights(population)) <= set(weights(genomes))


def test_other_children_are_mutated():
    genomes, fitnesses, species_set, innovations = small_world((6,))
    cfg = always_mutates(elitism_count=0, mate_only_prob=0.0, survival_threshold=1.0)
    rng = np.random.default_rng(17)
    population = reproduce(genomes, fitnesses, species_set, {0: 20}, innovations, rng, cfg)
    originals = set(weights(genomes))
    mutated = [w for w in weights(population) if w not in originals]
    # the champion (weight 6) is kept unchanged, everything else was nudged
    assert len(mutated) == 19


def test_mate_only_prob_is_respected():
    genomes, fitnesses, species_set, innovations = small_world((6,))
    cfg = always_mutates(elitism_count=0, mate_only_prob=0.3, survival_threshold=1.0)
    rng = np.random.default_rng(18)
    population = reproduce(genomes, fitnesses, species_set, {0: 1000}, innovations, rng, cfg)
    originals = set(weights(genomes))
    unmutated = sum(1 for w in weights(population) if w in originals)
    assert 250 < unmutated < 350


def test_a_pool_of_one_makes_mutated_clones():
    genomes = [tagged(1.0)]
    species_set = make_set({0: [0]})
    cfg = always_mutates(elitism_count=0, mate_only_prob=1.0)
    rng = np.random.default_rng(19)
    population = reproduce(genomes, [1.0], species_set, {0: 6}, init_innovations(1, 1), rng, cfg)
    assert len(population) == 6
    # the champion is kept (one exact copy); the other five are mutated clones
    assert n_exact_copies(population, genomes) == 1
    for child in population:
        assert sorted(child["connections"]) == [0]


def test_new_structure_is_registered_in_the_shared_registry():
    genomes, fitnesses, species_set, innovations = small_world((6, 6))
    cfg = no_mutation(node_add_prob=1.0, elitism_count=0, mate_only_prob=0.0)
    rng = np.random.default_rng(20)
    before = innovations["next_node_id"]
    population = reproduce(genomes, fitnesses, species_set, {0: 6, 1: 6}, innovations, rng, cfg)
    assert innovations["next_node_id"] == before + 1      # one connection, one node id
    assert len(innovations["splits"]) == 1
    assert max(len(g["nodes"]) for g in population) == 3

###############################################################################
# reproduce: counts
###############################################################################

def test_species_with_zero_children_leave_no_trace():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    cfg = always_mutates(elitism_count=1, mate_only_prob=0.0)
    rng = np.random.default_rng(21)
    # the champion is in species 1, so species 0 (zero children) must stay absent
    population = reproduce(genomes, fitnesses, species_set, {0: 0, 1: 10}, innovations, rng, cfg)
    assert len(population) == 10
    assert n_exact_copies(population, genomes[:5]) == 0


def test_the_slot_for_the_champion_comes_out_of_the_biggest_species():
    genomes, fitnesses, species_set, innovations = small_world((5, 5, 5))
    cfg = always_mutates(elitism_count=0, mate_only_prob=0.0)
    rng = np.random.default_rng(22)
    # the champion (genome 14) is in species 2, which got 0 children: it takes
    # one child away from species 0 (the biggest, lowest id) and keeps itself
    population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5, 2: 0}, innovations,
                           rng, cfg)
    assert len(population) == 10
    assert sum(1 for g in population if g == genomes[14]) == 1
    # species 2 has exactly one child (the champion): no other genome of species 2
    assert n_exact_copies(population, genomes[10:14]) == 0


def test_counts_must_match_the_species_ids():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    rng = np.random.default_rng(23)
    for bad in ({0: 10}, {0: 5, 1: 5, 2: 0}, {0: 5, 2: 5}):
        try:
            reproduce(genomes, fitnesses, species_set, bad, innovations, rng, no_mutation())
        except ValueError:
            continue
        raise AssertionError(f"counts {bad} must raise ValueError")


def test_negative_counts_raise():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    rng = np.random.default_rng(24)
    try:
        reproduce(genomes, fitnesses, species_set, {0: 12, 1: -2}, innovations, rng, no_mutation())
    except ValueError:
        return
    raise AssertionError("a negative count must raise ValueError")


def test_nan_fitness_raises():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    fitnesses = list(fitnesses)
    fitnesses[2] = float("nan")
    rng = np.random.default_rng(25)
    try:
        reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, no_mutation())
    except ValueError:
        return
    raise AssertionError("NaN fitness must raise ValueError")

###############################################################################
# copies, purity, randomness
###############################################################################

def test_inputs_are_not_modified():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    counts = {0: 5, 1: 5}
    cfg = always_mutates(node_add_prob=0.5, conn_add_prob=0.5)
    snaps = (copy.deepcopy(genomes), list(fitnesses), copy.deepcopy(species_set),
             dict(counts), copy.deepcopy(cfg))
    rng = np.random.default_rng(26)
    for _ in range(10):
        reproduce(genomes, fitnesses, species_set, counts, copy.deepcopy(innovations), rng, cfg)
    assert (genomes, fitnesses, species_set, counts, cfg) == snaps


def test_children_share_nothing_with_the_parents():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    cfg = no_mutation(elitism_count=2, mate_only_prob=1.0)
    rng = np.random.default_rng(27)
    population = reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations, rng, cfg)
    snapshot = copy.deepcopy(genomes)
    for child in population:
        for parent in genomes:
            assert child is not parent
            assert child["nodes"] is not parent["nodes"]
            assert child["connections"] is not parent["connections"]
        child["connections"][0]["weight"] = 999.0
    assert genomes == snapshot


def test_rng_and_innovations_are_required():
    genomes, fitnesses, species_set, innovations = small_world((5, 5))
    rng = np.random.default_rng(28)
    for call in (lambda: reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations),
                 lambda: reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, rng=rng)):
        try:
            call()
        except TypeError:
            continue
        raise AssertionError("reproduce must require innovations and rng")


def test_same_seed_gives_the_same_population():
    results = []
    for _ in range(2):
        genomes, fitnesses, species_set, innovations = small_world((5, 5))
        cfg = cfg_with(node_add_prob=0.5, conn_add_prob=0.5, weight_mutate_prob=0.8)
        rng = np.random.default_rng(29)
        results.append((reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations,
                                  rng, cfg), innovations))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_different_seeds_give_different_populations():
    outcomes = []
    for seed in (30, 31):
        genomes, fitnesses, species_set, innovations = small_world((5, 5))
        cfg = cfg_with(weight_mutate_prob=0.8)
        outcomes.append(reproduce(genomes, fitnesses, species_set, {0: 5, 1: 5}, innovations,
                                  np.random.default_rng(seed), cfg))
    assert outcomes[0] != outcomes[1]

###############################################################################
# a few generations together
###############################################################################

def toy_fitness(genome):
    """A deterministic function of the genome: sum of the weights of its
    enabled connections, plus a small bonus per hidden node."""
    total = 0.0
    for c in genome["connections"].values():
        if c["enabled"]:
            total += c["weight"]
    for node in genome["nodes"].values():
        if node["type"] == "hidden":
            total += 0.1
    return total


def run_generations(n_generations, seed):
    rng = np.random.default_rng(seed)
    n_in, n_out, size = 4, 2, 40
    cfg = default_config()
    cfg["population_size"] = size
    cfg["compatibility_threshold"] = 1.5
    cfg["node_add_prob"] = 0.15
    cfg["conn_add_prob"] = 0.25
    innovations = init_innovations(n_in, n_out)
    population = [init_genome(n_in, n_out, rng) for _ in range(size)]
    species_set = init_species()
    best = []
    species_counts = []
    for _ in range(n_generations):
        fitnesses = [toy_fitness(g) for g in population]
        best.append(max(fitnesses))
        species_set = speciate(population, species_set, cfg)
        species_counts.append(len(species_set["species"]))
        counts = compute_offspring(species_set, fitnesses, cfg)
        population = reproduce(population, fitnesses, species_set, counts, innovations, rng, cfg)
        assert len(population) == size
    return population, innovations, best, species_counts


def test_the_best_fitness_never_goes_down():
    for seed in (1, 2, 3):
        population, innovations, best, species_counts = run_generations(25, seed)
        for before, after in zip(best, best[1:]):
            assert after >= before


def test_the_population_stays_valid_over_generations():
    population, innovations, best, species_counts = run_generations(25, 4)
    for genome in population:
        assert is_acyclic(genome)
        for key, c in genome["connections"].items():
            assert c["src"] in genome["nodes"] and c["dst"] in genome["nodes"]
            assert genome["nodes"][c["dst"]]["type"] != "input"
            assert innovations["connections"][(c["src"], c["dst"])] == key
    assert max(len(g["nodes"]) for g in population) > 6
    assert max(species_counts) > 1


def test_the_toy_problem_improves():
    population, innovations, best, species_counts = run_generations(30, 5)
    assert best[-1] > best[0]

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"reproduction: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()