"""
Created on Mon Oct 19 08:27:09 2026

@author: Angelo Antonio Manzatto
"""


"""
Tests for neat/compatibility.py: compatibility_distance(genome_a, genome_b, cfg=None).

The compatibility distance is the original NEAT formula. It looks ONLY at the
connection genes, matched by innovation number:

    distance = c1 * E / N  +  c2 * D / N  +  c3 * W

  matching  gene in both genomes (same innovation number)
  excess    gene in one genome only, with a number HIGHER than the biggest
            number of the other genome
  disjoint  gene in one genome only, with a number inside the other genome's
            range
  E, D      how many excess / disjoint genes (counting both genomes)
  W         average |weight_a - weight_b| over the matching genes
            (0.0 if there are no matching genes)
  N         number of connection genes of the BIGGER genome (disabled genes
            count); N = 1 when that number is smaller than cfg["compatibility_n_min"]
  c1, c2, c3, n_min   cfg["compatibility_c1"], ["compatibility_c2"],
                      ["compatibility_c3"], ["compatibility_n_min"]

Decisions fixed by these tests:
  - the enabled flag does NOT matter, only the weights of matching genes;
  - nodes (bias, activation) do NOT matter, only connections;
  - the result is a plain Python float, symmetric, and 0.0 for equal genomes.

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
from neat.mutation import add_node, mutate
from neat.compatibility import compatibility_distance

###############################################################################
# helpers
###############################################################################

def genome_with(weights, enabled=None):
    """A genome that only has connections: {innovation: weight}.

    `enabled` is an optional {innovation: bool}; everything else is enabled.
    Only the connections matter for the distance.
    """
    enabled = enabled or {}
    connections = {}
    for innovation, weight in weights.items():
        connections[innovation] = {"src": 0, "dst": 1, "weight": weight,
                                   "enabled": enabled.get(innovation, True)}
    return {"n_inputs": 1, "n_outputs": 1, "nodes": {}, "connections": connections}


def cfg_with(c1=1.0, c2=1.0, c3=0.4, n_min=20):
    cfg = default_config()
    cfg["compatibility_c1"] = c1
    cfg["compatibility_c2"] = c2
    cfg["compatibility_c3"] = c3
    cfg["compatibility_n_min"] = n_min
    return cfg


def close(a, b):
    return abs(a - b) < 1e-9

###############################################################################
# equal genomes
###############################################################################

def test_a_genome_is_at_distance_zero_from_itself():
    rng = np.random.default_rng(1)
    genome = init_genome(5, 3, rng)
    assert compatibility_distance(genome, genome) == 0.0


def test_a_copy_is_at_distance_zero():
    rng = np.random.default_rng(2)
    genome = init_genome(5, 3, rng)
    assert compatibility_distance(genome, copy.deepcopy(genome)) == 0.0


def test_two_empty_genomes_are_at_distance_zero():
    assert compatibility_distance(genome_with({}), genome_with({})) == 0.0


def test_two_empty_genomes_even_with_n_min_zero():
    cfg = cfg_with(n_min=0)
    assert compatibility_distance(genome_with({}), genome_with({}), cfg) == 0.0

###############################################################################
# weight term only
###############################################################################

def test_only_weight_differences():
    a = genome_with({0: 1.0, 1: 2.0})
    b = genome_with({0: 2.0, 1: 0.0})
    # |1-2| = 1, |2-0| = 2 -> average 1.5
    cfg = cfg_with(c3=0.4)
    assert close(compatibility_distance(a, b, cfg), 0.4 * 1.5)


def test_weight_difference_is_an_absolute_value():
    a = genome_with({0: 1.0})
    b = genome_with({0: 4.0})
    cfg = cfg_with(c3=1.0)
    assert close(compatibility_distance(a, b, cfg), 3.0)
    assert close(compatibility_distance(b, a, cfg), 3.0)


def test_weight_term_is_an_average_not_a_sum():
    a = genome_with({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    b = genome_with({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    cfg = cfg_with(c3=1.0)
    assert close(compatibility_distance(a, b, cfg), 1.0)


def test_weight_term_ignores_genes_that_do_not_match():
    a = genome_with({0: 1.0, 1: 5.0})
    b = genome_with({0: 2.0, 2: 9.0})
    # only gene 0 matches: average difference 1.0; genes 1 and 2 are not weighed
    cfg = cfg_with(c1=0.0, c2=0.0, c3=1.0)
    assert close(compatibility_distance(a, b, cfg), 1.0)


def test_c3_zero_ignores_weights():
    a = genome_with({0: 1.0, 1: 2.0})
    b = genome_with({0: 9.0, 1: -9.0})
    assert compatibility_distance(a, b, cfg_with(c3=0.0)) == 0.0

###############################################################################
# excess and disjoint
###############################################################################

def test_excess_genes():
    a = genome_with({0: 1.0, 1: 1.0, 2: 1.0})
    b = genome_with({0: 1.0, 1: 1.0})
    # gene 2 is above b's biggest number (1): excess. Small genomes: N = 1
    assert close(compatibility_distance(a, b, cfg_with(c1=2.0, c2=5.0)), 2.0)


def test_disjoint_genes():
    a = genome_with({0: 1.0, 2: 1.0})
    b = genome_with({0: 1.0, 1: 1.0, 2: 1.0})
    # gene 1 is inside a's range (0..2): disjoint
    assert close(compatibility_distance(a, b, cfg_with(c1=5.0, c2=3.0)), 3.0)


def test_excess_and_disjoint_use_their_own_coefficients():
    a = genome_with({0: 1.0, 1: 1.0, 3: 1.0})
    b = genome_with({0: 1.0, 2: 1.0, 4: 1.0, 5: 1.0})
    # matching: 0. a only: 1 (disjoint), 3 (disjoint). b only: 2 (disjoint),
    # 4 and 5 (excess).  E = 2, D = 3, W = 0.  N = 1 (small genomes)
    cfg = cfg_with(c1=1.0, c2=10.0, c3=0.0)
    assert close(compatibility_distance(a, b, cfg), 1.0 * 2 + 10.0 * 3)


def test_everything_together():
    a = genome_with({0: 1.0, 1: 1.0, 3: 1.0})
    b = genome_with({0: 2.0, 2: 1.0, 4: 1.0, 5: 1.0})
    # E = 2, D = 3, W = |1-2| = 1.  N = max(3, 4) = 4 with n_min = 1
    cfg = cfg_with(c1=1.0, c2=2.0, c3=0.5, n_min=1)
    expected = 1.0 * 2 / 4 + 2.0 * 3 / 4 + 0.5 * 1.0
    assert close(compatibility_distance(a, b, cfg), expected)


def test_excess_is_counted_in_both_directions():
    a = genome_with({0: 1.0})
    b = genome_with({0: 1.0, 1: 1.0, 2: 1.0})
    cfg = cfg_with(c1=1.0, c2=0.0, c3=0.0)
    assert close(compatibility_distance(a, b, cfg), 2.0)
    assert close(compatibility_distance(b, a, cfg), 2.0)


def test_genome_with_no_genes_makes_everything_excess():
    empty = genome_with({})
    other = genome_with({0: 1.0, 1: 1.0, 2: 1.0})
    cfg = cfg_with(c1=1.0, c2=100.0, c3=100.0)
    assert close(compatibility_distance(empty, other, cfg), 3.0)
    assert close(compatibility_distance(other, empty, cfg), 3.0)


def test_gene_zero_against_an_empty_genome_is_excess():
    empty = genome_with({})
    other = genome_with({0: 1.0})
    cfg = cfg_with(c1=1.0, c2=100.0, c3=0.0)
    assert close(compatibility_distance(empty, other, cfg), 1.0)

###############################################################################
# normalisation by N
###############################################################################

def test_small_genomes_are_not_normalised():
    a = genome_with({i: 1.0 for i in range(10)})
    b = genome_with({i: 1.0 for i in range(8)})
    # 10 genes < n_min 20 -> N = 1;  E = 2
    assert close(compatibility_distance(a, b, cfg_with(c1=1.0)), 2.0)


def test_big_genomes_are_divided_by_the_bigger_genome():
    a = genome_with({i: 1.0 for i in range(25)})
    b = genome_with({i: 1.0 for i in range(23)})
    # 25 genes >= n_min 20 -> N = 25;  E = 2
    assert close(compatibility_distance(a, b, cfg_with(c1=1.0)), 2 / 25)


def test_n_is_the_bigger_genome_not_the_smaller_one():
    a = genome_with({i: 1.0 for i in range(30)})
    b = genome_with({i: 1.0 for i in range(10)})
    # E = 20, N = 30 (not 10)
    assert close(compatibility_distance(a, b, cfg_with(c1=1.0)), 20 / 30)


def test_n_min_boundary():
    a = genome_with({0: 1.0, 1: 1.0, 2: 1.0})
    b = genome_with({0: 1.0, 1: 1.0})
    # E = 1, the bigger genome has 3 genes
    assert close(compatibility_distance(a, b, cfg_with(c1=1.0, n_min=3)), 1 / 3)  # 3 is not < 3
    assert close(compatibility_distance(a, b, cfg_with(c1=1.0, n_min=4)), 1.0)    # 3 < 4 -> N = 1


def test_disabled_genes_count_for_n():
    a = genome_with({i: 1.0 for i in range(25)}, enabled={i: False for i in range(25)})
    b = genome_with({i: 1.0 for i in range(23)})
    assert close(compatibility_distance(a, b, cfg_with(c1=1.0)), 2 / 25)


def test_weight_term_is_not_divided_by_n():
    a = genome_with({i: 0.0 for i in range(30)})
    b = genome_with({i: 2.0 for i in range(30)})
    assert close(compatibility_distance(a, b, cfg_with(c3=1.0)), 2.0)

###############################################################################
# what does NOT matter
###############################################################################

def test_enabled_flag_is_ignored():
    a = genome_with({0: 1.0, 1: 2.0}, enabled={0: False})
    b = genome_with({0: 1.0, 1: 2.0})
    assert compatibility_distance(a, b) == 0.0


def test_weights_of_disabled_matching_genes_still_count():
    a = genome_with({0: 1.0}, enabled={0: False})
    b = genome_with({0: 3.0}, enabled={0: False})
    assert close(compatibility_distance(a, b, cfg_with(c3=1.0)), 2.0)
    c = genome_with({0: 3.0})
    assert close(compatibility_distance(a, c, cfg_with(c3=1.0)), 2.0)


def test_nodes_are_ignored():
    rng = np.random.default_rng(3)
    a = init_genome(4, 2, rng)
    b = copy.deepcopy(a)
    for node in b["nodes"].values():
        if node["type"] != "input":
            node["bias"] += 5.0
            node["activation"] = "tanh"
    assert compatibility_distance(a, b) == 0.0

###############################################################################
# general properties
###############################################################################

def test_distance_is_symmetric():
    innovations = init_innovations(3, 2)
    rng = np.random.default_rng(4)
    genomes = [init_genome(3, 2, rng) for _ in range(6)]
    cfg = default_config()
    cfg["node_add_prob"] = 0.5
    cfg["conn_add_prob"] = 0.5
    for _ in range(15):
        genomes = [mutate(g, rng, innovations, cfg) for g in genomes]
    for i in range(len(genomes)):
        for j in range(len(genomes)):
            ab = compatibility_distance(genomes[i], genomes[j])
            ba = compatibility_distance(genomes[j], genomes[i])
            assert close(ab, ba)


def test_distance_is_never_negative():
    innovations = init_innovations(3, 2)
    rng = np.random.default_rng(5)
    genomes = [init_genome(3, 2, rng) for _ in range(6)]
    cfg = default_config()
    cfg["node_add_prob"] = 0.5
    for _ in range(10):
        genomes = [mutate(g, rng, innovations, cfg) for g in genomes]
    for a in genomes:
        for b in genomes:
            assert compatibility_distance(a, b) >= 0.0


def test_result_is_a_python_float():
    a = genome_with({0: 1.0, 1: 2.0})
    b = genome_with({0: 2.0})
    assert type(compatibility_distance(a, b)) is float
    assert type(compatibility_distance(a, a)) is float


def test_inputs_are_not_modified():
    a = genome_with({0: 1.0, 1: 2.0, 4: 3.0})
    b = genome_with({0: 2.0, 2: 2.0})
    cfg = cfg_with()
    snap_a, snap_b, snap_cfg = copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(cfg)
    compatibility_distance(a, b, cfg)
    assert a == snap_a and b == snap_b and cfg == snap_cfg

###############################################################################
# cfg
###############################################################################

def test_default_cfg_is_used_when_none():
    a = genome_with({0: 1.0})
    b = genome_with({0: 2.0, 1: 0.0})
    # defaults: c1 = 1.0, c3 = 0.4, n_min = 20.  E = 1, W = 1, N = 1
    assert close(compatibility_distance(a, b), 1.0 + 0.4)
    assert close(compatibility_distance(a, b), compatibility_distance(a, b, default_config()))


def test_every_coefficient_is_read_from_cfg():
    a = genome_with({0: 0.0, 1: 0.0, 3: 0.0})
    b = genome_with({0: 1.0, 2: 0.0, 4: 0.0})
    # matching 0 (W = 1), a only: 1 disjoint, 3 disjoint; b only: 2 disjoint, 4 excess
    # E = 1, D = 3, W = 1, N = 1
    assert close(compatibility_distance(a, b, cfg_with(c1=7.0, c2=0.0, c3=0.0)), 7.0)
    assert close(compatibility_distance(a, b, cfg_with(c1=0.0, c2=7.0, c3=0.0)), 21.0)
    assert close(compatibility_distance(a, b, cfg_with(c1=0.0, c2=0.0, c3=7.0)), 7.0)

###############################################################################
# real genomes
###############################################################################

def test_same_structure_different_weights():
    rng = np.random.default_rng(6)
    a = init_genome(4, 2, rng)
    b = init_genome(4, 2, rng)
    diffs = [abs(a["connections"][k]["weight"] - b["connections"][k]["weight"])
             for k in a["connections"]]
    expected = 0.4 * (sum(diffs) / len(diffs))
    assert close(compatibility_distance(a, b), expected)


def test_a_child_that_split_one_connection_is_two_excess_genes_away():
    rng = np.random.default_rng(7)
    parent = init_genome(3, 2, rng)
    innovations = init_innovations(3, 2)
    cfg = default_config()
    cfg["node_add_prob"] = 1.0
    child = add_node(parent, rng, innovations, cfg)
    # the split adds two new numbers above the parent's biggest; the old
    # connection is only disabled (flag ignored) and keeps its weight
    assert close(compatibility_distance(parent, child), 2.0)

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"compatibility: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()