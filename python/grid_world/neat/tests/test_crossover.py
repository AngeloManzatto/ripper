"""
Created on Mon Oct 19 10:11:04 2026

@author: Angelo Antonio Manzatto
"""


"""
Tests for neat/crossover.py: crossover(parent_a, parent_b, rng).

Crossover builds ONE child from two parents. parent_a must be the fitter
parent (or equal); the caller decides the order, because fitness is not stored
in the genome. For equal fitness the caller should pick the order at random.

What crossover must do
----------------------
Genes are matched by identity: connections by innovation number, nodes by id.

1. The child has exactly the genes of parent_a: the same connection keys and
   the same node ids. Genes that only parent_b has (disjoint or excess) are
   NOT inherited, and genes that only parent_a has are copied from parent_a.
   Because the child has the topology of parent_a, it can never contain a
   cycle that parent_a does not have.
2. A gene that BOTH parents have (matching gene) is copied whole from parent_a
   or from parent_b, with probability 1/2 each, independently for every gene.
   "Whole" means a connection keeps its weight and its enabled flag from the
   same parent, and a node keeps its bias and its activation from the same
   parent. The enabled flag is simply inherited (no extra chance to disable).
3. n_inputs and n_outputs come from parent_a. Input nodes are copied as they
   are.
4. Fail loudly when the parents cannot belong to the same registry:
   different n_inputs or n_outputs, a node id with different types, or an
   innovation number that joins different nodes -> ValueError.
5. The child shares nothing with the parents (deep copies). The parents are
   never modified. rng is required. The same seed gives the same child.

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
from neat.crossover import crossover

###############################################################################
# helpers
###############################################################################

def node(kind, bias=0.0, activation="sigmoid"):
    if kind == "input":
        return {"type": "input"}
    return {"type": kind, "bias": bias, "activation": activation}


def conn(src, dst, weight=1.0, enabled=True):
    return {"src": src, "dst": dst, "weight": weight, "enabled": enabled}


def genome_of(nodes, connections, n_inputs=1, n_outputs=1):
    return {"n_inputs": n_inputs, "n_outputs": n_outputs,
            "nodes": copy.deepcopy(nodes), "connections": copy.deepcopy(connections)}


def base_nodes():
    return {0: node("input"), 1: node("output")}


def many_matching(n, weight_a, weight_b):
    """Two parents with n matching connections 0..n-1 (all 0 -> 1)."""
    a = genome_of(base_nodes(), {k: conn(0, 1, weight_a) for k in range(n)})
    b = genome_of(base_nodes(), {k: conn(0, 1, weight_b) for k in range(n)})
    return a, b


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


def grown_population(seed=0, size=12, rounds=12):
    """A population that shares one registry and has diverged by mutation."""
    innovations = init_innovations(3, 2)
    rng = np.random.default_rng(seed)
    cfg = default_config()
    cfg["node_add_prob"] = 0.4
    cfg["conn_add_prob"] = 0.6
    genomes = [init_genome(3, 2, rng) for _ in range(size)]
    for _ in range(rounds):
        genomes = [mutate(g, rng, innovations, cfg) for g in genomes]
    return genomes

###############################################################################
# which genes the child has
###############################################################################

def test_child_has_exactly_the_connections_of_parent_a():
    a = genome_of(base_nodes(), {0: conn(0, 1), 1: conn(0, 1), 5: conn(0, 1)})
    b = genome_of(base_nodes(), {0: conn(0, 1), 2: conn(0, 1), 7: conn(0, 1)})
    rng = np.random.default_rng(1)
    child = crossover(a, b, rng)
    assert sorted(child["connections"]) == [0, 1, 5]


def test_genes_only_parent_b_has_are_not_inherited():
    a = genome_of(base_nodes(), {0: conn(0, 1)})
    b = genome_of(base_nodes(), {0: conn(0, 1), 1: conn(0, 1, 9.0), 2: conn(0, 1, 9.0)})
    rng = np.random.default_rng(2)
    for _ in range(50):
        child = crossover(a, b, rng)
        assert sorted(child["connections"]) == [0]


def test_genes_only_parent_a_has_are_copied_from_parent_a():
    a = genome_of(base_nodes(), {0: conn(0, 1), 3: conn(0, 1, -4.5, False), 8: conn(0, 1, 2.25)})
    b = genome_of(base_nodes(), {0: conn(0, 1)})
    rng = np.random.default_rng(3)
    child = crossover(a, b, rng)
    assert child["connections"][3] == a["connections"][3]
    assert child["connections"][8] == a["connections"][8]


def test_swapping_the_parents_swaps_which_genes_are_kept():
    a = genome_of(base_nodes(), {0: conn(0, 1), 1: conn(0, 1)})
    b = genome_of(base_nodes(), {0: conn(0, 1), 2: conn(0, 1)})
    rng = np.random.default_rng(4)
    assert sorted(crossover(a, b, rng)["connections"]) == [0, 1]
    assert sorted(crossover(b, a, rng)["connections"]) == [0, 2]


def test_child_has_exactly_the_nodes_of_parent_a():
    nodes_a = {0: node("input"), 1: node("output"), 2: node("hidden")}
    nodes_b = {0: node("input"), 1: node("output"), 3: node("hidden")}
    a = genome_of(nodes_a, {0: conn(0, 2), 1: conn(2, 1)})
    # b uses other innovation numbers, as a real registry would give it
    b = genome_of(nodes_b, {5: conn(0, 3), 6: conn(3, 1)})
    rng = np.random.default_rng(5)
    child = crossover(a, b, rng)
    assert sorted(child["nodes"]) == [0, 1, 2]
    assert [child["nodes"][n]["type"] for n in sorted(child["nodes"])] == ["input", "output", "hidden"]


def test_identical_parents_give_an_identical_child():
    genomes = grown_population(seed=1)
    rng = np.random.default_rng(6)
    for genome in genomes:
        assert crossover(genome, copy.deepcopy(genome), rng) == genome


def test_n_inputs_and_n_outputs_come_from_the_parents():
    a, b = many_matching(3, 1.0, 2.0)
    child = crossover(a, b, np.random.default_rng(7))
    assert child["n_inputs"] == a["n_inputs"]
    assert child["n_outputs"] == a["n_outputs"]

###############################################################################
# matching connection genes
###############################################################################

def test_a_matching_connection_comes_whole_from_one_parent():
    a = genome_of(base_nodes(), {0: conn(0, 1, 1.0, True)})
    b = genome_of(base_nodes(), {0: conn(0, 1, 2.0, False)})
    rng = np.random.default_rng(8)
    seen = set()
    for _ in range(200):
        gene = crossover(a, b, rng)["connections"][0]
        seen.add((gene["weight"], gene["enabled"]))
    # weight and enabled flag always travel together
    assert seen == {(1.0, True), (2.0, False)}


def test_each_parent_gives_about_half_of_the_matching_genes():
    a, b = many_matching(400, 1.0, 2.0)
    child = crossover(a, b, np.random.default_rng(9))
    from_a = sum(1 for g in child["connections"].values() if g["weight"] == 1.0)
    assert 160 < from_a < 240


def test_the_choice_is_made_gene_by_gene():
    a, b = many_matching(60, 1.0, 2.0)
    rng = np.random.default_rng(10)
    for _ in range(20):
        child = crossover(a, b, rng)
        weights = {g["weight"] for g in child["connections"].values()}
        assert weights == {1.0, 2.0}


def test_both_parents_are_chosen_by_both_roles():
    # the fitter parent does not always win the matching genes
    a, b = many_matching(50, 1.0, 2.0)
    rng = np.random.default_rng(11)
    totals = {1.0: 0, 2.0: 0}
    for _ in range(40):
        child = crossover(a, b, rng)
        for gene in child["connections"].values():
            totals[gene["weight"]] += 1
    assert 800 < totals[1.0] < 1200
    assert 800 < totals[2.0] < 1200


def test_enabled_flag_is_inherited_without_extra_disabling():
    both_off_a = genome_of(base_nodes(), {k: conn(0, 1, 1.0, False) for k in range(30)})
    both_off_b = genome_of(base_nodes(), {k: conn(0, 1, 2.0, False) for k in range(30)})
    both_on_a = genome_of(base_nodes(), {k: conn(0, 1, 1.0, True) for k in range(30)})
    both_on_b = genome_of(base_nodes(), {k: conn(0, 1, 2.0, True) for k in range(30)})
    rng = np.random.default_rng(12)
    for _ in range(20):
        off = crossover(both_off_a, both_off_b, rng)
        on = crossover(both_on_a, both_on_b, rng)
        assert all(g["enabled"] is False for g in off["connections"].values())
        assert all(g["enabled"] is True for g in on["connections"].values())

###############################################################################
# matching node genes
###############################################################################

def test_a_matching_node_comes_whole_from_one_parent():
    a = genome_of({0: node("input"), 1: node("output", 1.0, "sigmoid")}, {0: conn(0, 1)})
    b = genome_of({0: node("input"), 1: node("output", 2.0, "tanh")}, {0: conn(0, 1)})
    rng = np.random.default_rng(13)
    seen = set()
    for _ in range(200):
        n = crossover(a, b, rng)["nodes"][1]
        seen.add((n["bias"], n["activation"]))
    # bias and activation always travel together
    assert seen == {(1.0, "sigmoid"), (2.0, "tanh")}


def test_each_parent_gives_about_half_of_the_matching_nodes():
    nodes_a = {0: node("input"), 1: node("output", 1.0, "sigmoid")}
    nodes_b = {0: node("input"), 1: node("output", 2.0, "tanh")}
    for k in range(2, 402):
        nodes_a[k] = node("hidden", 1.0, "sigmoid")
        nodes_b[k] = node("hidden", 2.0, "tanh")
    a = genome_of(nodes_a, {0: conn(0, 1)})
    b = genome_of(nodes_b, {0: conn(0, 1)})
    child = crossover(a, b, np.random.default_rng(14))
    from_a = sum(1 for n, v in child["nodes"].items() if n >= 2 and v["bias"] == 1.0)
    assert 160 < from_a < 240


def test_input_nodes_are_copied_as_they_are():
    a, b = many_matching(3, 1.0, 2.0)
    child = crossover(a, b, np.random.default_rng(15))
    assert child["nodes"][0] == {"type": "input"}

###############################################################################
# no cycles, consistent children
###############################################################################

def check_consistent(genome):
    for key, c in genome["connections"].items():
        assert c["src"] in genome["nodes"]
        assert c["dst"] in genome["nodes"]
        assert genome["nodes"][c["dst"]]["type"] != "input"
    assert is_acyclic(genome)


def test_children_of_a_diverged_population_are_acyclic_and_consistent():
    genomes = grown_population(seed=2, size=14, rounds=15)
    rng = np.random.default_rng(16)
    checked = 0
    for i in range(len(genomes)):
        for j in range(len(genomes)):
            if i == j:
                continue
            child = crossover(genomes[i], genomes[j], rng)
            check_consistent(child)
            assert sorted(child["connections"]) == sorted(genomes[i]["connections"])
            assert sorted(child["nodes"]) == sorted(genomes[i]["nodes"])
            checked += 1
    assert checked > 100
    assert max(len(g["nodes"]) for g in genomes) > 8

###############################################################################
# fail loudly
###############################################################################

def test_different_number_of_inputs_raises():
    a = genome_of(base_nodes(), {0: conn(0, 1)}, n_inputs=1)
    b = genome_of(base_nodes(), {0: conn(0, 1)}, n_inputs=2)
    try:
        crossover(a, b, np.random.default_rng(17))
    except ValueError:
        return
    raise AssertionError("different n_inputs must raise ValueError")


def test_different_number_of_outputs_raises():
    a = genome_of(base_nodes(), {0: conn(0, 1)}, n_outputs=1)
    b = genome_of(base_nodes(), {0: conn(0, 1)}, n_outputs=3)
    try:
        crossover(a, b, np.random.default_rng(18))
    except ValueError:
        return
    raise AssertionError("different n_outputs must raise ValueError")


def test_same_node_id_with_different_types_raises():
    a = genome_of({0: node("input"), 1: node("output"), 2: node("hidden")}, {0: conn(0, 1)})
    b = genome_of({0: node("input"), 1: node("output"), 2: node("output")}, {0: conn(0, 1)})
    try:
        crossover(a, b, np.random.default_rng(19))
    except ValueError:
        return
    raise AssertionError("a node with different types must raise ValueError")


def test_same_innovation_between_different_nodes_raises():
    nodes = {0: node("input"), 1: node("output"), 2: node("hidden")}
    a = genome_of(nodes, {0: conn(0, 1)})
    b = genome_of(nodes, {0: conn(0, 2)})
    try:
        crossover(a, b, np.random.default_rng(20))
    except ValueError:
        return
    raise AssertionError("an innovation joining different nodes must raise ValueError")

###############################################################################
# copies, purity, randomness
###############################################################################

def test_parents_are_never_modified():
    genomes = grown_population(seed=3)
    a, b = genomes[0], genomes[1]
    snap_a, snap_b = copy.deepcopy(a), copy.deepcopy(b)
    rng = np.random.default_rng(21)
    for _ in range(30):
        crossover(a, b, rng)
    assert a == snap_a and b == snap_b


def test_child_shares_nothing_with_the_parents():
    a, b = many_matching(5, 1.0, 2.0)
    child = crossover(a, b, np.random.default_rng(22))
    assert child["nodes"] is not a["nodes"] and child["nodes"] is not b["nodes"]
    assert child["connections"] is not a["connections"]
    assert child["connections"] is not b["connections"]
    for key, gene in child["connections"].items():
        assert gene is not a["connections"][key]
        assert gene is not b["connections"][key]
    for n, content in child["nodes"].items():
        assert content is not a["nodes"][n]
        assert content is not b["nodes"][n]


def test_changing_the_child_does_not_change_the_parents():
    a, b = many_matching(5, 1.0, 2.0)
    snap_a, snap_b = copy.deepcopy(a), copy.deepcopy(b)
    child = crossover(a, b, np.random.default_rng(23))
    for gene in child["connections"].values():
        gene["weight"] = 99.0
    for content in child["nodes"].values():
        content["extra"] = True
    assert a == snap_a and b == snap_b


def test_rng_is_required():
    a, b = many_matching(3, 1.0, 2.0)
    try:
        crossover(a, b)
    except TypeError:
        return
    raise AssertionError("crossover must require rng")


def test_same_seed_gives_the_same_child():
    a, b = many_matching(50, 1.0, 2.0)
    first = crossover(a, b, np.random.default_rng(24))
    second = crossover(a, b, np.random.default_rng(24))
    assert first == second


def test_different_seeds_give_different_children():
    a, b = many_matching(50, 1.0, 2.0)
    first = crossover(a, b, np.random.default_rng(25))
    second = crossover(a, b, np.random.default_rng(26))
    assert first != second


def test_connection_keys_are_python_ints():
    genomes = grown_population(seed=4)
    child = crossover(genomes[0], genomes[1], np.random.default_rng(27))
    for key in child["connections"]:
        assert type(key) is int
    for node_id in child["nodes"]:
        assert type(node_id) is int

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"crossover: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()