"""
Created on Mon Oct 19 08:03:48 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/mutation.py: mutate(genome, rng, innovations, cfg=None).

mutate is only the COMBINER. The five steps are already tested one by one, so
these tests check the combination:

    1. mutate_weights      2. mutate_biases      3. mutate_activation
    4. add_node            5. add_connection

What mutate must do
-------------------
- Run the five steps in that order, giving the same cfg to all of them.
  Each step keeps its own dice, so setting one probability to 1.0 and the
  others to 0.0 isolates one step.
- Value mutations (1-3) run BEFORE the structural ones (4-5): a node or
  connection created in the same call keeps its starting values (weight 1.0
  into the new node, the old weight out of it, bias 0.0, hidden_activation).
- add_connection runs after add_node, so a new connection can use the node
  that was just created.
- Return a NEW genome; the input genome is never modified. The registry is
  changed in place, only by steps 4 and 5.
- rng and innovations are required, cfg=None means default_config().
- Same seed, same result.

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

###############################################################################
# helpers
###############################################################################

def only(**on):
    """cfg where every mutation probability is 0 except the ones in `on`."""
    cfg = default_config()
    for key in ("weight_mutate_prob", "bias_mutate_prob", "activation_mutate_prob",
                "node_add_prob", "conn_add_prob"):
        cfg[key] = 0.0
    cfg["weight_perturb_prob"] = 1.0
    cfg["bias_perturb_prob"] = 1.0
    cfg.update(on)
    return cfg


def world(n_inputs=3, n_outputs=2, seed=0):
    """A fresh minimal genome and a matching registry."""
    rng = np.random.default_rng(seed)
    genome = init_genome(n_inputs, n_outputs, rng)
    innovations = init_innovations(n_inputs, n_outputs)
    return genome, innovations


def weights_of(genome):
    return {key: conn["weight"] for key, conn in genome["connections"].items()}


def biases_of(genome):
    return {n: node["bias"] for n, node in genome["nodes"].items() if node["type"] != "input"}


def activations_of(genome):
    return {n: node["activation"] for n, node in genome["nodes"].items() if node["type"] != "input"}


def is_acyclic(genome):
    """Kahn's algorithm over ALL connections (enabled or not)."""
    incoming = {n: 0 for n in genome["nodes"]}
    for conn in genome["connections"].values():
        incoming[conn["dst"]] += 1
    ready = [n for n, count in incoming.items() if count == 0]
    removed = 0
    while ready:
        node = ready.pop()
        removed += 1
        for conn in genome["connections"].values():
            if conn["src"] == node:
                incoming[conn["dst"]] -= 1
                if incoming[conn["dst"]] == 0:
                    ready.append(conn["dst"])
    return removed == len(genome["nodes"])


def check_consistent(genome, innovations):
    """Every structural rule that must hold for any genome built by mutate."""
    ids = list(genome["nodes"].keys())
    assert len(ids) == len(set(ids))
    seen_pairs = set()
    for key, conn in genome["connections"].items():
        assert conn["src"] in genome["nodes"]
        assert conn["dst"] in genome["nodes"]
        assert genome["nodes"][conn["dst"]]["type"] != "input"
        pair = (conn["src"], conn["dst"])
        assert pair not in seen_pairs
        seen_pairs.add(pair)
        assert innovations["connections"][pair] == key
    assert is_acyclic(genome)

###############################################################################
# nothing happens when every probability is 0
###############################################################################

def test_all_probabilities_zero_changes_nothing():
    genome, innovations = world()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(1)
    for _ in range(100):
        result = mutate(genome, rng, innovations, only())
        assert result == genome
    assert innovations == snapshot


def test_returns_a_new_object_even_when_nothing_changes():
    genome, innovations = world()
    rng = np.random.default_rng(2)
    result = mutate(genome, rng, innovations, only())
    assert result is not genome
    assert result["nodes"] is not genome["nodes"]
    assert result["connections"] is not genome["connections"]

###############################################################################
# each step alone
###############################################################################

def test_only_weights():
    genome, innovations = world()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(3)
    result = mutate(genome, rng, innovations, only(weight_mutate_prob=1.0))
    before, after = weights_of(genome), weights_of(result)
    assert set(before) == set(after)
    for key in before:
        assert after[key] != before[key]
    assert biases_of(result) == biases_of(genome)
    assert activations_of(result) == activations_of(genome)
    assert set(result["nodes"]) == set(genome["nodes"])
    assert innovations == snapshot


def test_only_biases():
    genome, innovations = world()
    rng = np.random.default_rng(4)
    result = mutate(genome, rng, innovations, only(bias_mutate_prob=1.0))
    before, after = biases_of(genome), biases_of(result)
    for node in before:
        assert after[node] != before[node]
    assert weights_of(result) == weights_of(genome)
    assert activations_of(result) == activations_of(genome)


def test_only_activations():
    genome, innovations = world()
    rng = np.random.default_rng(5)
    result = mutate(genome, rng, innovations, only(activation_mutate_prob=1.0))
    before, after = activations_of(genome), activations_of(result)
    for node in before:
        assert after[node] != before[node]
    assert weights_of(result) == weights_of(genome)
    assert biases_of(result) == biases_of(genome)


def test_only_add_node():
    genome, innovations = world()
    rng = np.random.default_rng(6)
    result = mutate(genome, rng, innovations, only(node_add_prob=1.0))
    assert len(result["nodes"]) == len(genome["nodes"]) + 1
    assert len(result["connections"]) == len(genome["connections"]) + 2
    assert innovations["next_node_id"] == len(genome["nodes"]) + 1
    assert len(innovations["splits"]) == 1


def test_only_add_connection():
    # a genome with room: split one connection first, then add a connection
    genome, innovations = world()
    rng = np.random.default_rng(7)
    genome = mutate(genome, rng, innovations, only(node_add_prob=1.0))
    result = mutate(genome, rng, innovations, only(conn_add_prob=1.0))
    assert len(result["nodes"]) == len(genome["nodes"])
    assert len(result["connections"]) == len(genome["connections"]) + 1

###############################################################################
# the order of the steps
###############################################################################

def test_connection_step_runs_after_the_node_step():
    # a fully connected 3 -> 1 genome has NO room for a new connection, so the
    # only way to get a third new connection is to use the node just created
    genome, innovations = world(3, 1)
    rng = np.random.default_rng(8)
    cfg = only(node_add_prob=1.0, conn_add_prob=1.0)
    result = mutate(genome, rng, innovations, cfg)
    assert len(result["nodes"]) == len(genome["nodes"]) + 1
    assert len(result["connections"]) == len(genome["connections"]) + 3
    new_node = max(result["nodes"])
    touching = [c for key, c in result["connections"].items()
                if key not in genome["connections"]
                and (c["src"] == new_node or c["dst"] == new_node)]
    assert len(touching) == 3


def test_new_node_keeps_its_starting_values():
    # every value mutation is ON; the new node must still be untouched
    genome, innovations = world()
    rng = np.random.default_rng(9)
    cfg = only(weight_mutate_prob=1.0, bias_mutate_prob=1.0,
               activation_mutate_prob=1.0, node_add_prob=1.0,
               hidden_activation="tanh")
    for _ in range(50):
        probe = copy.deepcopy(innovations)
        result = mutate(genome, rng, probe, cfg)
        new_node = max(result["nodes"])
        assert result["nodes"][new_node]["bias"] == 0.0
        assert result["nodes"][new_node]["activation"] == "tanh"


def test_split_weights_are_exact_even_with_weight_mutation_on():
    genome, innovations = world()
    rng = np.random.default_rng(10)
    cfg = only(weight_mutate_prob=1.0, node_add_prob=1.0)
    for _ in range(50):
        probe = copy.deepcopy(innovations)
        result = mutate(genome, rng, probe, cfg)
        new_node = max(result["nodes"])
        into = [c for c in result["connections"].values() if c["dst"] == new_node]
        out = [c for c in result["connections"].values() if c["src"] == new_node]
        assert len(into) == 1 and len(out) == 1
        assert into[0]["weight"] == 1.0
        disabled = [c for c in result["connections"].values() if not c["enabled"]]
        assert len(disabled) == 1
        # the outgoing weight is the weight the old connection has NOW
        assert out[0]["weight"] == disabled[0]["weight"]


def test_new_connection_keeps_its_starting_weight():
    # with weight_init_stdev 0 the new weight is exactly the mean, and the
    # weight step (which would nudge it) runs before the connection exists
    genome, innovations = world()
    rng = np.random.default_rng(11)
    genome = mutate(genome, rng, innovations, only(node_add_prob=1.0))
    cfg = only(weight_mutate_prob=1.0, conn_add_prob=1.0,
               weight_init_mean=7.0, weight_init_stdev=0.0)
    result = mutate(genome, rng, innovations, cfg)
    added = [c for key, c in result["connections"].items()
             if key not in genome["connections"]]
    assert len(added) == 1
    assert added[0]["weight"] == 7.0

###############################################################################
# cfg and probabilities
###############################################################################

def test_the_same_cfg_reaches_every_step():
    genome, innovations = world()
    rng = np.random.default_rng(12)
    cfg = only(weight_mutate_prob=1.0, weight_perturb_prob=0.0,
               weight_init_mean=3.0, weight_init_stdev=0.0)
    result = mutate(genome, rng, innovations, cfg)
    for weight in weights_of(result).values():
        assert weight == 3.0


def test_default_cfg_is_used_when_none():
    genome, innovations = world()
    rng = np.random.default_rng(13)
    added = 0
    for _ in range(2000):
        probe = copy.deepcopy(innovations)
        result = mutate(genome, rng, probe)
        if len(result["nodes"]) > len(genome["nodes"]):
            added += 1
    # default node_add_prob is 0.05 -> about 100 out of 2000
    assert 60 < added < 150


def test_cfg_is_not_modified():
    genome, innovations = world()
    cfg = only(weight_mutate_prob=1.0, node_add_prob=1.0, conn_add_prob=1.0)
    snapshot = copy.deepcopy(cfg)
    rng = np.random.default_rng(14)
    mutate(genome, rng, innovations, cfg)
    assert cfg == snapshot

###############################################################################
# input, registry, reproducibility
###############################################################################

def test_input_genome_is_never_modified():
    genome, innovations = world()
    snapshot = copy.deepcopy(genome)
    cfg = only(weight_mutate_prob=1.0, bias_mutate_prob=1.0,
               activation_mutate_prob=1.0, node_add_prob=1.0, conn_add_prob=1.0)
    rng = np.random.default_rng(15)
    for _ in range(50):
        probe = copy.deepcopy(innovations)
        mutate(genome, rng, probe, cfg)
    assert genome == snapshot


def test_registry_only_changes_when_structure_is_added():
    genome, innovations = world()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(16)
    cfg = only(weight_mutate_prob=1.0, bias_mutate_prob=1.0, activation_mutate_prob=1.0)
    for _ in range(50):
        mutate(genome, rng, innovations, cfg)
    assert innovations == snapshot


def test_same_seed_gives_the_same_result():
    cfg = only(weight_mutate_prob=0.5, bias_mutate_prob=0.5, activation_mutate_prob=0.5,
               node_add_prob=0.8, conn_add_prob=0.8)
    results = []
    for _ in range(2):
        genome, innovations = world(seed=99)
        rng = np.random.default_rng(123)
        for _ in range(30):
            genome = mutate(genome, rng, innovations, cfg)
        results.append((genome, innovations))
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_different_seeds_give_different_results():
    cfg = only(weight_mutate_prob=0.5, node_add_prob=0.8, conn_add_prob=0.8)
    outcomes = []
    for seed in (1, 2):
        genome, innovations = world(seed=99)
        rng = np.random.default_rng(seed)
        for _ in range(30):
            genome = mutate(genome, rng, innovations, cfg)
        outcomes.append(genome)
    assert outcomes[0] != outcomes[1]

###############################################################################
# growing a population
###############################################################################

def test_many_generations_keep_every_genome_consistent():
    # a small population sharing ONE registry, mutated for many generations
    innovations = init_innovations(3, 2)
    rng = np.random.default_rng(17)
    population = [init_genome(3, 2, rng) for _ in range(10)]
    cfg = only(weight_mutate_prob=0.8, bias_mutate_prob=0.7,
               activation_mutate_prob=0.1, node_add_prob=0.3, conn_add_prob=0.5)
    for _ in range(40):
        population = [mutate(genome, rng, innovations, cfg) for genome in population]
    for genome in population:
        check_consistent(genome, innovations)
    assert max(len(g["nodes"]) for g in population) > 5 + 3


def test_two_genomes_that_split_the_same_connection_share_the_node_id():
    # one parent, many children mutated with the shared registry: every node
    # id in the registry's splits is used for exactly one connection
    innovations = init_innovations(2, 1)
    rng = np.random.default_rng(18)
    parent = init_genome(2, 1, rng)
    cfg = only(node_add_prob=1.0)
    children = [mutate(parent, rng, innovations, cfg) for _ in range(100)]
    assert len(innovations["splits"]) == 2           # two connections to split
    assert sorted(innovations["splits"].values()) == [3, 4]
    for child in children:
        new_nodes = set(child["nodes"]) - set(parent["nodes"])
        assert len(new_nodes) == 1
        assert new_nodes <= {3, 4}

###############################################################################
# required arguments
###############################################################################

def test_rng_is_required():
    genome, innovations = world()
    try:
        mutate(genome, innovations=innovations)
    except TypeError:
        return
    raise AssertionError("mutate must require rng")


def test_innovations_is_required():
    genome, innovations = world()
    rng = np.random.default_rng(19)
    try:
        mutate(genome, rng)
    except TypeError:
        return
    raise AssertionError("mutate must require innovations")

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"mutate: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()