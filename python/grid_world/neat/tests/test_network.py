"""
Created on Mon Oct 19 14:59:18 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/network.py: compile_genome, activate.

A genome is a recipe. To run it fast it is first COMPILED once into a network
(plain dict), and then `activate` is called many times on that network.

The activation functions live in neat/activations.py (tested separately).

compile_genome(genome) -> network
    Only ENABLED connections are used. Nodes that cannot reach an output are
    left out (they cannot change any output). Nodes are put in an order where
    every node comes after all nodes that feed it. Inputs are taken in
    increasing node id, outputs also in increasing node id.
    The network shares nothing with the genome (changing the genome afterwards
    does not change the network) and is plain data (picklable, so it can be
    sent to worker processes).
    ValueError: unknown activation name, connection to/from a node that does
    not exist, connection INTO an input node, or a cycle among enabled
    connections.

activate(network, inputs) -> list of floats
    One value per output node. node value = activation(bias + sum(weight *
    value of source)) over enabled incoming connections. A node without
    incoming connections gives activation(bias). Feed-forward and stateless:
    same inputs, same outputs, the network is never changed.
    ValueError: wrong number of inputs, or an input that is NaN / infinite.
    Accepts lists, tuples and numpy arrays.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy
import math
import pickle

import numpy as np

from neat.config import default_config
from neat.genome import init_genome
from neat.innovations import init_innovations
from neat.mutation import mutate
from neat.activations import ACTIVATIONS
from neat.network import compile_genome, activate

###############################################################################
# helpers
###############################################################################

def sig(x):
    return 1.0 / (1.0 + math.exp(-4.9 * x))


def make(n_in, n_out, hidden=(), connections=(), act="identity", bias=0.0):
    """Genome: inputs 0..n_in-1, outputs next, then hidden ids given.
    connections: (src, dst, weight[, enabled]) tuples."""
    nodes = {i: {"type": "input"} for i in range(n_in)}
    for i in range(n_in, n_in + n_out):
        nodes[i] = {"type": "output", "bias": bias, "activation": act}
    for h in hidden:
        nodes[h] = {"type": "hidden", "bias": bias, "activation": act}
    conns = {}
    for k, c in enumerate(connections):
        enabled = c[3] if len(c) > 3 else True
        conns[k] = {"src": c[0], "dst": c[1], "weight": c[2], "enabled": enabled}
    return {"n_inputs": n_in, "n_outputs": n_out, "nodes": nodes, "connections": conns}


def reference(genome, inputs):
    """Slow, obviously-correct evaluation straight from the genome."""
    input_ids = sorted(n for n, v in genome["nodes"].items() if v["type"] == "input")
    output_ids = sorted(n for n, v in genome["nodes"].items() if v["type"] == "output")
    memo = {n: float(x) for n, x in zip(input_ids, inputs)}

    def value(node_id):
        if node_id in memo:
            return memo[node_id]
        node = genome["nodes"][node_id]
        total = node["bias"]
        for c in genome["connections"].values():
            if c["enabled"] and c["dst"] == node_id:
                total += c["weight"] * value(c["src"])
        memo[node_id] = ACTIVATIONS[node["activation"]](total)
        return memo[node_id]

    return [value(n) for n in output_ids]


def evolved(seed, n_in=5, n_out=3, steps=40):
    cfg = default_config()
    cfg["node_add_prob"] = 0.5
    cfg["conn_add_prob"] = 0.5
    cfg["activation_mutate_prob"] = 0.5
    rng = np.random.default_rng(seed)
    genome = init_genome(n_in, n_out, rng, cfg)
    innovations = init_innovations(n_in, n_out)
    for _ in range(steps):
        genome = mutate(genome, rng, innovations, cfg)
    return genome


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False

###############################################################################
# activate: hand-computed examples
###############################################################################

def test_single_connection():
    genome = make(1, 1, [], [(0, 1, 2.0)], act="sigmoid", bias=0.5)
    out = activate(compile_genome(genome), [0.25])
    assert len(out) == 1
    assert abs(out[0] - sig(2.0 * 0.25 + 0.5)) < 1e-12


def test_two_inputs_are_summed_with_their_own_weights():
    genome = make(2, 1, [], [(0, 2, 3.0), (1, 2, -1.0)], act="identity", bias=1.0)
    out = activate(compile_genome(genome), [2.0, 5.0])
    assert abs(out[0] - (1.0 + 6.0 - 5.0)) < 1e-12


def test_input_order_matters():
    genome = make(2, 1, [], [(0, 2, 1.0), (1, 2, 10.0)])
    net = compile_genome(genome)
    assert activate(net, [1.0, 0.0]) == [1.0]
    assert activate(net, [0.0, 1.0]) == [10.0]


def test_hidden_node_chain():
    genome = make(1, 1, [2], [(0, 2, 2.0), (2, 1, 3.0)], act="identity")
    genome["nodes"][2]["bias"] = 1.0
    out = activate(compile_genome(genome), [4.0])
    assert abs(out[0] - 3.0 * (2.0 * 4.0 + 1.0)) < 1e-12


def test_hidden_activation_is_applied_before_the_next_layer():
    genome = make(1, 1, [2], [(0, 2, 1.0), (2, 1, 1.0)], act="identity")
    genome["nodes"][2]["activation"] = "relu"
    net = compile_genome(genome)
    assert activate(net, [-5.0]) == [0.0]
    assert activate(net, [5.0]) == [5.0]


def test_diamond():
    # 0 -> 2, 0 -> 3, 2 -> 1, 3 -> 1
    genome = make(1, 1, [2, 3], [(0, 2, 1.0), (0, 3, 2.0), (2, 1, 10.0), (3, 1, 100.0)])
    out = activate(compile_genome(genome), [1.0])
    assert abs(out[0] - (10.0 + 200.0)) < 1e-12


def test_disabled_connection_is_ignored():
    genome = make(2, 1, [], [(0, 2, 1.0), (1, 2, 100.0, False)])
    assert activate(compile_genome(genome), [1.0, 1.0]) == [1.0]


def test_output_without_connections_gives_activation_of_bias():
    genome = make(2, 2, [], [(0, 2, 1.0)], act="sigmoid", bias=0.7)
    out = activate(compile_genome(genome), [0.0, 0.0])
    assert abs(out[1] - sig(0.7)) < 1e-12


def test_outputs_come_in_node_id_order():
    genome = make(1, 3, [], [(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)])
    assert activate(compile_genome(genome), [1.0]) == [1.0, 2.0, 3.0]


def test_output_feeding_another_output_works_in_any_id_order():
    # output 2 is computed from output 1, output 1 from the input
    genome = make(1, 2, [], [(0, 1, 2.0), (1, 2, 5.0)])
    assert activate(compile_genome(genome), [1.0]) == [2.0, 10.0]
    # and the other way round: output 1 is computed from output 2
    genome = make(1, 2, [], [(0, 2, 2.0), (2, 1, 5.0)])
    assert activate(compile_genome(genome), [1.0]) == [10.0, 2.0]


def test_hidden_node_ids_larger_than_outputs_and_any_order():
    genome = make(1, 1, [9, 5], [(0, 9, 2.0), (9, 5, 3.0), (5, 1, 4.0)])
    assert activate(compile_genome(genome), [1.0]) == [24.0]


def test_connection_keys_do_not_need_to_be_in_topological_order():
    genome = make(1, 1, [2, 3], [])
    genome["connections"] = {
        7: {"src": 3, "dst": 1, "weight": 2.0, "enabled": True},
        2: {"src": 2, "dst": 3, "weight": 3.0, "enabled": True},
        5: {"src": 0, "dst": 2, "weight": 5.0, "enabled": True},
    }
    assert activate(compile_genome(genome), [1.0]) == [30.0]

###############################################################################
# activate: inputs and results
###############################################################################

def test_returns_a_list_of_floats():
    genome = make(2, 2, [], [(0, 2, 1.0)])
    out = activate(compile_genome(genome), [1, 2])
    assert type(out) is list
    assert all(type(v) is float for v in out)


def test_accepts_tuples_and_numpy_arrays():
    genome = make(3, 1, [], [(0, 3, 1.0), (1, 3, 2.0), (2, 3, 3.0)])
    net = compile_genome(genome)
    expected = activate(net, [1.0, 1.0, 1.0])
    assert activate(net, (1.0, 1.0, 1.0)) == expected
    assert activate(net, np.array([1.0, 1.0, 1.0])) == expected
    assert activate(net, np.array([1, 1, 1], dtype=np.int64)) == expected
    assert activate(net, np.array([1.0, 1.0, 1.0], dtype=np.float32)) == expected


def test_wrong_number_of_inputs_raises():
    net = compile_genome(make(3, 1, [], [(0, 3, 1.0)]))
    for bad in ([], [1.0], [1.0, 2.0], [1.0] * 4):
        assert raises(ValueError, activate, net, bad)


def test_non_finite_input_raises():
    net = compile_genome(make(2, 1, [], [(0, 2, 1.0)]))
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert raises(ValueError, activate, net, [1.0, bad])
        assert raises(ValueError, activate, net, [bad, 1.0])


def test_activate_is_stateless_and_does_not_change_the_network():
    genome = evolved(1)
    net = compile_genome(genome)
    snapshot = copy.deepcopy(net)
    rng = np.random.default_rng(0)
    x = list(rng.random(5))
    first = activate(net, x)
    for _ in range(3):
        activate(net, list(rng.random(5)))
    assert activate(net, x) == first
    assert net == snapshot


def test_the_input_list_is_not_changed():
    net = compile_genome(make(2, 1, [], [(0, 2, 1.0)]))
    x = [1.0, 2.0]
    activate(net, x)
    assert x == [1.0, 2.0]


def test_a_network_with_no_enabled_connections_still_works():
    genome = make(2, 2, [], [(0, 2, 1.0, False)], act="sigmoid", bias=0.0)
    assert activate(compile_genome(genome), [1.0, 1.0]) == [0.5, 0.5]

###############################################################################
# compile_genome
###############################################################################

def test_network_shape():
    net = compile_genome(make(4, 2, [6], [(0, 6, 1.0), (6, 4, 1.0)]))
    assert net["n_inputs"] == 4
    assert net["n_outputs"] == 2
    assert len(net["output_slots"]) == 2


def test_compile_does_not_change_the_genome():
    genome = evolved(2)
    snapshot = copy.deepcopy(genome)
    compile_genome(genome)
    assert genome == snapshot


def test_network_shares_nothing_with_the_genome():
    genome = make(1, 1, [], [(0, 1, 2.0)])
    net = compile_genome(genome)
    genome["connections"][0]["weight"] = 99.0
    genome["nodes"][1]["bias"] = 99.0
    genome["nodes"][1]["activation"] = "step"
    assert activate(net, [1.0]) == [2.0]


def test_compile_is_deterministic():
    genome = evolved(3)
    assert compile_genome(genome) == compile_genome(copy.deepcopy(genome))


def test_dead_nodes_are_left_out():
    # hidden 3 only feeds hidden 4, which feeds nothing: both are dead
    genome = make(1, 1, [3, 4], [(0, 1, 2.0), (0, 3, 1.0), (3, 4, 1.0)])
    net = compile_genome(genome)
    assert len(net["steps"]) == 1
    assert activate(net, [1.0]) == [2.0]


def test_a_node_fed_only_by_disabled_connections_is_still_computed():
    # hidden 2 has no live input, so it is activation(bias), and it feeds the output
    genome = make(1, 1, [2], [(0, 2, 5.0, False), (2, 1, 1.0)], act="sigmoid", bias=0.3)
    net = compile_genome(genome)
    out = activate(net, [1.0])
    assert abs(out[0] - sig(0.3 + sig(0.3))) < 1e-12


def test_a_path_through_a_disabled_connection_does_not_keep_nodes_alive():
    genome = make(1, 1, [2], [(0, 2, 1.0), (2, 1, 1.0, False)])
    assert len(compile_genome(genome)["steps"]) == 1


def test_unused_input_is_fine():
    genome = make(3, 1, [], [(1, 3, 2.0)])
    assert activate(compile_genome(genome), [100.0, 1.0, 100.0]) == [2.0]


def test_unknown_activation_raises():
    genome = make(1, 1, [], [(0, 1, 1.0)])
    genome["nodes"][1]["activation"] = "banana"
    assert raises(ValueError, compile_genome, genome)


def test_unknown_activation_raises_even_on_a_dead_node():
    genome = make(1, 1, [2], [(0, 1, 1.0)])
    genome["nodes"][2]["activation"] = "banana"
    assert raises(ValueError, compile_genome, genome)


def test_missing_node_raises():
    genome = make(1, 1, [], [(0, 1, 1.0)])
    genome["connections"][1] = {"src": 0, "dst": 77, "weight": 1.0, "enabled": True}
    assert raises(ValueError, compile_genome, genome)
    genome = make(1, 1, [], [(0, 1, 1.0)])
    genome["connections"][1] = {"src": 77, "dst": 1, "weight": 1.0, "enabled": True}
    assert raises(ValueError, compile_genome, genome)


def test_connection_into_an_input_raises():
    genome = make(2, 1, [], [(0, 2, 1.0), (1, 0, 1.0)])
    assert raises(ValueError, compile_genome, genome)


def test_cycle_raises():
    genome = make(1, 1, [2, 3], [(0, 2, 1.0), (2, 3, 1.0), (3, 2, 1.0), (3, 1, 1.0)])
    assert raises(ValueError, compile_genome, genome)
    genome = make(1, 2, [], [(0, 1, 1.0), (1, 2, 1.0), (2, 1, 1.0)])
    assert raises(ValueError, compile_genome, genome)


def test_self_loop_raises():
    genome = make(1, 1, [2], [(0, 2, 1.0), (2, 2, 1.0), (2, 1, 1.0)])
    assert raises(ValueError, compile_genome, genome)


def test_a_disabled_cycle_is_fine():
    genome = make(1, 1, [2], [(0, 2, 1.0), (2, 1, 1.0), (1, 2, 1.0, False)])
    assert activate(compile_genome(genome), [3.0]) == [3.0]


def test_network_can_be_pickled():
    genome = evolved(4)
    net = compile_genome(genome)
    clone = pickle.loads(pickle.dumps(net))
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert activate(clone, x) == activate(net, x)

###############################################################################
# against the slow reference, on many evolved genomes
###############################################################################

def test_matches_reference_on_evolved_genomes():
    rng = np.random.default_rng(123)
    for seed in range(60):
        genome = evolved(seed, n_in=5, n_out=3, steps=10 + seed)
        net = compile_genome(genome)
        for _ in range(5):
            x = list(rng.normal(0.0, 2.0, 5))
            got = activate(net, x)
            want = reference(genome, x)
            for g, w in zip(got, want):
                assert abs(g - w) < 1e-9, (seed, got, want)


def test_evolved_genomes_really_have_hidden_nodes():
    hidden = 0
    for seed in range(20):
        genome = evolved(seed, steps=50)
        hidden += sum(1 for n in genome["nodes"].values() if n["type"] == "hidden")
        net = compile_genome(genome)
        assert len(net["steps"]) <= len(genome["nodes"]) - 5
    assert hidden > 20


def test_player_sized_network_runs():
    genome = evolved(9, n_in=29, n_out=4, steps=30)
    net = compile_genome(genome)
    out = activate(net, [0.5] * 29)
    assert len(out) == 4

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"network: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()