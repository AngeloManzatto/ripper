#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 06:35:26 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/mutation.py: add_connection(genome, rng, innovations, cfg=None).

What add_connection must do
---------------------------
1. Roll the dice once: with probability cfg["conn_add_prob"] it tries to add
   one connection, otherwise it changes nothing.
2. A candidate is a pair (src, dst) where
     - src is ANY node (input, hidden or output),
     - dst is any node that is NOT an input,
     - src != dst (no self loops),
     - the pair is not already in the genome (enabled OR disabled),
     - adding it does not create a cycle (disabled connections count).
3. If at least one candidate exists, exactly one of them is added (chosen at
   random, every candidate possible). If none exists, nothing is added.
4. The new connection is enabled, has a weight drawn from the INITIAL weight
   distribution (weight_init_mean / weight_init_stdev, no clipping needed),
   and is stored under its innovation number, obtained from
   get_connection_innovation(innovations, src, dst).
5. It returns a NEW genome (the input is never touched), also when nothing
   was added. The registry `innovations` is changed (in place) only when a
   connection was really added, and only through get_connection_innovation.

Test genomes are tiny and hand built, so every case has a known answer.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.innovations import init_innovations, get_connection_innovation
from neat.mutation import add_connection

###############################################################################
# helpers
###############################################################################

def make_world(n_inputs, n_outputs, hidden_ids, conns):
    """Build (genome, innovations) by hand.

    conns is a list of (src, dst) or (src, dst, enabled). The genome keys are
    the numbers given by the registry, so genome and registry agree.
    """
    nodes = {}
    for node_id in range(n_inputs):
        nodes[node_id] = {"type": "input"}
    for node_id in range(n_inputs, n_inputs + n_outputs):
        nodes[node_id] = {"type": "output", "bias": 0.0, "activation": "sigmoid"}
    for node_id in hidden_ids:
        nodes[node_id] = {"type": "hidden", "bias": 0.0, "activation": "sigmoid"}

    innovations = init_innovations(n_inputs, n_outputs)
    innovations["next_node_id"] = max(nodes.keys()) + 1

    connections = {}
    for conn in conns:
        src, dst = conn[0], conn[1]
        enabled = conn[2] if len(conn) == 3 else True
        number = get_connection_innovation(innovations, src, dst)
        connections[number] = {"src": src, "dst": dst, "weight": 1.0, "enabled": enabled}

    genome = {"n_inputs": n_inputs, "n_outputs": n_outputs,
              "nodes": nodes, "connections": connections}
    return genome, innovations


def always(**overrides):
    """cfg where the dice always say yes."""
    cfg = default_config()
    cfg["conn_add_prob"] = 1.0
    cfg.update(overrides)
    return cfg


def pairs_of(genome):
    """Set of (src, dst) in a genome."""
    return {(c["src"], c["dst"]) for c in genome["connections"].values()}


def new_pairs(before, after):
    """The pairs that are in `after` but not in `before`."""
    return pairs_of(after) - pairs_of(before)


def world_room():
    """inputs 0,1  output 2  hidden 3.   0->2, 1->2.

    Valid candidates (4): (0,3) (1,3) (2,3) (3,2).
    Not valid: (0,2) (1,2) duplicates, (2,2) (3,3) self loops, any dst input.
    """
    return make_world(2, 1, [3], [(0, 2), (1, 2)])


def world_one_valid():
    """inputs 0,1  output 2  hidden 3.   0->2, 1->2, 0->3, 3->2.

    The ONLY valid candidate is (1,3):
      (0,2) (1,2) (0,3) (3,2) duplicates, (2,2) (3,3) self loops,
      (2,3) would close the loop 3 -> 2 -> 3.
    """
    return make_world(2, 1, [3], [(0, 2), (1, 2), (0, 3), (3, 2)])


def world_none_valid():
    """Same as world_one_valid but 1->3 already exists: nothing is valid."""
    return make_world(2, 1, [3], [(0, 2), (1, 2), (0, 3), (1, 3), (3, 2)])


def world_chain():
    """input 0  output 1  hidden 2,3.   0->1, 0->2, 2->3, 3->1.

    Valid candidates (2): (0,3) and (2,1)  (both go "forward").
    Cycles that must be refused: (1,2) (1,3) (3,2).
    """
    return make_world(1, 1, [2, 3], [(0, 1), (0, 2), (2, 3), (3, 1)])

###############################################################################
# the dice
###############################################################################

def test_probability_zero_never_adds():
    genome, innovations = world_room()
    cfg = always(conn_add_prob=0.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        result = add_connection(genome, rng, innovations, cfg)
        assert pairs_of(result) == pairs_of(genome)


def test_probability_one_always_adds_when_possible():
    genome, innovations = world_room()
    rng = np.random.default_rng(2)
    for _ in range(100):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        assert len(result["connections"]) == len(genome["connections"]) + 1


def test_probability_is_respected():
    genome, innovations = world_room()
    cfg = always(conn_add_prob=0.5)
    rng = np.random.default_rng(3)
    added = 0
    for _ in range(1000):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, cfg)
        if len(result["connections"]) > len(genome["connections"]):
            added += 1
    assert 420 < added < 580


def test_default_cfg_is_used_when_none():
    genome, innovations = world_room()
    rng = np.random.default_rng(4)
    added = 0
    for _ in range(2000):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe)
        if len(result["connections"]) > len(genome["connections"]):
            added += 1
    # default conn_add_prob is 0.03 -> about 60 out of 2000
    assert 25 < added < 110

###############################################################################
# what is a valid connection
###############################################################################

def test_adds_exactly_one_connection():
    genome, innovations = world_room()
    rng = np.random.default_rng(5)
    result = add_connection(genome, rng, innovations, always())
    assert len(result["connections"]) == len(genome["connections"]) + 1
    assert len(new_pairs(genome, result)) == 1


def test_new_connection_is_enabled_with_known_fields():
    genome, innovations = world_room()
    rng = np.random.default_rng(6)
    result = add_connection(genome, rng, innovations, always())
    added = [c for key, c in result["connections"].items()
             if key not in genome["connections"]]
    assert len(added) == 1
    conn = added[0]
    assert set(conn.keys()) == {"src", "dst", "weight", "enabled"}
    assert conn["enabled"] is True
    assert conn["src"] in result["nodes"]
    assert conn["dst"] in result["nodes"]


def test_destination_is_never_an_input():
    genome, innovations = world_room()
    rng = np.random.default_rng(7)
    for _ in range(300):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        for src, dst in new_pairs(genome, result):
            assert result["nodes"][dst]["type"] != "input"


def test_never_a_self_loop():
    genome, innovations = world_room()
    rng = np.random.default_rng(8)
    for _ in range(300):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        for src, dst in new_pairs(genome, result):
            assert src != dst


def test_never_a_duplicate():
    genome, innovations = world_room()
    rng = np.random.default_rng(9)
    for _ in range(300):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        listed = [(c["src"], c["dst"]) for c in result["connections"].values()]
        assert len(listed) == len(set(listed))


def test_every_valid_candidate_can_be_chosen():
    genome, innovations = world_room()
    rng = np.random.default_rng(10)
    counts = {}
    trials = 2000
    for _ in range(trials):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        pair = next(iter(new_pairs(genome, result)))
        counts[pair] = counts.get(pair, 0) + 1
    assert set(counts.keys()) == {(0, 3), (1, 3), (2, 3), (3, 2)}
    for pair, count in counts.items():
        assert count > 0.15 * trials, (pair, count)   # expected 25% each


def test_the_only_valid_candidate_is_always_found():
    genome, innovations = world_one_valid()
    rng = np.random.default_rng(11)
    for _ in range(300):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        assert new_pairs(genome, result) == {(1, 3)}


def test_nothing_valid_returns_the_same_connections():
    genome, innovations = world_none_valid()
    rng = np.random.default_rng(12)
    for _ in range(100):
        result = add_connection(genome, rng, innovations, always())
        assert result == genome


def test_full_initial_genome_has_no_room():
    # fully connected input->output, no hidden nodes: nothing can be added
    # (output->output would be a loop or a duplicate in this tiny net)
    genome, innovations = make_world(3, 1, [], [(0, 3), (1, 3), (2, 3)])
    rng = np.random.default_rng(13)
    result = add_connection(genome, rng, innovations, always())
    assert result == genome

###############################################################################
# cycles
###############################################################################

def test_cycle_closing_pairs_are_refused():
    genome, innovations = world_chain()
    rng = np.random.default_rng(14)
    seen = set()
    for _ in range(400):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, always())
        seen |= new_pairs(genome, result)
    assert seen == {(0, 3), (2, 1)}


def test_disabled_connections_still_count_for_cycles():
    # 3->2 is disabled, but 2->3 would still close a loop through it
    genome, innovations = make_world(2, 1, [3],
                                     [(0, 2), (1, 2), (0, 3), (1, 3), (3, 2, False)])
    rng = np.random.default_rng(15)
    for _ in range(100):
        result = add_connection(genome, rng, innovations, always())
        assert result == genome


def test_result_has_no_cycles_after_many_additions():
    # grow a genome with many hidden nodes; the pairs must stay acyclic
    hidden = list(range(4, 14))
    genome, innovations = make_world(3, 1, hidden, [(0, 3), (1, 3), (2, 3)])
    rng = np.random.default_rng(16)
    cfg = always()
    for _ in range(150):
        genome = add_connection(genome, rng, innovations, cfg)
    assert len(genome["connections"]) > 20
    # Kahn's algorithm: all nodes can be removed in topological order
    incoming = {node_id: 0 for node_id in genome["nodes"]}
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
    assert removed == len(genome["nodes"])

###############################################################################
# the weight
###############################################################################

def test_weight_is_a_python_float():
    genome, innovations = world_room()
    rng = np.random.default_rng(17)
    result = add_connection(genome, rng, innovations, always())
    for key, conn in result["connections"].items():
        if key not in genome["connections"]:
            assert type(conn["weight"]) is float


def test_weight_follows_the_init_distribution():
    genome, innovations = world_one_valid()
    cfg = always(weight_init_mean=5.0, weight_init_stdev=0.0)
    rng = np.random.default_rng(18)
    result = add_connection(genome, rng, innovations, cfg)
    for key, conn in result["connections"].items():
        if key not in genome["connections"]:
            assert conn["weight"] == 5.0


def test_weight_has_spread_and_the_right_mean():
    genome, innovations = world_one_valid()
    cfg = always(weight_init_mean=2.0, weight_init_stdev=1.0)
    rng = np.random.default_rng(19)
    weights = []
    for _ in range(2000):
        probe = copy.deepcopy(innovations)
        result = add_connection(genome, rng, probe, cfg)
        added = [c for key, c in result["connections"].items()
                 if key not in genome["connections"]]
        weights.append(added[0]["weight"])
    assert abs(np.mean(weights) - 2.0) < 0.1
    assert abs(np.std(weights) - 1.0) < 0.1

###############################################################################
# innovation numbers
###############################################################################

def test_new_pair_gets_the_next_innovation_number():
    genome, innovations = world_one_valid()
    expected = innovations["next_innovation"]
    rng = np.random.default_rng(20)
    result = add_connection(genome, rng, innovations, always())
    assert expected not in genome["connections"]
    assert result["connections"][expected]["src"] == 1
    assert result["connections"][expected]["dst"] == 3


def test_new_pair_is_registered_and_counter_moves_by_one():
    genome, innovations = world_one_valid()
    before = innovations["next_innovation"]
    rng = np.random.default_rng(21)
    add_connection(genome, rng, innovations, always())
    assert innovations["connections"][(1, 3)] == before
    assert innovations["next_innovation"] == before + 1


def test_known_pair_reuses_its_number_and_counter_is_untouched():
    genome, innovations = world_one_valid()
    # another genome already found (1,3): the registry knows it
    number = get_connection_innovation(innovations, 1, 3)
    counter = innovations["next_innovation"]
    rng = np.random.default_rng(22)
    result = add_connection(genome, rng, innovations, always())
    assert result["connections"][number]["src"] == 1
    assert result["connections"][number]["dst"] == 3
    assert innovations["next_innovation"] == counter


def test_two_genomes_adding_the_same_pair_share_the_number():
    genome_a, innovations = world_one_valid()
    genome_b = copy.deepcopy(genome_a)
    rng = np.random.default_rng(23)
    result_a = add_connection(genome_a, rng, innovations, always())
    counter = innovations["next_innovation"]
    result_b = add_connection(genome_b, rng, innovations, always())
    keys_a = set(result_a["connections"]) - set(genome_a["connections"])
    keys_b = set(result_b["connections"]) - set(genome_b["connections"])
    assert keys_a == keys_b
    assert innovations["next_innovation"] == counter


def test_connection_key_is_a_python_int():
    genome, innovations = world_one_valid()
    rng = np.random.default_rng(24)
    result = add_connection(genome, rng, innovations, always())
    for key in result["connections"]:
        assert type(key) is int


def test_registry_untouched_when_the_dice_say_no():
    genome, innovations = world_room()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(25)
    for _ in range(100):
        add_connection(genome, rng, innovations, always(conn_add_prob=0.0))
    assert innovations == snapshot


def test_registry_untouched_when_nothing_is_valid():
    genome, innovations = world_none_valid()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(26)
    for _ in range(100):
        add_connection(genome, rng, innovations, always())
    assert innovations == snapshot


def test_registry_only_changes_connections_and_counter():
    genome, innovations = world_one_valid()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(27)
    add_connection(genome, rng, innovations, always())
    assert innovations["next_node_id"] == snapshot["next_node_id"]
    assert innovations["splits"] == snapshot["splits"]

###############################################################################
# new genome, nothing else touched
###############################################################################

def test_input_genome_is_never_modified():
    genome, innovations = world_room()
    snapshot = copy.deepcopy(genome)
    rng = np.random.default_rng(28)
    for _ in range(100):
        probe = copy.deepcopy(innovations)
        add_connection(genome, rng, probe, always())
    assert genome == snapshot


def test_returns_a_new_object_even_when_nothing_is_added():
    rng = np.random.default_rng(29)
    genome, innovations = world_none_valid()
    assert add_connection(genome, rng, innovations, always()) is not genome
    genome, innovations = world_room()
    result = add_connection(genome, rng, innovations, always(conn_add_prob=0.0))
    assert result is not genome
    assert result == genome


def test_result_does_not_share_inner_dicts_with_the_input():
    genome, innovations = world_room()
    rng = np.random.default_rng(30)
    result = add_connection(genome, rng, innovations, always())
    assert result["nodes"] is not genome["nodes"]
    assert result["connections"] is not genome["connections"]
    for key, conn in genome["connections"].items():
        assert result["connections"][key] is not conn


def test_old_connections_and_nodes_are_kept_unchanged():
    genome, innovations = world_room()
    rng = np.random.default_rng(31)
    result = add_connection(genome, rng, innovations, always())
    assert result["nodes"] == genome["nodes"]
    assert result["n_inputs"] == genome["n_inputs"]
    assert result["n_outputs"] == genome["n_outputs"]
    for key, conn in genome["connections"].items():
        assert result["connections"][key] == conn


def test_cfg_is_not_modified():
    genome, innovations = world_room()
    cfg = always()
    snapshot = copy.deepcopy(cfg)
    rng = np.random.default_rng(32)
    add_connection(genome, rng, innovations, cfg)
    assert cfg == snapshot

###############################################################################
# required arguments
###############################################################################

def test_rng_is_required():
    genome, innovations = world_room()
    try:
        add_connection(genome, innovations=innovations)
    except TypeError:
        return
    raise AssertionError("add_connection must require rng")


def test_innovations_is_required():
    genome, innovations = world_room()
    rng = np.random.default_rng(33)
    try:
        add_connection(genome, rng)
    except TypeError:
        return
    raise AssertionError("add_connection must require innovations")

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"add_connection: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()