"""
Created on Mon Oct 19 07:02:59 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Tests for neat/mutation.py: add_node(genome, rng, innovations, cfg=None).

What add_node must do
---------------------
1. Roll the dice once: with probability cfg["node_add_prob"] it tries to add
   one node, otherwise it changes nothing.
2. It splits ONE connection A -> B, chosen at random among the candidates:
     - the connection must be enabled;
     - a connection whose recorded split node (innovations["splits"]) is
       already a node of this genome is NOT a candidate (no duplicate ids).
   If there is no candidate, nothing changes.
3. Splitting means:
     - the old connection stays in the genome but becomes enabled=False
       (its key, src, dst and weight are kept);
     - a new hidden node N is added: {"type": "hidden", "bias": 0.0,
       "activation": cfg["hidden_activation"]};
     - connection A -> N is added: weight 1.0, enabled;
     - connection N -> B is added: weight = the OLD weight, enabled.
       (so the network behaves almost like before the split)
4. Registry (changed in place, only when a node is really added):
     - the id of N is innovations["splits"][old_key] when that connection was
       split before (by any genome), so every genome that splits the same
       connection gets the same node id;
     - otherwise N = innovations["next_node_id"], it is stored in
       innovations["splits"][old_key] and next_node_id moves up by one;
     - the two new connections get their numbers from
       get_connection_innovation (A, N) and (N, B).
5. It returns a NEW genome (the input is never touched), also when nothing
   was added.

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
from neat.mutation import add_node

###############################################################################
# helpers
###############################################################################

def make_world(n_inputs, n_outputs, hidden_ids, conns, splits=None):
    """Build (genome, innovations) by hand.

    conns is a list of (src, dst), (src, dst, enabled) or
    (src, dst, enabled, weight). The genome keys are the numbers given by the
    registry, so genome and registry agree. `splits` pre-fills
    innovations["splits"] (a dict {connection_key: node_id}).
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
    if splits is not None:
        innovations["splits"] = dict(splits)

    connections = {}
    for conn in conns:
        src, dst = conn[0], conn[1]
        enabled = conn[2] if len(conn) >= 3 else True
        weight = conn[3] if len(conn) == 4 else 1.0
        number = get_connection_innovation(innovations, src, dst)
        connections[number] = {"src": src, "dst": dst, "weight": weight, "enabled": enabled}

    genome = {"n_inputs": n_inputs, "n_outputs": n_outputs,
              "nodes": nodes, "connections": connections}
    return genome, innovations


def always(**overrides):
    """cfg where the dice always say yes."""
    cfg = default_config()
    cfg["node_add_prob"] = 1.0
    cfg.update(overrides)
    return cfg


def world_one():
    """inputs 0,1  output 2.   0->2 (weight 2.5) and 1->2 (weight -1.5)."""
    return make_world(2, 1, [], [(0, 2, True, 2.5), (1, 2, True, -1.5)])


def world_single():
    """input 0  output 1.   One connection 0->1 with weight 2.5 (key 0)."""
    return make_world(1, 1, [], [(0, 1, True, 2.5)])


def world_mixed():
    """inputs 0,1  output 2.   0->2 disabled, 1->2 enabled."""
    return make_world(2, 1, [], [(0, 2, False, 2.5), (1, 2, True, -1.5)])


def world_nothing():
    """inputs 0,1  output 2.   Both connections are disabled."""
    return make_world(2, 1, [], [(0, 2, False), (1, 2, False)])


def split_of(before, after):
    """Return (new_node_id, old_key, in_key, out_key) of a one-node split."""
    new_ids = set(after["nodes"]) - set(before["nodes"])
    assert len(new_ids) == 1
    node_id = next(iter(new_ids))
    old_keys = [key for key, conn in after["connections"].items()
                if key in before["connections"]
                and conn["enabled"] != before["connections"][key]["enabled"]]
    assert len(old_keys) == 1
    in_keys = [key for key, conn in after["connections"].items()
               if key not in before["connections"] and conn["dst"] == node_id]
    out_keys = [key for key, conn in after["connections"].items()
                if key not in before["connections"] and conn["src"] == node_id]
    assert len(in_keys) == 1 and len(out_keys) == 1
    return node_id, old_keys[0], in_keys[0], out_keys[0]

###############################################################################
# the dice
###############################################################################

def test_probability_zero_never_adds():
    genome, innovations = world_one()
    cfg = always(node_add_prob=0.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        result = add_node(genome, rng, innovations, cfg)
        assert result == genome


def test_probability_one_always_adds_when_possible():
    genome, innovations = world_one()
    rng = np.random.default_rng(2)
    for _ in range(100):
        probe = copy.deepcopy(innovations)
        result = add_node(genome, rng, probe, always())
        assert len(result["nodes"]) == len(genome["nodes"]) + 1


def test_probability_is_respected():
    genome, innovations = world_one()
    cfg = always(node_add_prob=0.5)
    rng = np.random.default_rng(3)
    added = 0
    for _ in range(1000):
        probe = copy.deepcopy(innovations)
        result = add_node(genome, rng, probe, cfg)
        if len(result["nodes"]) > len(genome["nodes"]):
            added += 1
    assert 420 < added < 580


def test_default_cfg_is_used_when_none():
    genome, innovations = world_one()
    rng = np.random.default_rng(4)
    added = 0
    for _ in range(2000):
        probe = copy.deepcopy(innovations)
        result = add_node(genome, rng, probe)
        if len(result["nodes"]) > len(genome["nodes"]):
            added += 1
    # default node_add_prob is 0.05 -> about 100 out of 2000
    assert 60 < added < 150

###############################################################################
# the split: nodes
###############################################################################

def test_adds_exactly_one_node_and_two_connections():
    genome, innovations = world_one()
    rng = np.random.default_rng(5)
    result = add_node(genome, rng, innovations, always())
    assert len(result["nodes"]) == len(genome["nodes"]) + 1
    assert len(result["connections"]) == len(genome["connections"]) + 2


def test_new_node_is_a_hidden_node_with_known_fields():
    genome, innovations = world_one()
    rng = np.random.default_rng(6)
    result = add_node(genome, rng, innovations, always())
    node_id = split_of(genome, result)[0]
    node = result["nodes"][node_id]
    assert set(node.keys()) == {"type", "bias", "activation"}
    assert node["type"] == "hidden"


def test_new_node_bias_is_zero():
    genome, innovations = world_one()
    rng = np.random.default_rng(7)
    result = add_node(genome, rng, innovations, always())
    node_id = split_of(genome, result)[0]
    assert result["nodes"][node_id]["bias"] == 0.0
    assert type(result["nodes"][node_id]["bias"]) is float


def test_new_node_activation_comes_from_cfg():
    genome, innovations = world_one()
    rng = np.random.default_rng(8)
    cfg = always(hidden_activation="tanh")
    result = add_node(genome, rng, innovations, cfg)
    node_id = split_of(genome, result)[0]
    assert result["nodes"][node_id]["activation"] == "tanh"


def test_new_node_id_is_the_registry_next_node_id():
    genome, innovations = world_one()
    expected = innovations["next_node_id"]
    rng = np.random.default_rng(9)
    result = add_node(genome, rng, innovations, always())
    assert expected in result["nodes"]
    assert expected not in genome["nodes"]


def test_new_node_id_is_a_python_int():
    genome, innovations = world_one()
    rng = np.random.default_rng(10)
    result = add_node(genome, rng, innovations, always())
    for node_id in result["nodes"]:
        assert type(node_id) is int

###############################################################################
# the split: connections
###############################################################################

def test_old_connection_is_kept_but_disabled():
    genome, innovations = world_single()
    rng = np.random.default_rng(11)
    result = add_node(genome, rng, innovations, always())
    old = result["connections"][0]
    assert old["enabled"] is False
    assert old["src"] == 0
    assert old["dst"] == 1
    assert old["weight"] == 2.5


def test_incoming_connection_has_weight_one():
    genome, innovations = world_single()
    rng = np.random.default_rng(12)
    result = add_node(genome, rng, innovations, always())
    node_id, old_key, in_key, out_key = split_of(genome, result)
    conn = result["connections"][in_key]
    assert conn["src"] == 0
    assert conn["dst"] == node_id
    assert conn["weight"] == 1.0
    assert conn["enabled"] is True


def test_outgoing_connection_keeps_the_old_weight():
    genome, innovations = world_single()
    rng = np.random.default_rng(13)
    result = add_node(genome, rng, innovations, always())
    node_id, old_key, in_key, out_key = split_of(genome, result)
    conn = result["connections"][out_key]
    assert conn["src"] == node_id
    assert conn["dst"] == 1
    assert conn["weight"] == 2.5
    assert conn["enabled"] is True


def test_new_connections_have_exactly_the_known_fields():
    genome, innovations = world_single()
    rng = np.random.default_rng(14)
    result = add_node(genome, rng, innovations, always())
    node_id, old_key, in_key, out_key = split_of(genome, result)
    for key in (in_key, out_key):
        assert set(result["connections"][key].keys()) == {"src", "dst", "weight", "enabled"}
        assert type(result["connections"][key]["weight"]) is float


def test_connection_keys_are_python_ints():
    genome, innovations = world_one()
    rng = np.random.default_rng(15)
    result = add_node(genome, rng, innovations, always())
    for key in result["connections"]:
        assert type(key) is int


def test_only_the_split_connection_changes():
    genome, innovations = world_one()
    rng = np.random.default_rng(16)
    result = add_node(genome, rng, innovations, always())
    node_id, old_key, in_key, out_key = split_of(genome, result)
    for key, conn in genome["connections"].items():
        if key != old_key:
            assert result["connections"][key] == conn
    for node, content in genome["nodes"].items():
        assert result["nodes"][node] == content

###############################################################################
# which connection is split
###############################################################################

def test_every_enabled_connection_can_be_split():
    genome, innovations = world_one()
    rng = np.random.default_rng(17)
    counts = {}
    trials = 2000
    for _ in range(trials):
        probe = copy.deepcopy(innovations)
        result = add_node(genome, rng, probe, always())
        old_key = split_of(genome, result)[1]
        counts[old_key] = counts.get(old_key, 0) + 1
    assert set(counts.keys()) == set(genome["connections"].keys())
    for key, count in counts.items():
        assert count > 0.35 * trials, (key, count)      # expected 50% each


def test_disabled_connections_are_never_split():
    genome, innovations = world_mixed()
    rng = np.random.default_rng(18)
    for _ in range(300):
        probe = copy.deepcopy(innovations)
        result = add_node(genome, rng, probe, always())
        node_id, old_key, in_key, out_key = split_of(genome, result)
        assert genome["connections"][old_key]["src"] == 1


def test_nothing_enabled_returns_the_same_genome():
    genome, innovations = world_nothing()
    rng = np.random.default_rng(19)
    for _ in range(100):
        result = add_node(genome, rng, innovations, always())
        assert result == genome


def test_connection_whose_split_node_already_exists_is_skipped():
    # connection 0 (0->2) was split before into node 3, and node 3 is in this
    # genome: splitting it again would repeat the id 3
    genome, innovations = make_world(2, 1, [3], [(0, 2), (1, 2)], splits={0: 3})
    rng = np.random.default_rng(20)
    for _ in range(200):
        probe = copy.deepcopy(innovations)
        result = add_node(genome, rng, probe, always())
        node_id, old_key, in_key, out_key = split_of(genome, result)
        assert old_key == 1
        assert node_id == 4


def test_when_the_only_enabled_connection_is_blocked_nothing_changes():
    genome, innovations = make_world(1, 1, [2], [(0, 1)], splits={0: 2})
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(21)
    result = add_node(genome, rng, innovations, always())
    assert result == genome
    assert innovations == snapshot

###############################################################################
# the registry
###############################################################################

def test_first_split_is_recorded_in_the_registry():
    genome, innovations = world_single()
    before = innovations["next_node_id"]
    rng = np.random.default_rng(22)
    result = add_node(genome, rng, innovations, always())
    node_id = split_of(genome, result)[0]
    assert node_id == before
    assert innovations["splits"] == {0: before}
    assert innovations["next_node_id"] == before + 1


def test_new_connections_are_registered_with_their_numbers():
    genome, innovations = world_single()
    counter = innovations["next_innovation"]
    rng = np.random.default_rng(23)
    result = add_node(genome, rng, innovations, always())
    node_id, old_key, in_key, out_key = split_of(genome, result)
    assert innovations["connections"][(0, node_id)] == in_key
    assert innovations["connections"][(node_id, 1)] == out_key
    assert sorted([in_key, out_key]) == [counter, counter + 1]
    assert innovations["next_innovation"] == counter + 2


def test_second_genome_splitting_the_same_connection_reuses_everything():
    genome_a, innovations = world_single()
    genome_b = copy.deepcopy(genome_a)
    rng = np.random.default_rng(24)
    result_a = add_node(genome_a, rng, innovations, always())
    snapshot = copy.deepcopy(innovations)
    result_b = add_node(genome_b, rng, innovations, always())
    assert split_of(genome_a, result_a) == split_of(genome_b, result_b)
    assert innovations == snapshot


def test_registry_untouched_when_the_dice_say_no():
    genome, innovations = world_one()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(25)
    for _ in range(100):
        add_node(genome, rng, innovations, always(node_add_prob=0.0))
    assert innovations == snapshot


def test_registry_untouched_when_nothing_can_be_split():
    genome, innovations = world_nothing()
    snapshot = copy.deepcopy(innovations)
    rng = np.random.default_rng(26)
    for _ in range(100):
        add_node(genome, rng, innovations, always())
    assert innovations == snapshot


def test_different_connections_get_different_node_ids():
    genome, innovations = world_one()
    rng = np.random.default_rng(27)
    ids = set()
    for _ in range(300):
        result = add_node(genome, rng, innovations, always())
        ids.add(split_of(genome, result)[0])
    # only two connections can be split -> only two node ids exist
    assert len(ids) == 2
    assert len(innovations["splits"]) == 2

###############################################################################
# growing a genome
###############################################################################

def test_many_splits_keep_the_genome_consistent():
    genome, innovations = world_one()
    rng = np.random.default_rng(28)
    cfg = always()
    for _ in range(40):
        genome = add_node(genome, rng, innovations, cfg)

    hidden = [n for n, node in genome["nodes"].items() if node["type"] == "hidden"]
    assert len(hidden) == 40
    assert sorted(genome["nodes"].keys()) == list(range(len(genome["nodes"])))
    for conn in genome["connections"].values():
        assert conn["src"] in genome["nodes"]
        assert conn["dst"] in genome["nodes"]
    # every split adds 2 connections
    assert len(genome["connections"]) == 2 + 2 * 40
    # no cycles: all nodes can be removed in topological order
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
    assert removed == len(genome["nodes"])

###############################################################################
# new genome, nothing else touched
###############################################################################

def test_input_genome_is_never_modified():
    genome, innovations = world_one()
    snapshot = copy.deepcopy(genome)
    rng = np.random.default_rng(29)
    for _ in range(100):
        probe = copy.deepcopy(innovations)
        add_node(genome, rng, probe, always())
    assert genome == snapshot


def test_returns_a_new_object_even_when_nothing_is_added():
    rng = np.random.default_rng(30)
    genome, innovations = world_nothing()
    assert add_node(genome, rng, innovations, always()) is not genome
    genome, innovations = world_one()
    result = add_node(genome, rng, innovations, always(node_add_prob=0.0))
    assert result is not genome
    assert result == genome


def test_result_does_not_share_inner_dicts_with_the_input():
    genome, innovations = world_one()
    rng = np.random.default_rng(31)
    result = add_node(genome, rng, innovations, always())
    assert result["nodes"] is not genome["nodes"]
    assert result["connections"] is not genome["connections"]
    for key, conn in genome["connections"].items():
        assert result["connections"][key] is not conn


def test_sizes_and_other_fields_are_kept():
    genome, innovations = world_one()
    rng = np.random.default_rng(32)
    result = add_node(genome, rng, innovations, always())
    assert result["n_inputs"] == genome["n_inputs"]
    assert result["n_outputs"] == genome["n_outputs"]


def test_cfg_is_not_modified():
    genome, innovations = world_one()
    cfg = always()
    snapshot = copy.deepcopy(cfg)
    rng = np.random.default_rng(33)
    add_node(genome, rng, innovations, cfg)
    assert cfg == snapshot

###############################################################################
# required arguments
###############################################################################

def test_rng_is_required():
    genome, innovations = world_one()
    try:
        add_node(genome, innovations=innovations)
    except TypeError:
        return
    raise AssertionError("add_node must require rng")


def test_innovations_is_required():
    genome, innovations = world_one()
    rng = np.random.default_rng(34)
    try:
        add_node(genome, rng)
    except TypeError:
        return
    raise AssertionError("add_node must require innovations")

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"add_node: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()