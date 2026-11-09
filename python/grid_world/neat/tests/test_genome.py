"""
Created on Sat Oct 17 19:19:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy
import json

import numpy as np

from neat.config import default_config
from neat.genome import init_genome


def make(n_inputs=3, n_outputs=2, seed=0):
    return init_genome(n_inputs, n_outputs, np.random.default_rng(seed))

###############################################################################
# Shape of the genome
###############################################################################

def test_top_level_keys():
    genome = make()
    assert set(genome.keys()) == {"n_inputs", "n_outputs", "nodes", "connections"}


def test_sizes_are_stored():
    genome = make(3, 2)
    assert genome["n_inputs"] == 3
    assert genome["n_outputs"] == 2


def test_nodes_and_connections_are_dicts():
    # catches the dict.update() mistake: nodes must never be None
    genome = make()
    assert isinstance(genome["nodes"], dict)
    assert isinstance(genome["connections"], dict)

###############################################################################
# Nodes
###############################################################################

def test_node_count_and_ids():
    genome = make(3, 2)
    assert len(genome["nodes"]) == 5
    assert sorted(genome["nodes"].keys()) == [0, 1, 2, 3, 4]


def test_inputs_come_first_then_outputs():
    genome = make(3, 2)
    for node_id in range(3):
        assert genome["nodes"][node_id]["type"] == "input"
    for node_id in range(3, 5):
        assert genome["nodes"][node_id]["type"] == "output"


def test_output_nodes_have_bias_and_activation():
    cfg = default_config()
    genome = make(3, 2)
    for node_id in range(3, 5):
        node = genome["nodes"][node_id]
        assert type(node["bias"]) is float
        assert node["activation"] == cfg["output_activation"]


def test_input_nodes_have_no_bias_or_activation():
    genome = make(3, 2)
    for node_id in range(3):
        assert genome["nodes"][node_id] == {"type": "input"}

###############################################################################
# Connections
###############################################################################

def test_connection_count_is_fully_connected():
    genome = make(3, 2)
    assert len(genome["connections"]) == 6


def test_innovation_keys_are_zero_to_n_minus_one():
    genome = make(3, 2)
    assert sorted(genome["connections"].keys()) == [0, 1, 2, 3, 4, 5]


def test_every_connection_goes_from_an_input_to_an_output():
    genome = make(4, 3)
    for connection in genome["connections"].values():
        assert genome["nodes"][connection["src"]]["type"] == "input"
        assert genome["nodes"][connection["dst"]]["type"] == "output"


def test_every_input_output_pair_appears_exactly_once():
    genome = make(4, 3)
    pairs = []
    for connection in genome["connections"].values():
        pairs.append((connection["src"], connection["dst"]))
    expected = []
    for src in range(4):
        for dst in range(4, 7):
            expected.append((src, dst))
    assert sorted(pairs) == expected


def test_innovation_number_follows_position():
    # innovation = src * n_outputs + (dst - n_inputs): fixed by sizes alone
    n_inputs = 4
    n_outputs = 3
    genome = make(n_inputs, n_outputs)
    for innovation, connection in genome["connections"].items():
        expected = connection["src"] * n_outputs + (connection["dst"] - n_inputs)
        assert innovation == expected


def test_connections_start_enabled():
    genome = make()
    for connection in genome["connections"].values():
        assert connection["enabled"] is True


def test_weights_are_plain_floats():
    genome = make()
    for connection in genome["connections"].values():
        assert type(connection["weight"]) is float

###############################################################################
# Randomness
###############################################################################

def test_same_seed_gives_same_genome():
    assert make(seed=7) == make(seed=7)


def test_different_seed_gives_different_weights_and_biases():
    a = make(seed=1)
    b = make(seed=2)
    types_a = [n["type"] for n in a["nodes"].values()]
    types_b = [n["type"] for n in b["nodes"].values()]
    assert types_a == types_b                            # structure identical
    weights_a = [c["weight"] for c in a["connections"].values()]
    weights_b = [c["weight"] for c in b["connections"].values()]
    assert weights_a != weights_b
    biases_a = [n["bias"] for n in a["nodes"].values() if n["type"] == "output"]
    biases_b = [n["bias"] for n in b["nodes"].values() if n["type"] == "output"]
    assert biases_a != biases_b


def test_structure_does_not_depend_on_seed():
    a = make(5, 4, seed=1)
    b = make(5, 4, seed=99)
    pairs_a = [(c["src"], c["dst"]) for c in a["connections"].values()]
    pairs_b = [(c["src"], c["dst"]) for c in b["connections"].values()]
    assert pairs_a == pairs_b


def test_weights_look_like_a_standard_normal():
    genome = make(29, 4, seed=0)                         # 116 weights
    weights = [c["weight"] for c in genome["connections"].values()]
    assert abs(float(np.mean(weights))) < 0.4
    assert 0.7 < float(np.std(weights)) < 1.3

###############################################################################
# Config and default arguments
###############################################################################

def test_weights_follow_the_config_mean_and_stdev():
    cfg = default_config()
    cfg["weight_init_mean"] = 5.0
    cfg["weight_init_stdev"] = 0.0                       # no spread: all weights equal the mean
    genome = init_genome(3, 2, np.random.default_rng(0), cfg)
    for connection in genome["connections"].values():
        assert connection["weight"] == 5.0


def test_larger_stdev_gives_a_larger_spread():
    narrow = default_config()
    narrow["weight_init_stdev"] = 0.1
    wide = default_config()
    wide["weight_init_stdev"] = 5.0
    g_narrow = init_genome(29, 4, np.random.default_rng(0), narrow)
    g_wide = init_genome(29, 4, np.random.default_rng(0), wide)
    std_narrow = np.std([c["weight"] for c in g_narrow["connections"].values()])
    std_wide = np.std([c["weight"] for c in g_wide["connections"].values()])
    assert std_wide > 10 * std_narrow


def test_cfg_is_not_modified():
    cfg = default_config()
    before = dict(cfg)
    init_genome(3, 2, np.random.default_rng(0), cfg)
    assert cfg == before


def test_no_cfg_means_default_config():
    explicit = init_genome(3, 2, np.random.default_rng(4), default_config())
    implicit = init_genome(3, 2, np.random.default_rng(4))
    assert explicit == implicit


def test_rng_is_required():
    try:
        init_genome(3, 2)
    except TypeError:
        return
    raise AssertionError("expected TypeError: rng has no default")


def test_biases_follow_the_config_mean_and_stdev():
    cfg = default_config()
    cfg["bias_init_mean"] = 2.0
    cfg["bias_init_stdev"] = 0.0                         # no spread: every bias equals the mean
    genome = init_genome(3, 2, np.random.default_rng(0), cfg)
    for node_id in range(3, 5):
        assert genome["nodes"][node_id]["bias"] == 2.0


def test_output_activation_follows_the_config():
    cfg = default_config()
    cfg["output_activation"] = "relu"
    genome = init_genome(3, 2, np.random.default_rng(0), cfg)
    for node_id in range(3, 5):
        assert genome["nodes"][node_id]["activation"] == "relu"


def test_input_nodes_ignore_bias_config():
    cfg = default_config()
    cfg["bias_init_mean"] = 2.0
    genome = init_genome(3, 2, np.random.default_rng(0), cfg)
    for node_id in range(3):
        assert genome["nodes"][node_id] == {"type": "input"}


def test_explicit_rng_is_consumed_so_next_genome_differs():
    rng = np.random.default_rng(0)
    a = init_genome(3, 2, rng)
    b = init_genome(3, 2, rng)
    assert a["connections"][0]["weight"] != b["connections"][0]["weight"]

###############################################################################
# Plain data
###############################################################################

def test_genome_survives_json_round_trip():
    genome = make(29, 4)
    text = json.dumps(genome)
    back = json.loads(text)
    # json turns integer keys into strings; compare after normalizing them
    assert len(back["nodes"]) == len(genome["nodes"])
    assert len(back["connections"]) == len(genome["connections"])


def test_deepcopy_is_independent():
    genome = make()
    clone = copy.deepcopy(genome)
    clone["connections"][0]["weight"] = 123.0
    assert genome["connections"][0]["weight"] != 123.0


def test_two_calls_do_not_share_state():
    a = make(seed=0)
    b = make(seed=0)
    a["nodes"][0]["type"] = "changed"
    assert b["nodes"][0]["type"] == "input"

###############################################################################
# Realistic size
###############################################################################

def test_project_size_29_inputs_4_outputs():
    genome = make(29, 4)
    assert len(genome["nodes"]) == 33
    assert len(genome["connections"]) == 116

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"genome: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()