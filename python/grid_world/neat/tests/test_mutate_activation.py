"""
Created on Sat Oct 17 22:15:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.genome import init_genome
from neat.mutation import mutate_activation


def make_genome(n_inputs=3, n_outputs=2, seed=0):
    return init_genome(n_inputs, n_outputs, np.random.default_rng(seed))


def make_cfg(**changes):
    cfg = default_config()
    cfg.update(changes)
    return cfg


def activations_of(genome):
    names = []
    for node_id in sorted(genome["nodes"]):
        node = genome["nodes"][node_id]
        if node["type"] != "input":
            names.append(node["activation"])
    return names


def set_all_activations(genome, name):
    for node in genome["nodes"].values():
        if node["type"] != "input":
            node["activation"] = name
    return genome


def with_hidden_node(genome, activation="tanh"):
    genome = copy.deepcopy(genome)
    hidden_id = max(genome["nodes"]) + 1
    genome["nodes"][hidden_id] = {"type": "hidden", "bias": 0.3, "activation": activation}
    return genome, hidden_id

###############################################################################
# The original genome is never touched
###############################################################################

def test_returns_a_new_genome():
    genome = make_genome()
    mutated = mutate_activation(genome, np.random.default_rng(1))
    assert mutated is not genome
    assert mutated["nodes"] is not genome["nodes"]
    assert mutated["connections"] is not genome["connections"]


def test_original_is_not_modified():
    genome = make_genome()
    snapshot = copy.deepcopy(genome)
    cfg = make_cfg(activation_mutate_prob=1.0)
    mutate_activation(genome, np.random.default_rng(1), cfg)
    assert genome == snapshot


def test_changing_the_result_does_not_change_the_original():
    genome = make_genome()
    snapshot = copy.deepcopy(genome)
    mutated = mutate_activation(genome, np.random.default_rng(1))
    mutated["nodes"][3]["activation"] = "changed"
    mutated["connections"][0]["weight"] = 999.0
    assert genome == snapshot

###############################################################################
# Only activations change
###############################################################################

def test_everything_but_activations_is_unchanged():
    genome, _ = with_hidden_node(make_genome(5, 3))
    cfg = make_cfg(activation_mutate_prob=1.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    assert mutated["connections"] == genome["connections"]
    assert mutated["n_inputs"] == genome["n_inputs"]
    assert mutated["n_outputs"] == genome["n_outputs"]
    assert sorted(mutated["nodes"].keys()) == sorted(genome["nodes"].keys())
    for node_id, node in genome["nodes"].items():
        new_node = mutated["nodes"][node_id]
        assert new_node["type"] == node["type"]
        if node["type"] != "input":
            assert new_node["bias"] == node["bias"]


def test_input_nodes_are_never_touched():
    genome = make_genome(5, 3)
    cfg = make_cfg(activation_mutate_prob=1.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    for node_id in range(5):
        assert mutated["nodes"][node_id] == {"type": "input"}


def test_hidden_nodes_are_mutated_too():
    genome, hidden_id = with_hidden_node(make_genome())
    cfg = make_cfg(activation_mutate_prob=1.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    assert mutated["nodes"][hidden_id]["activation"] != "tanh"
    assert mutated["nodes"][hidden_id]["type"] == "hidden"

###############################################################################
# activation_mutate_prob
###############################################################################

def test_probability_zero_changes_nothing():
    genome, _ = with_hidden_node(make_genome(5, 3))
    cfg = make_cfg(activation_mutate_prob=0.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    assert mutated == genome


def test_probability_one_changes_every_activation():
    genome = set_all_activations(make_genome(29, 4), "tanh")
    cfg = make_cfg(activation_mutate_prob=1.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    for old, new in zip(activations_of(genome), activations_of(mutated)):
        assert old != new


def test_probability_half_changes_about_half_of_the_nodes():
    genome = set_all_activations(make_genome(2, 500), "tanh")   # 500 output nodes
    cfg = make_cfg(activation_mutate_prob=0.5)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    changed = 0
    for old, new in zip(activations_of(genome), activations_of(mutated)):
        if old != new:
            changed += 1
    assert 0.4 < changed / 500 < 0.6

###############################################################################
# Which activation is chosen
###############################################################################

def test_new_activation_is_always_one_of_the_options():
    genome = make_genome(5, 3)
    cfg = make_cfg(activation_mutate_prob=1.0)
    rng = np.random.default_rng(1)
    for _ in range(100):
        genome = mutate_activation(genome, rng, cfg)
        for name in activations_of(genome):
            assert name in cfg["activation_options"]


def test_new_activation_is_a_plain_string():
    genome = make_genome(5, 3)
    cfg = make_cfg(activation_mutate_prob=1.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    for name in activations_of(mutated):
        assert type(name) is str


def test_choice_is_spread_over_the_other_options():
    options = ("a", "b", "c", "d")
    genome = set_all_activations(make_genome(2, 600), "a")      # 600 output nodes
    cfg = make_cfg(activation_mutate_prob=1.0, activation_options=options,
                   output_activation="a", hidden_activation="a")
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    counts = {"a": 0, "b": 0, "c": 0, "d": 0}
    for name in activations_of(mutated):
        counts[name] += 1
    assert counts["a"] == 0                                      # never the same one
    for name in ("b", "c", "d"):
        assert 150 < counts[name] < 250                          # about 200 each


def test_only_one_alternative_is_always_chosen():
    genome = set_all_activations(make_genome(3, 5), "x")
    cfg = make_cfg(activation_mutate_prob=1.0, activation_options=("x", "y"),
                   output_activation="x", hidden_activation="x")
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    for name in activations_of(mutated):
        assert name == "y"


def test_single_option_equal_to_current_changes_nothing():
    genome = set_all_activations(make_genome(3, 5), "x")
    cfg = make_cfg(activation_mutate_prob=1.0, activation_options=("x",),
                   output_activation="x", hidden_activation="x")
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    assert mutated == genome


def test_current_activation_outside_the_options_is_replaced():
    genome = set_all_activations(make_genome(3, 5), "tanh")
    cfg = make_cfg(activation_mutate_prob=1.0, activation_options=("relu", "sigmoid"),
                   output_activation="relu", hidden_activation="relu")
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    for name in activations_of(mutated):
        assert name in ("relu", "sigmoid")

def test_a_node_with_nothing_to_choose_does_not_stop_the_others():
    # options hold only "x": node 3 already has it (nothing to choose from),
    # node 4 has "tanh" (it can switch to "x"). The stuck node comes first.
    genome = make_genome(3, 2)
    genome["nodes"][3]["activation"] = "x"
    genome["nodes"][4]["activation"] = "tanh"
    cfg = make_cfg(activation_mutate_prob=1.0, activation_options=("x",),
                   output_activation="x", hidden_activation="x")
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    assert mutated["nodes"][3]["activation"] == "x"
    assert mutated["nodes"][4]["activation"] == "x"


###############################################################################
# Uses the activation key, not the other probabilities
###############################################################################

def test_other_probabilities_do_not_affect_activations():
    genome = make_genome(5, 3)
    cfg = make_cfg(activation_mutate_prob=0.0, weight_mutate_prob=1.0,
                   bias_mutate_prob=1.0, conn_add_prob=1.0, node_add_prob=1.0)
    mutated = mutate_activation(genome, np.random.default_rng(1), cfg)
    assert mutated == genome

###############################################################################
# Randomness and arguments
###############################################################################

def test_same_seed_gives_same_result():
    genome = make_genome(29, 4)
    cfg = make_cfg(activation_mutate_prob=0.5)
    a = mutate_activation(genome, np.random.default_rng(5), cfg)
    b = mutate_activation(genome, np.random.default_rng(5), cfg)
    assert a == b


def test_different_seed_gives_different_result():
    genome = set_all_activations(make_genome(2, 200), "tanh")
    cfg = make_cfg(activation_mutate_prob=0.5)
    a = mutate_activation(genome, np.random.default_rng(5), cfg)
    b = mutate_activation(genome, np.random.default_rng(6), cfg)
    assert activations_of(a) != activations_of(b)


def test_cfg_is_not_modified():
    genome = make_genome()
    cfg = make_cfg(activation_mutate_prob=1.0)
    before = copy.deepcopy(cfg)
    mutate_activation(genome, np.random.default_rng(1), cfg)
    assert cfg == before


def test_no_cfg_means_default_config():
    genome = set_all_activations(make_genome(2, 200), "tanh")
    explicit = mutate_activation(genome, np.random.default_rng(3), default_config())
    implicit = mutate_activation(genome, np.random.default_rng(3))
    assert explicit == implicit


def test_rng_is_required():
    genome = make_genome()
    try:
        mutate_activation(genome)
    except TypeError:
        return
    raise AssertionError("expected TypeError: rng has no default")

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"mutate_activation: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()