
"""
Created on Sat Oct 17 21:18:37 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.genome import init_genome
from neat.mutation import mutate_biases


def make_genome(n_inputs=3, n_outputs=2, seed=0):
    return init_genome(n_inputs, n_outputs, np.random.default_rng(seed))


def make_cfg(**changes):
    cfg = default_config()
    cfg.update(changes)
    return cfg


def biases_of(genome):
    biases = []
    for node_id in sorted(genome["nodes"]):
        node = genome["nodes"][node_id]
        if node["type"] != "input":
            biases.append(node["bias"])
    return biases


def with_hidden_node(genome, bias=0.3):
    genome = copy.deepcopy(genome)
    hidden_id = max(genome["nodes"]) + 1
    genome["nodes"][hidden_id] = {"type": "hidden", "bias": bias, "activation": "tanh"}
    return genome, hidden_id

###############################################################################
# The original genome is never touched
###############################################################################

def test_returns_a_new_genome():
    genome = make_genome()
    mutated = mutate_biases(genome, np.random.default_rng(1))
    assert mutated is not genome
    assert mutated["nodes"] is not genome["nodes"]
    assert mutated["connections"] is not genome["connections"]


def test_original_is_not_modified():
    genome = make_genome()
    snapshot = copy.deepcopy(genome)
    cfg = make_cfg(bias_mutate_prob=1.0)
    mutate_biases(genome, np.random.default_rng(1), cfg)
    assert genome == snapshot


def test_changing_the_result_does_not_change_the_original():
    genome = make_genome()
    snapshot = copy.deepcopy(genome)
    mutated = mutate_biases(genome, np.random.default_rng(1))
    mutated["nodes"][3]["bias"] = 999.0
    mutated["connections"][0]["weight"] = 999.0
    assert genome == snapshot

###############################################################################
# Only biases change
###############################################################################

def test_connections_and_sizes_are_unchanged():
    genome = make_genome(5, 3)
    cfg = make_cfg(bias_mutate_prob=1.0, weight_mutate_prob=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert mutated["connections"] == genome["connections"]
    assert mutated["n_inputs"] == genome["n_inputs"]
    assert mutated["n_outputs"] == genome["n_outputs"]


def test_types_and_activations_are_unchanged():
    genome = make_genome(5, 3)
    cfg = make_cfg(bias_mutate_prob=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert sorted(mutated["nodes"].keys()) == sorted(genome["nodes"].keys())
    for node_id, node in genome["nodes"].items():
        assert mutated["nodes"][node_id]["type"] == node["type"]
        if node["type"] != "input":
            assert mutated["nodes"][node_id]["activation"] == node["activation"]


def test_input_nodes_are_never_touched():
    genome = make_genome(5, 3)
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=0.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for node_id in range(5):
        assert mutated["nodes"][node_id] == {"type": "input"}


def test_biases_stay_plain_floats():
    genome = make_genome()
    cfg = make_cfg(bias_mutate_prob=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for bias in biases_of(mutated):
        assert type(bias) is float


def test_hidden_nodes_are_mutated_too():
    genome, hidden_id = with_hidden_node(make_genome())
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=1.0,
                   bias_perturb_power=0.5)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert mutated["nodes"][hidden_id]["bias"] != 0.3
    assert mutated["nodes"][hidden_id]["type"] == "hidden"

###############################################################################
# bias_mutate_prob
###############################################################################

def test_probability_zero_changes_nothing():
    genome, _ = with_hidden_node(make_genome(5, 3))
    cfg = make_cfg(bias_mutate_prob=0.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert mutated == genome


def test_probability_one_touches_every_bias():
    genome = make_genome(29, 4)
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=1.0,
                   bias_perturb_power=0.5)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for old, new in zip(biases_of(genome), biases_of(mutated)):
        assert old != new


def test_probability_half_touches_about_half_of_the_biases():
    genome = make_genome(2, 500)                         # 500 output nodes
    cfg = make_cfg(bias_mutate_prob=0.5, bias_perturb_prob=1.0,
                   bias_perturb_power=0.5)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    changed = 0
    for old, new in zip(biases_of(genome), biases_of(mutated)):
        if old != new:
            changed += 1
    assert 0.4 < changed / 500 < 0.6

###############################################################################
# Perturbation (a small nudge)
###############################################################################

def test_zero_power_nudge_changes_nothing():
    genome = make_genome(29, 4)
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=1.0,
                   bias_perturb_power=0.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert biases_of(mutated) == biases_of(genome)


def test_nudge_size_follows_the_power():
    genome = make_genome(2, 500)
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=1.0,
                   bias_perturb_power=0.5, bias_min_value=-1000.0,
                   bias_max_value=1000.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    differences = np.array(biases_of(mutated)) - np.array(biases_of(genome))
    assert abs(float(np.mean(differences))) < 0.1        # nudges go both ways
    assert 0.4 < float(np.std(differences)) < 0.6        # spread is about the power

###############################################################################
# Replacement (a fresh random value)
###############################################################################

def test_replacement_draws_from_the_bias_init_distribution():
    genome, hidden_id = with_hidden_node(make_genome(29, 4))
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=0.0,
                   bias_init_mean=3.0, bias_init_stdev=0.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for bias in biases_of(mutated):
        assert bias == 3.0


def test_replacement_ignores_the_old_bias():
    genome = make_genome(2, 500)
    for node in genome["nodes"].values():
        if node["type"] != "input":
            node["bias"] = 12.0
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=0.0,
                   bias_init_mean=0.0, bias_init_stdev=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert abs(float(np.mean(biases_of(mutated)))) < 0.2
    assert max(biases_of(mutated)) < 6.0

###############################################################################
# Clipping
###############################################################################

def test_replacement_is_clipped_to_the_maximum():
    genome = make_genome()
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=0.0,
                   bias_init_mean=10.0, bias_init_stdev=0.0,
                   bias_min_value=-1.0, bias_max_value=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for bias in biases_of(mutated):
        assert bias == 1.0


def test_replacement_is_clipped_to_the_minimum():
    genome = make_genome()
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=0.0,
                   bias_init_mean=-10.0, bias_init_stdev=0.0,
                   bias_min_value=-1.0, bias_max_value=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for bias in biases_of(mutated):
        assert bias == -1.0


def test_nudges_are_clipped_too():
    genome = make_genome(2, 200)
    for node in genome["nodes"].values():
        if node["type"] != "input":
            node["bias"] = 0.99
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=1.0,
                   bias_perturb_power=5.0,
                   bias_min_value=-1.0, bias_max_value=1.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for bias in biases_of(mutated):
        assert -1.0 <= bias <= 1.0


def test_biases_stay_in_range_after_many_mutations():
    genome = make_genome(5, 3)
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_power=3.0,
                   bias_min_value=-2.0, bias_max_value=2.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        genome = mutate_biases(genome, rng, cfg)
    for bias in biases_of(genome):
        assert -2.0 <= bias <= 2.0

###############################################################################
# Uses the bias keys, not the weight keys
###############################################################################

def test_weight_settings_do_not_affect_biases():
    genome = make_genome(5, 3)
    cfg = make_cfg(bias_mutate_prob=0.0, weight_mutate_prob=1.0,
                   weight_perturb_prob=0.0, weight_init_mean=7.0,
                   weight_init_stdev=0.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    assert mutated == genome


def test_bias_bounds_are_used_not_weight_bounds():
    genome = make_genome()
    cfg = make_cfg(bias_mutate_prob=1.0, bias_perturb_prob=0.0,
                   bias_init_mean=10.0, bias_init_stdev=0.0,
                   bias_min_value=-1.0, bias_max_value=1.0,
                   weight_min_value=-30.0, weight_max_value=30.0)
    mutated = mutate_biases(genome, np.random.default_rng(1), cfg)
    for bias in biases_of(mutated):
        assert bias == 1.0

###############################################################################
# Randomness and arguments
###############################################################################

def test_same_seed_gives_same_result():
    genome = make_genome(29, 4)
    a = mutate_biases(genome, np.random.default_rng(5))
    b = mutate_biases(genome, np.random.default_rng(5))
    assert a == b


def test_different_seed_gives_different_result():
    genome = make_genome(29, 4)
    a = mutate_biases(genome, np.random.default_rng(5))
    b = mutate_biases(genome, np.random.default_rng(6))
    assert biases_of(a) != biases_of(b)


def test_cfg_is_not_modified():
    genome = make_genome()
    cfg = make_cfg(bias_mutate_prob=1.0)
    before = dict(cfg)
    mutate_biases(genome, np.random.default_rng(1), cfg)
    assert cfg == before


def test_no_cfg_means_default_config():
    genome = make_genome(29, 4)
    explicit = mutate_biases(genome, np.random.default_rng(3), default_config())
    implicit = mutate_biases(genome, np.random.default_rng(3))
    assert explicit == implicit


def test_rng_is_required():
    genome = make_genome()
    try:
        mutate_biases(genome)
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
    print(f"mutate_biases: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()