"""
Created on Sat Oct 17 20:56:45 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.config import default_config
from neat.genome import init_genome
from neat.mutation import mutate_weights


def make_genome(n_inputs=3, n_outputs=2, seed=0):
    return init_genome(n_inputs, n_outputs, np.random.default_rng(seed))


def make_cfg(**changes):
    cfg = default_config()
    cfg.update(changes)
    return cfg


def weights_of(genome):
    weights = []
    for innovation in sorted(genome["connections"]):
        weights.append(genome["connections"][innovation]["weight"])
    return weights

###############################################################################
# The original genome is never touched
###############################################################################

def test_returns_a_new_genome():
    genome = make_genome()
    mutated = mutate_weights(genome, np.random.default_rng(1))
    assert mutated is not genome
    assert mutated["connections"] is not genome["connections"]
    assert mutated["nodes"] is not genome["nodes"]


def test_original_is_not_modified():
    genome = make_genome()
    snapshot = copy.deepcopy(genome)
    cfg = make_cfg(weight_mutate_prob=1.0)
    mutate_weights(genome, np.random.default_rng(1), cfg)
    assert genome == snapshot


def test_changing_the_result_does_not_change_the_original():
    genome = make_genome()
    snapshot = copy.deepcopy(genome)
    mutated = mutate_weights(genome, np.random.default_rng(1))
    mutated["connections"][0]["weight"] = 999.0
    mutated["nodes"][3]["bias"] = 999.0
    assert genome == snapshot

###############################################################################
# Only weights change
###############################################################################

def test_structure_and_biases_are_unchanged():
    genome = make_genome(5, 3)
    cfg = make_cfg(weight_mutate_prob=1.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    assert mutated["nodes"] == genome["nodes"]           # includes the biases
    assert mutated["n_inputs"] == genome["n_inputs"]
    assert mutated["n_outputs"] == genome["n_outputs"]
    assert sorted(mutated["connections"].keys()) == sorted(genome["connections"].keys())
    for innovation, connection in genome["connections"].items():
        new_connection = mutated["connections"][innovation]
        assert new_connection["src"] == connection["src"]
        assert new_connection["dst"] == connection["dst"]
        assert new_connection["enabled"] == connection["enabled"]


def test_weights_stay_plain_floats():
    genome = make_genome()
    cfg = make_cfg(weight_mutate_prob=1.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    for connection in mutated["connections"].values():
        assert type(connection["weight"]) is float


def test_genome_without_connections_is_returned_equal():
    genome = make_genome()
    genome["connections"] = {}
    mutated = mutate_weights(genome, np.random.default_rng(1))
    assert mutated == genome
    assert mutated is not genome

###############################################################################
# weight_mutate_prob
###############################################################################

def test_probability_zero_changes_nothing():
    genome = make_genome(29, 4)
    cfg = make_cfg(weight_mutate_prob=0.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    assert mutated == genome


def test_probability_one_touches_every_weight():
    genome = make_genome(29, 4)
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                   weight_perturb_power=0.5)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    changed = 0
    for old, new in zip(weights_of(genome), weights_of(mutated)):
        if old != new:
            changed += 1
    assert changed == 116


def test_probability_half_touches_about_half_of_the_weights():
    genome = make_genome(50, 10)                         # 500 weights
    cfg = make_cfg(weight_mutate_prob=0.5, weight_perturb_prob=1.0,
                   weight_perturb_power=0.5)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    changed = 0
    for old, new in zip(weights_of(genome), weights_of(mutated)):
        if old != new:
            changed += 1
    assert 0.4 < changed / 500 < 0.6

###############################################################################
# Perturbation (a small nudge)
###############################################################################

def test_zero_power_nudge_changes_nothing():
    genome = make_genome(29, 4)
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                   weight_perturb_power=0.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    assert weights_of(mutated) == weights_of(genome)


def test_nudge_size_follows_the_power():
    genome = make_genome(50, 10)                         # 500 weights
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                   weight_perturb_power=0.5, weight_min_value=-1000.0,
                   weight_max_value=1000.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    differences = []
    for old, new in zip(weights_of(genome), weights_of(mutated)):
        differences.append(new - old)
    assert abs(float(np.mean(differences))) < 0.1        # nudges go both ways
    assert 0.4 < float(np.std(differences)) < 0.6        # spread is about the power


def test_larger_power_gives_larger_nudges():
    genome = make_genome(50, 10)
    small = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                     weight_perturb_power=0.1)
    large = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                     weight_perturb_power=2.0)
    m_small = mutate_weights(genome, np.random.default_rng(1), small)
    m_large = mutate_weights(genome, np.random.default_rng(1), large)
    d_small = np.std(np.array(weights_of(m_small)) - np.array(weights_of(genome)))
    d_large = np.std(np.array(weights_of(m_large)) - np.array(weights_of(genome)))
    assert d_large > 10 * d_small

###############################################################################
# Replacement (a fresh random value)
###############################################################################

def test_replacement_draws_from_the_init_distribution():
    genome = make_genome(29, 4)
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=0.0,
                   weight_init_mean=5.0, weight_init_stdev=0.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    for weight in weights_of(mutated):
        assert weight == 5.0


def test_replacement_ignores_the_old_weight():
    genome = make_genome(29, 4)
    for connection in genome["connections"].values():
        connection["weight"] = 12.0
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=0.0,
                   weight_init_mean=0.0, weight_init_stdev=1.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    assert abs(float(np.mean(weights_of(mutated)))) < 0.4
    assert max(weights_of(mutated)) < 6.0

###############################################################################
# Clipping
###############################################################################

def test_replacement_is_clipped_to_the_maximum():
    genome = make_genome()
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=0.0,
                   weight_init_mean=10.0, weight_init_stdev=0.0,
                   weight_min_value=-1.0, weight_max_value=1.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    for weight in weights_of(mutated):
        assert weight == 1.0


def test_replacement_is_clipped_to_the_minimum():
    genome = make_genome()
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=0.0,
                   weight_init_mean=-10.0, weight_init_stdev=0.0,
                   weight_min_value=-1.0, weight_max_value=1.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    for weight in weights_of(mutated):
        assert weight == -1.0


def test_nudges_are_clipped_too():
    genome = make_genome(29, 4)
    for connection in genome["connections"].values():
        connection["weight"] = 0.99
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                   weight_perturb_power=5.0,
                   weight_min_value=-1.0, weight_max_value=1.0)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    for weight in weights_of(mutated):
        assert -1.0 <= weight <= 1.0


def test_weights_stay_in_range_after_many_mutations():
    genome = make_genome(5, 3)
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_power=3.0,
                   weight_min_value=-2.0, weight_max_value=2.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        genome = mutate_weights(genome, rng, cfg)
    for weight in weights_of(genome):
        assert -2.0 <= weight <= 2.0

###############################################################################
# Disabled connections
###############################################################################

def test_disabled_connections_are_mutated_and_stay_disabled():
    genome = make_genome(3, 2)
    for connection in genome["connections"].values():
        connection["enabled"] = False
    cfg = make_cfg(weight_mutate_prob=1.0, weight_perturb_prob=1.0,
                   weight_perturb_power=0.5)
    mutated = mutate_weights(genome, np.random.default_rng(1), cfg)
    for innovation, connection in mutated["connections"].items():
        assert connection["enabled"] is False
        assert connection["weight"] != genome["connections"][innovation]["weight"]

###############################################################################
# Randomness and arguments
###############################################################################

def test_same_seed_gives_same_result():
    genome = make_genome(29, 4)
    a = mutate_weights(genome, np.random.default_rng(5))
    b = mutate_weights(genome, np.random.default_rng(5))
    assert a == b


def test_different_seed_gives_different_result():
    genome = make_genome(29, 4)
    a = mutate_weights(genome, np.random.default_rng(5))
    b = mutate_weights(genome, np.random.default_rng(6))
    assert weights_of(a) != weights_of(b)


def test_cfg_is_not_modified():
    genome = make_genome()
    cfg = make_cfg(weight_mutate_prob=1.0)
    before = dict(cfg)
    mutate_weights(genome, np.random.default_rng(1), cfg)
    assert cfg == before


def test_no_cfg_means_default_config():
    genome = make_genome(29, 4)
    explicit = mutate_weights(genome, np.random.default_rng(3), default_config())
    implicit = mutate_weights(genome, np.random.default_rng(3))
    assert explicit == implicit


def test_rng_is_required():
    genome = make_genome()
    try:
        mutate_weights(genome)
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
    print(f"mutation: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()