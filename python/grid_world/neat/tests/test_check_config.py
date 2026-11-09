"""
Created on Tue Oct 20 09:00:39 2026

@author: Angelo Antonio Manzatto
"""

"""
Tests for neat/check_config.py: check_config(cfg).

check_config returns None for a good config and raises ValueError for a bad
one. The message lists EVERY problem (one per line, starting with the key
name), so you can fix them all in one go.

Refused:
  - not a dict; missing keys; unknown keys (typos)
  - wrong types (bool where a number is expected, text for a probability...)
  - out of range (probability outside [0, 1], population_size < 2,
    survival_threshold of 0 or above 1, weight_min_value >= weight_max_value...)
  - NaN or infinite numbers
  - unknown activation names, empty or duplicated activation_options
  - things the library does not implement yet:
      fitness_maximize False, conn/node_delete_prob > 0, n_initial_hidden > 0

Accepted: every value default_config() gives; numpy numbers; ints where a float
is fine; fitness_threshold None; seed None or a non-negative integer; tuple or
list for activation_options.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from neat.activations import ACTIVATIONS
from neat.check_config import check_config
from neat.config import default_config

###############################################################################
# helpers
###############################################################################

def refused(**changes):
    """Message of the ValueError for default_config() with changes, or None if accepted."""
    cfg = default_config()
    cfg.update(changes)
    try:
        check_config(cfg)
    except ValueError as error:
        return str(error)
    return None


def accepted(**changes):
    message = refused(**changes)
    assert message is None, message


def must_refuse(key, *values):
    for value in values:
        message = refused(**{key: value})
        assert message is not None, (key, value, "was accepted")
        assert key in message, (key, value, message)

###############################################################################
# the whole config
###############################################################################

def test_default_config_is_accepted_and_returns_none():
    assert check_config(default_config()) is None


def test_the_config_is_not_changed():
    cfg = default_config()
    snapshot = copy.deepcopy(cfg)
    check_config(cfg)
    assert cfg == snapshot


def test_not_a_dict_is_refused():
    for bad in (None, [], "cfg", 3):
        try:
            check_config(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must raise ValueError")


def test_missing_key_is_refused_and_named():
    for key in default_config():
        cfg = default_config()
        del cfg[key]
        try:
            check_config(cfg)
        except ValueError as error:
            assert f"{key}: missing" in str(error), key
            continue
        raise AssertionError(f"missing {key} must raise ValueError")


def test_unknown_key_is_refused_and_named():
    cfg = default_config()
    cfg["populaton_size"] = 10
    try:
        check_config(cfg)
    except ValueError as error:
        assert "populaton_size: unknown key" in str(error)
        return
    raise AssertionError("unknown key must raise ValueError")


def test_all_problems_are_listed_together():
    message = refused(population_size=1, mate_only_prob=2.0, output_activation="banana")
    assert "population_size" in message
    assert "mate_only_prob" in message
    assert "output_activation" in message
    assert message.count("\n") >= 3


def test_a_typo_and_a_missing_key_show_up_together():
    cfg = default_config()
    value = cfg.pop("population_size")
    cfg["populaton_size"] = value
    try:
        check_config(cfg)
    except ValueError as error:
        assert "population_size: missing" in str(error)
        assert "populaton_size: unknown key" in str(error)
        return
    raise AssertionError("must raise ValueError")

###############################################################################
# integers
###############################################################################

def test_population_size():
    accepted(population_size=2)
    accepted(population_size=500)
    accepted(population_size=np.int64(50))
    must_refuse("population_size", 1, 0, -5, 10.0, 10.5, "150", None, True, float("nan"))


def test_max_generations():
    accepted(max_generations=1)
    must_refuse("max_generations", 0, -1, 2.0, "10", None, True)


def test_other_integers():
    accepted(compatibility_n_min=1, elitism_min_species_size=1, elitism_count=0,
             min_species_to_keep=1, max_stagnation=1, report_every=0)
    must_refuse("compatibility_n_min", 0, 1.5)
    must_refuse("elitism_min_species_size", 0, 2.0)
    must_refuse("elitism_count", -1, 1.0, True)
    must_refuse("min_species_to_keep", 0, 1.5)
    must_refuse("max_stagnation", 0, 3.0)
    must_refuse("report_every", -1, 1.5, False)

###############################################################################
# probabilities
###############################################################################

PROBABILITIES = ["initial_conn_prob", "weight_mutate_prob", "weight_perturb_prob",
                 "bias_mutate_prob", "bias_perturb_prob", "conn_add_prob",
                 "node_add_prob", "activation_mutate_prob", "mate_only_prob"]


def test_probabilities_accept_zero_one_and_between():
    for key in PROBABILITIES:
        for value in (0.0, 0.5, 1.0, 0, 1, np.float64(0.3)):
            accepted(**{key: value})


def test_probabilities_refuse_outside_0_1_and_wrong_types():
    for key in PROBABILITIES:
        must_refuse(key, -0.01, 1.01, 5, float("nan"), float("inf"), "0.5", None, True)

###############################################################################
# real numbers
###############################################################################

def test_init_mean_and_stdev():
    accepted(weight_init_mean=-2.0, weight_init_stdev=0.0, bias_init_mean=3, bias_init_stdev=2)
    must_refuse("weight_init_mean", float("nan"), float("inf"), "0", None, True)
    must_refuse("weight_init_stdev", -0.1, float("nan"), "1", None)
    must_refuse("bias_init_mean", float("nan"), None)
    must_refuse("bias_init_stdev", -1.0, float("inf"))


def test_perturb_powers():
    accepted(weight_perturb_power=0.0, bias_perturb_power=2.0)
    must_refuse("weight_perturb_power", -0.1, float("nan"), None)
    must_refuse("bias_perturb_power", -0.1, float("inf"), "0.5")


def test_weight_and_bias_limits():
    accepted(weight_min_value=-5.0, weight_max_value=5.0)
    accepted(bias_min_value=-1, bias_max_value=1)
    must_refuse("weight_min_value", 30.0, 31.0, float("nan"), "a")
    must_refuse("weight_max_value", -30.0, -31.0, float("inf"))
    must_refuse("bias_min_value", 30.0, 100.0)
    must_refuse("bias_max_value", -30.0, None)


def test_limits_must_be_ordered_even_when_both_are_valid_numbers():
    message = refused(weight_min_value=4.0, weight_max_value=2.0)
    assert "weight_min_value" in message and "weight_max_value" in message
    message = refused(bias_min_value=1.0, bias_max_value=1.0)
    assert "bias_min_value" in message


def test_compatibility_numbers():
    accepted(compatibility_threshold=0.1, compatibility_c1=0.0, compatibility_c2=0.0, compatibility_c3=0.0)
    must_refuse("compatibility_threshold", 0.0, -1.0, float("nan"), None)
    must_refuse("compatibility_c1", -0.1, float("inf"))
    must_refuse("compatibility_c2", -1.0, "1")
    must_refuse("compatibility_c3", -0.4, None)


def test_survival_threshold():
    accepted(survival_threshold=0.01)
    accepted(survival_threshold=1.0)
    must_refuse("survival_threshold", 0.0, -0.1, 1.01, float("nan"), None)


def test_fitness_threshold():
    accepted(fitness_threshold=None)
    accepted(fitness_threshold=0.95)
    accepted(fitness_threshold=-3)
    must_refuse("fitness_threshold", float("nan"), float("inf"), "3.9", True)

###############################################################################
# identity
###############################################################################

def test_seed():
    accepted(seed=None)
    accepted(seed=0)
    accepted(seed=12345)
    must_refuse("seed", -1, 1.5, "7", True)


def test_experiment_name():
    accepted(experiment_name="player_v1")
    must_refuse("experiment_name", 5, None, ["a"])

###############################################################################
# activations
###############################################################################

def test_activation_names():
    for name in ACTIVATIONS:
        accepted(output_activation=name, hidden_activation=name)
    must_refuse("output_activation", "banana", "", "Sigmoid", None, 3)
    must_refuse("hidden_activation", "banana", "", None)


def test_the_error_lists_the_valid_activations():
    assert "sigmoid" in refused(output_activation="banana")


def test_activation_options():
    accepted(activation_options=("sigmoid",))
    accepted(activation_options=["sigmoid", "tanh"])
    accepted(activation_options=tuple(ACTIVATIONS))
    must_refuse("activation_options", (), [], ("sigmoid", "banana"), ("sigmoid", "sigmoid"),
                "sigmoid", None, ("sigmoid", 3))

###############################################################################
# not implemented yet
###############################################################################

def test_minimizing_is_refused_with_a_hint():
    accepted(fitness_maximize=True)
    message = refused(fitness_maximize=False)
    assert message is not None and "fitness_maximize" in message and "negate" in message
    must_refuse("fitness_maximize", 1, None, "True")


def test_delete_mutations_are_refused():
    accepted(conn_delete_prob=0.0, node_delete_prob=0.0)
    accepted(conn_delete_prob=0, node_delete_prob=0)
    must_refuse("conn_delete_prob", 0.01, 1.0, -0.1, None)
    must_refuse("node_delete_prob", 0.01, 1.0, None)


def test_initial_hidden_nodes_are_refused():
    accepted(n_initial_hidden=0)
    must_refuse("n_initial_hidden", 1, 3, -1, 0.0, None)

###############################################################################
# a few real configs
###############################################################################

def test_configs_used_in_the_other_tests_are_accepted():
    cfg = default_config()
    cfg.update(population_size=30, max_generations=10, fitness_threshold=None,
               compatibility_threshold=1.5, node_add_prob=0.15, conn_add_prob=0.25)
    check_config(cfg)
    cfg.update(population_size=12, compatibility_threshold=1.0, weight_mutate_prob=0.0,
               bias_mutate_prob=0.0, activation_mutate_prob=0.0, node_add_prob=0.0,
               conn_add_prob=0.0, mate_only_prob=1.0, survival_threshold=1.0,
               max_stagnation=3, min_species_to_keep=1)
    check_config(cfg)

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"check_config: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()