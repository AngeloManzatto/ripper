"""
Created on Sat Oct 17 19:34:09 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import json

from neat.config import default_config

###############################################################################
# Content
###############################################################################

def test_returns_a_dict():
    assert isinstance(default_config(), dict)

def test_known_default_values():
    cfg = default_config()
    assert cfg["weight_init_mean"] == 0.0
    assert cfg["weight_init_stdev"] == 1.0
    assert cfg["weight_min_value"] == -30.0
    assert cfg["weight_max_value"] == 30.0
    assert cfg["population_size"] == 150
    assert cfg["compatibility_threshold"] == 3.0
    assert cfg["seed"] is None
    assert cfg["fitness_maximize"] is True


def test_activation_options_hold_the_fifteen_names():
    options = default_config()["activation_options"]
    assert len(options) == 15
    assert len(set(options)) == 15


def test_default_activations_are_in_the_options():
    cfg = default_config()
    assert cfg["output_activation"] in cfg["activation_options"]
    assert cfg["hidden_activation"] in cfg["activation_options"]


def test_probabilities_are_between_zero_and_one():
    cfg = default_config()
    for name, value in cfg.items():
        if name.endswith("_prob") or name == "survival_threshold":
            assert 0.0 <= value <= 1.0, name

###############################################################################
# Independence
###############################################################################

def test_each_call_returns_a_new_dict():
    a = default_config()
    b = default_config()
    assert a == b
    assert a is not b


def test_changing_one_config_does_not_change_another():
    a = default_config()
    a["weight_init_stdev"] = 99.0
    b = default_config()
    assert b["weight_init_stdev"] == 1.0

###############################################################################
# Plain data
###############################################################################

def test_config_can_be_saved_as_json():
    text = json.dumps(default_config())
    back = json.loads(text)
    assert back["population_size"] == 150
    assert back["seed"] is None
    assert back["activation_options"][0] == "sigmoid"      # tuple comes back as list

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"config: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()