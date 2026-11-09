"""
Created on Mon Oct 19 14:08:28 2026

@author: Angelo Antonio Manzatto
"""

"""
Tests for neat/activations.py: the activation functions, ACTIVATIONS, get_activation.

ACTIVATIONS maps a name to a plain function float -> float. It has every name
that can appear in cfg["activation_options"], "output_activation" and
"hidden_activation".

Every function is safe for any finite input (no OverflowError, no NaN).
The exact definitions are pinned down below:
    sigmoid     1 / (1 + exp(-4.9 x))      (original NEAT steepness)
    inverse     -x                         (negation, not 1/x)
    leaky_relu  x, or 0.01 x when x < 0
    elu         x, or exp(x) - 1 when x <= 0
    swish       x * logistic(x) with steepness 1
    gaussian    exp(-x^2), the input is clamped to [-10, 10]
    step        1 if x > 0 else 0
    sign        -1, 0 or 1

get_activation(name) returns ACTIVATIONS[name]; an unknown name raises
ValueError (the message lists the valid names).

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import math
import pickle

from neat.config import default_config
from neat.activations import ACTIVATIONS, get_activation

###############################################################################
# helpers
###############################################################################

def logistic(x, steepness=1.0):
    return 1.0 / (1.0 + math.exp(-steepness * x))

###############################################################################
# the registry
###############################################################################

def test_every_configured_activation_exists():
    cfg = default_config()
    for name in cfg["activation_options"]:
        assert name in ACTIVATIONS, name
    assert cfg["output_activation"] in ACTIVATIONS
    assert cfg["hidden_activation"] in ACTIVATIONS


def test_every_activation_is_configured():
    assert set(default_config()["activation_options"]) == set(ACTIVATIONS)


def test_get_activation_returns_the_function():
    for name, fn in ACTIVATIONS.items():
        assert get_activation(name) is fn


def test_get_activation_unknown_name_raises_and_lists_the_valid_ones():
    for bad in ("banana", "", "Sigmoid", None):
        try:
            get_activation(bad)
        except ValueError as error:
            assert "sigmoid" in str(error)
            continue
        raise AssertionError(f"{bad!r} must raise ValueError")


def test_activations_are_picklable():
    for name, fn in ACTIVATIONS.items():
        clone = pickle.loads(pickle.dumps(fn))
        assert clone(0.7) == fn(0.7), name

###############################################################################
# values
###############################################################################

def test_activation_values():
    table = {
        "sigmoid": [(0.0, 0.5), (1.0, logistic(1.0, 4.9)), (-2.0, logistic(-2.0, 4.9))],
        "tanh": [(0.0, 0.0), (1.0, math.tanh(1.0))],
        "relu": [(-3.0, 0.0), (0.0, 0.0), (2.5, 2.5)],
        "leaky_relu": [(-3.0, -0.03), (0.0, 0.0), (2.5, 2.5)],
        "elu": [(2.0, 2.0), (0.0, 0.0), (-1.0, math.exp(-1.0) - 1.0)],
        "swish": [(0.0, 0.0), (2.0, 2.0 * logistic(2.0)), (-1.0, -logistic(-1.0))],
        "softplus": [(0.0, math.log(2.0)), (1.0, math.log(1.0 + math.e)), (-2.0, math.log(1.0 + math.exp(-2.0)))],
        "gaussian": [(0.0, 1.0), (1.0, math.exp(-1.0)), (-2.0, math.exp(-4.0))],
        "identity": [(-7.5, -7.5), (0.0, 0.0), (3.0, 3.0)],
        "sinusoid": [(0.0, 0.0), (1.0, math.sin(1.0))],
        "cosine": [(0.0, 1.0), (1.0, math.cos(1.0))],
        "inverse": [(2.0, -2.0), (-4.0, 4.0), (0.0, 0.0)],
        "step": [(-1.0, 0.0), (0.0, 0.0), (0.001, 1.0)],
        "arctan": [(0.0, 0.0), (1.0, math.atan(1.0))],
        "sign": [(-5.0, -1.0), (0.0, 0.0), (5.0, 1.0)],
    }
    assert set(table) == set(ACTIVATIONS)
    for name, cases in table.items():
        for x, expected in cases:
            got = ACTIVATIONS[name](x)
            assert abs(got - expected) < 1e-9, (name, x, got, expected)


def test_sigmoid_is_steeper_than_the_plain_logistic():
    assert ACTIVATIONS["sigmoid"](0.5) > logistic(0.5)
    assert ACTIVATIONS["sigmoid"](-0.5) < logistic(-0.5)

###############################################################################
# huge inputs and shapes
###############################################################################

def test_activations_survive_huge_inputs():
    unbounded = ("identity", "relu", "leaky_relu", "elu", "softplus", "swish", "inverse")
    for name, fn in ACTIVATIONS.items():
        for x in (-1e300, -1e6, -1e3, -100.0, 100.0, 1e3, 1e6, 1e300):
            y = fn(x)
            assert not math.isnan(y), (name, x)
            if name not in unbounded:
                assert math.isfinite(y), (name, x)


def test_bounded_activations_stay_in_range():
    xs = [-1e6, -50.0, -3.0, -0.5, 0.0, 0.5, 3.0, 50.0, 1e6]
    ranges = {"sigmoid": (0.0, 1.0), "tanh": (-1.0, 1.0), "gaussian": (0.0, 1.0),
              "sinusoid": (-1.0, 1.0), "cosine": (-1.0, 1.0), "step": (0.0, 1.0),
              "sign": (-1.0, 1.0), "arctan": (-math.pi / 2, math.pi / 2)}
    for name, (low, high) in ranges.items():
        for x in xs:
            assert low <= ACTIVATIONS[name](x) <= high, (name, x)


def test_limits():
    assert ACTIVATIONS["sigmoid"](1e6) == 1.0
    assert ACTIVATIONS["sigmoid"](-1e6) < 1e-100
    assert ACTIVATIONS["sigmoid"](-1e6) > 0.0
    assert ACTIVATIONS["softplus"](1e3) == 1e3
    assert abs(ACTIVATIONS["softplus"](-1e3)) < 1e-12
    assert ACTIVATIONS["elu"](-1e3) == -1.0
    assert ACTIVATIONS["swish"](1e3) == 1e3
    assert ACTIVATIONS["gaussian"](1e3) == ACTIVATIONS["gaussian"](10.0)
    assert ACTIVATIONS["gaussian"](-1e3) == ACTIVATIONS["gaussian"](10.0)


def test_symmetry_and_monotonic():
    for x in (0.25, 0.5, 2.0, 5.0):
        assert ACTIVATIONS["gaussian"](x) == ACTIVATIONS["gaussian"](-x)
        assert ACTIVATIONS["cosine"](x) == ACTIVATIONS["cosine"](-x)
        assert abs(ACTIVATIONS["sinusoid"](x) + ACTIVATIONS["sinusoid"](-x)) < 1e-12
        assert abs(ACTIVATIONS["tanh"](x) + ACTIVATIONS["tanh"](-x)) < 1e-12
        assert abs(ACTIVATIONS["sigmoid"](x) + ACTIVATIONS["sigmoid"](-x) - 1.0) < 1e-12
        assert ACTIVATIONS["inverse"](x) == -ACTIVATIONS["inverse"](-x)
    xs = [-5.0, -1.0, -0.2, 0.0, 0.3, 1.0, 4.0]
    for name in ("sigmoid", "tanh", "relu", "leaky_relu", "elu", "softplus", "identity", "arctan"):
        values = [ACTIVATIONS[name](x) for x in xs]
        for before, after in zip(values, values[1:]):
            assert after >= before, name

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"activations: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()