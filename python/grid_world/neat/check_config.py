"""
Created on Tue Oct 20 08:05:19 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import math
from numbers import Integral, Real

from neat.activations import ACTIVATIONS
from neat.config import default_config

###############################################################################
# one small check per kind of value; each returns an error text or None
###############################################################################

def is_int(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def is_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def check_int(minimum):
    def check(value):
        if not is_int(value):
            return "must be an integer"
        if value < minimum:
            return f"must be at least {minimum}"
        return None
    return check


def check_real(minimum=None, maximum=None, minimum_open=False):
    def check(value):
        if not is_real(value):
            return "must be a finite number"
        if minimum is not None:
            if value < minimum or (minimum_open and value == minimum):
                return f"must be {'above' if minimum_open else 'at least'} {minimum}"
        if maximum is not None and value > maximum:
            return f"must be at most {maximum}"
        return None
    return check


def check_probability(value):
    return check_real(0.0, 1.0)(value)


def check_bool(value):
    if not isinstance(value, bool):
        return "must be True or False"
    return None


def check_text(value):
    if not isinstance(value, str):
        return "must be a string"
    return None


def check_seed(value):
    if value is not None and (not is_int(value) or value < 0):
        return "must be None or a non-negative integer"
    return None


def check_threshold(value):
    if value is not None and not is_real(value):
        return "must be None or a finite number"
    return None


def check_activation_name(value):
    if not isinstance(value, str) or value not in ACTIVATIONS:
        return f"unknown activation; available: {sorted(ACTIVATIONS)}"
    return None


def check_activation_list(value):
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return "must be a non-empty list or tuple of activation names"
    for name in value:
        if not isinstance(name, str) or name not in ACTIVATIONS:
            return f"contains unknown activation {name!r}; available: {sorted(ACTIVATIONS)}"
    if len(set(value)) != len(value):
        return "contains duplicates"
    return None


def check_maximize(value):
    if value is not True:
        return "must be True: the library always maximizes (negate your fitness to minimize)"
    return None


def check_zero_probability(value):
    if not is_real(value) or value != 0.0:
        return "must be 0.0: this mutation is not implemented"
    return None


def check_zero_int(value):
    if not is_int(value) or value != 0:
        return "must be 0: genomes always start as inputs -> outputs"
    return None

###############################################################################
# rules: key -> check
###############################################################################

RULES = {
    "experiment_name": check_text,
    "seed": check_seed,
    "population_size": check_int(2),
    "max_generations": check_int(1),
    "n_initial_hidden": check_zero_int,
    "initial_conn_prob": check_probability,
    "fitness_threshold": check_threshold,
    "fitness_maximize": check_maximize,
    "weight_init_mean": check_real(),
    "weight_init_stdev": check_real(0.0),
    "weight_min_value": check_real(),
    "weight_max_value": check_real(),
    "weight_mutate_prob": check_probability,
    "weight_perturb_prob": check_probability,
    "weight_perturb_power": check_real(0.0),
    "bias_init_mean": check_real(),
    "bias_init_stdev": check_real(0.0),
    "bias_min_value": check_real(),
    "bias_max_value": check_real(),
    "bias_mutate_prob": check_probability,
    "bias_perturb_prob": check_probability,
    "bias_perturb_power": check_real(0.0),
    "conn_add_prob": check_probability,
    "conn_delete_prob": check_zero_probability,
    "node_add_prob": check_probability,
    "node_delete_prob": check_zero_probability,
    "activation_mutate_prob": check_probability,
    "activation_options": check_activation_list,
    "output_activation": check_activation_name,
    "hidden_activation": check_activation_name,
    "compatibility_threshold": check_real(0.0, None, True),
    "compatibility_c1": check_real(0.0),
    "compatibility_c2": check_real(0.0),
    "compatibility_c3": check_real(0.0),
    "compatibility_n_min": check_int(1),
    "survival_threshold": check_real(0.0, 1.0, True),
    "mate_only_prob": check_probability,
    "elitism_min_species_size": check_int(1),
    "elitism_count": check_int(0),
    "min_species_to_keep": check_int(1),
    "max_stagnation": check_int(1),
    "report_every": check_int(0),
}

###############################################################################
# check_config
###############################################################################

def check_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a dict, got {type(cfg).__name__}")

    known = default_config()
    problems = []

    for key in sorted(known):
        if key not in cfg:
            problems.append(f"{key}: missing")
    for key in sorted(cfg, key=str):
        if key not in known:
            problems.append(f"{key}: unknown key (typo?)")

    for key in sorted(known):
        if key in cfg and key in RULES:
            message = RULES[key](cfg[key])
            if message is not None:
                problems.append(f"{key}: {message} (got {cfg[key]!r})")

    # relations between keys, only when both values passed their own check
    for prefix in ("weight", "bias"):
        low, high = f"{prefix}_min_value", f"{prefix}_max_value"
        if (low in cfg and high in cfg and is_real(cfg[low]) and is_real(cfg[high])
                and cfg[low] >= cfg[high]):
            problems.append(f"{low}: must be below {high} (got {cfg[low]!r} and {cfg[high]!r})")

    if problems:
        raise ValueError("invalid config:\n  " + "\n  ".join(problems))