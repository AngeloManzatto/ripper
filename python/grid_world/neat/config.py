
"""
Created on Sat Oct 17 19:26:14 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

def default_config():
    """
    Return a new dictionary with every NEAT parameter at its default value.

    Returns
    -------
    dict
        Parameter name -> value, grouped below by concern.
    """
    return {

        # ── Experiment identity ───────────────────────────────────────────
        "experiment_name": "neat_experiment",
        "seed": None,                         # None = non-deterministic

        # ── Population ────────────────────────────────────────────────────
        "population_size": 150,
        "max_generations": 300,
        "n_initial_hidden": 0,                # 0 = start with input -> output only
        "initial_conn_prob": 0.5,             # only used when n_initial_hidden > 0

        # ── Fitness ───────────────────────────────────────────────────────
        "fitness_threshold": 3.9,             # stop early at this fitness (XOR value)
        "fitness_maximize": True,             # True = higher is better

        # ── Connection weights ────────────────────────────────────────────
        "weight_init_mean": 0.0,
        "weight_init_stdev": 1.0,
        "weight_min_value": -30.0,
        "weight_max_value": 30.0,
        "weight_mutate_prob": 0.80,           # chance each weight is mutated
        "weight_perturb_prob": 0.90,          # if mutated: nudge (else replace)
        "weight_perturb_power": 0.50,         # std of the nudge

        # ── Node biases ───────────────────────────────────────────────────
        "bias_init_mean": 0.0,
        "bias_init_stdev": 1.0,
        "bias_min_value": -30.0,
        "bias_max_value": 30.0,
        "bias_mutate_prob": 0.70,
        "bias_perturb_prob": 0.90,
        "bias_perturb_power": 0.50,

        # ── Structural mutation (chance per genome per generation) ────────
        "conn_add_prob": 0.03,
        "conn_delete_prob": 0.00,
        "node_add_prob": 0.05,
        "node_delete_prob": 0.00,

        # ── Activation functions ──────────────────────────────────────────
        "activation_mutate_prob": 0.10,
        "activation_options": (
            "sigmoid", 
            "tanh", 
            "relu", 
            "leaky_relu", 
            "elu", 
            "swish",
            "softplus", 
            "gaussian", 
            "identity", 
            "sinusoid", 
            "cosine",
            "inverse", 
            "step", 
            "arctan", 
            "sign",
        ),
        "output_activation": "tanh",
        "hidden_activation": "sigmoid",

        # ── Speciation ────────────────────────────────────────────────────
        # distance = c1 * excess / N + c2 * disjoint / N + c3 * weight_diff
        "compatibility_threshold": 3.0,       # same species if distance <= this
        "compatibility_c1": 1.0,              # excess genes
        "compatibility_c2": 1.0,              # disjoint genes
        "compatibility_c3": 0.4,              # average weight difference
        "compatibility_n_min": 20,            # genomes smaller than this count as N = 1

        # ── Reproduction ──────────────────────────────────────────────────
        "survival_threshold": 0.20,           # fraction of a species that may breed
        "mate_only_prob": 0.20,               # crossover without mutation
        "elitism_min_species_size": 3,        # smallest species that keeps an elite
        "elitism_count": 1,                   # elites kept per species
        "min_species_to_keep": 2,             # survive stagnation removal

        # ── Stagnation ────────────────────────────────────────────────────
        "max_stagnation": 15,                 # generations without improvement

        # ── Reporting ─────────────────────────────────────────────────────
        "report_every": 1,                    # 0 = silent
    }