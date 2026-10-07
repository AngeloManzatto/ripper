"""
Created on Sat Jun  6 18:50:21 2026

@author: Angelo Antonio Manzatto
--------------
Central configuration for the NEAT algorithm.

All hyperparameters live here — nothing is hardcoded elsewhere.
NEATConfig is a frozen dataclass: immutable after creation, safe to
share across threads and experiments.

To run a modified experiment without touching the original:

    from dataclasses import replace
    cfg = NEATConfig()
    cfg_large = replace(cfg, population_size=500, max_generations=1000)

Design notes
------------
- Every field has a type annotation and a default that reproduces the
  classic NEAT paper behaviour on XOR.
- Fields are grouped by concern: population, fitness, weights, biases,
  mutation, speciation, reproduction, stagnation.
- No field references another field — defaults are plain literals only.
  Cross-field validation lives in __post_init__.
- seed=None means non-deterministic. Pass an int for reproducibility.
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NEATConfig:

    # ── Experiment identity ───────────────────────────────────────────────────

    experiment_name: str   = "neat_experiment"
    seed:            Optional[int] = None   # None = non-deterministic

    # ── Population ────────────────────────────────────────────────────────────

    population_size:  int   = 150
    max_generations:  int   = 300

    # Number of hidden nodes in the minimal initial genome.
    # 0 = start with direct input→output connections only (classic NEAT).
    n_initial_hidden: int   = 0

    # Probability that a new connection is created in the initial genome.
    # Only used when n_initial_hidden > 0.
    initial_conn_prob: float = 0.5

    # ── Fitness ───────────────────────────────────────────────────────────────

    # Evolution stops early when any genome reaches this fitness.
    # Set to float('inf') to always run max_generations.
    fitness_threshold: float = 3.9

    # Whether higher fitness is better (True) or lower is better (False).
    # Our transform detection uses reconstruction error → lower is better.
    fitness_maximize: bool = True

    # ── Connection weights ────────────────────────────────────────────────────

    weight_init_mean:    float = 0.0
    weight_init_stdev:   float = 1.0
    weight_min_value:    float = -30.0
    weight_max_value:    float =  30.0

    # Probability that a weight mutation is applied to each connection.
    weight_mutate_prob:  float = 0.80

    # Given mutation is applied:
    #   with probability weight_perturb_prob  → perturb (add Gaussian noise)
    #   with probability 1-weight_perturb_prob → replace with random value
    weight_perturb_prob: float = 0.90
    weight_perturb_power: float = 0.50   # std of perturbation noise

    # ── Node biases ───────────────────────────────────────────────────────────

    bias_init_mean:    float = 0.0
    bias_init_stdev:   float = 1.0
    bias_min_value:    float = -30.0
    bias_max_value:    float =  30.0

    bias_mutate_prob:  float = 0.70
    bias_perturb_prob: float = 0.90
    bias_perturb_power: float = 0.50

    # ── Structural mutation ───────────────────────────────────────────────────

    # Probability of adding a new connection gene per genome per generation.
    conn_add_prob:    float = 0.03

    # Probability of deleting an existing connection gene.
    conn_delete_prob: float = 0.00

    # Probability of adding a new node (splitting an existing connection).
    node_add_prob:    float = 0.05

    # Probability of deleting an existing node.
    node_delete_prob: float = 0.00

    # ── Activation functions ──────────────────────────────────────────────────

    # Names of activation functions available to nodes.
    # Must all be registered in neat/activations.py.
    # Probability of mutating a node's activation function.
    activation_mutate_prob: float = 0.10

    activation_options: tuple = (
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
    )

    # Default activation for output nodes.
    # "identity" is suitable for continuous/regression outputs.
    # "sigmoid" is suitable for binary classification.
    output_activation: str = "sigmoid"

    # Default activation for hidden nodes at creation.
    hidden_activation: str = "sigmoid"

    # ── Speciation ────────────────────────────────────────────────────────────

    # Compatibility distance threshold. Genomes with distance <= threshold
    # are placed in the same species.
    compatibility_threshold: float = 3.0

    # Coefficients for the compatibility distance formula:
    #   d = c1*E/N + c2*D/N + c3*W
    # where E=excess genes, D=disjoint genes, W=avg weight diff of matching genes.
    # N = number of genes in the larger genome (normalisation factor).
    compatibility_c1: float = 1.0   # excess gene coefficient
    compatibility_c2: float = 1.0   # disjoint gene coefficient
    compatibility_c3: float = 0.4   # weight difference coefficient

    # Minimum gene count below which N is clamped to 1 (avoids division issues
    # for very small genomes at the start of evolution).
    compatibility_n_min: int = 20

    # ── Reproduction ─────────────────────────────────────────────────────────

    # Fraction of each species allowed to reproduce.
    survival_threshold: float = 0.20

    # Probability that reproduction is crossover-only (no mutation).
    mate_only_prob: float = 0.20

    # Minimum species size to apply elitism (copy best genome unchanged).
    elitism_min_species_size: int = 3

    # Number of elites to carry forward per species (when size >= above).
    elitism_count: int = 1

    # Minimum number of species to keep alive even if stagnated.
    min_species_to_keep: int = 2

    # ── Stagnation ────────────────────────────────────────────────────────────

    # Generations without improvement before a species is considered stagnant.
    max_stagnation: int = 15

    # ── Reporting ────────────────────────────────────────────────────────────

    # Print a summary every N generations. 0 = silent.
    report_every: int = 1

    # ── Validation ───────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # population
        _check_positive_int(self.population_size,  "population_size")
        _check_positive_int(self.max_generations,  "max_generations")
        _check_non_negative_int(self.n_initial_hidden, "n_initial_hidden")

        # fitness
        _check_finite(self.fitness_threshold, "fitness_threshold")

        # weights
        _check_range(self.weight_init_mean, self.weight_min_value,
                     self.weight_max_value, "weight_init_mean")
        _check_prob(self.weight_mutate_prob,   "weight_mutate_prob")
        _check_prob(self.weight_perturb_prob,  "weight_perturb_prob")
        _check_positive(self.weight_perturb_power, "weight_perturb_power")

        # biases
        _check_range(self.bias_init_mean, self.bias_min_value,
                     self.bias_max_value, "bias_init_mean")
        _check_prob(self.bias_mutate_prob,   "bias_mutate_prob")
        _check_prob(self.bias_perturb_prob,  "bias_perturb_prob")
        _check_positive(self.bias_perturb_power, "bias_perturb_power")

        # structural mutation
        _check_prob(self.conn_add_prob,    "conn_add_prob")
        _check_prob(self.conn_delete_prob, "conn_delete_prob")
        _check_prob(self.node_add_prob,    "node_add_prob")
        _check_prob(self.node_delete_prob, "node_delete_prob")

        # activations
        _check_prob(self.activation_mutate_prob, "activation_mutate_prob")
        if not self.activation_options:
            raise ValueError("activation_options must not be empty")
        if self.output_activation not in self.activation_options:
            raise ValueError(
                f"output_activation '{self.output_activation}' "
                f"not in activation_options"
            )
        if self.hidden_activation not in self.activation_options:
            raise ValueError(
                f"hidden_activation '{self.hidden_activation}' "
                f"not in activation_options"
            )

        # speciation
        _check_non_negative(self.compatibility_threshold, "compatibility_threshold")
        _check_positive(self.compatibility_c1, "compatibility_c1")
        _check_positive(self.compatibility_c2, "compatibility_c2")
        _check_positive(self.compatibility_c3, "compatibility_c3")
        _check_positive_int(self.compatibility_n_min, "compatibility_n_min")

        # reproduction
        _check_prob(self.survival_threshold, "survival_threshold")
        _check_prob(self.mate_only_prob,     "mate_only_prob")
        _check_positive_int(self.elitism_min_species_size, "elitism_min_species_size")
        _check_positive_int(self.elitism_count,            "elitism_count")
        _check_positive_int(self.min_species_to_keep,      "min_species_to_keep")

        # stagnation
        _check_positive_int(self.max_stagnation, "max_stagnation")

        # reporting
        _check_non_negative_int(self.report_every, "report_every")


###############################################################################
# Validation helpers (module-private)
###############################################################################

def _check_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _check_non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def _check_non_negative(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _check_positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _check_finite(value: float, name: str) -> None:
    import math
    if not isinstance(value, (int, float)) or math.isnan(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _check_prob(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")


def _check_range(value: float, lo: float, hi: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not (lo <= value <= hi):
        raise ValueError(
            f"{name} must be in [{lo}, {hi}], got {value!r}"
        )