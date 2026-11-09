"""
Created on Mon Oct 19 15:10:13 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
XOR: the classic end-to-end check of a NEAT library.

2 inputs, 1 output. Fitness = 4 - (sum of squared errors over the 4 cases),
so a perfect network scores 4.0. XOR cannot be solved without a hidden node,
so solving it proves that structure grows, species survive and selection works.

End-to-end check: the whole library (init_population, run_evolution, speciation,
reproduction, compiled networks) must solve XOR.

Fitness = 4 - sum of squared errors, solved at 3.9 (see examples/xor.py).

Two setups:
  - default config: all 15 activation functions may be mutated in. Some of them
    (gaussian, cosine...) can solve XOR with a single node, so this setup is
    easy and only proves that evolution works.
  - sigmoid only (activation_mutate_prob = 0, like the original NEAT): a
    single sigmoid node cannot compute XOR, so a solution NEEDS hidden nodes.
    This proves that structure really grows and is kept by selection.

Measured over 30 seeds (150 genomes, 100 generations at most):
  default config: 30/30 solved, median about 8 generations
  sigmoid only:   28/30 solved, 17 to 93 generations, every winner has >= 2 hidden nodes

The seeds below are fixed, so the runs are exactly reproducible.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import sys

import numpy as np

from neat.config import default_config
from neat.evolution import init_population, run_evolution
from neat.network import compile_genome, activate

from neat.diagnostics.record import run_recorded
from neat.diagnostics.table import generation_table, species_table
from neat.diagnostics.plots import plot_overview
from neat.diagnostics.genome_view import draw_genome, genome_table
from neat.diagnostics.distances import plot_distances, distance_table

import matplotlib.pyplot as plt


###############################################################################
# problem
###############################################################################

CASES = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]


def xor_fitness(genome):
    network = compile_genome(genome)
    error = 0.0
    for inputs, target in CASES:
        output = activate(network, inputs)[0]
        error += (output - target) ** 2
    return 4.0 - error


def evaluate(genomes):
    return [xor_fitness(genome) for genome in genomes]


def xor_config():
    cfg = default_config()
    cfg["population_size"] = 150
    cfg["max_generations"] = 100
    cfg["fitness_threshold"] = 3.9
    cfg["compatibility_threshold"] = 3.0
    cfg["output_activation"] = "sigmoid"     # targets are 0 and 1, so a sigmoid output fits
    cfg["hidden_activation"] = "sigmoid"
    return cfg



###############################################################################
# run
###############################################################################

def solve_xor(seed, cfg=None):
    """Returns (state, history). history[-1]["generation"] + 1 = generations used."""
    if cfg is None:
        cfg = xor_config()
    rng = np.random.default_rng(seed)
    state = init_population(2, 1, rng, cfg)
    return run_evolution(state, evaluate, rng, cfg)

def diagnose(seed, cfg=None, plots=True):
    """Same run as solve_xor, but shows what happens inside the population."""
    if cfg is None:
        cfg = xor_config()
    rng = np.random.default_rng(seed)
    state = init_population(2, 1, rng, cfg)
    state, history = run_recorded(state, evaluate, rng, cfg, keep_genomes=True)
    print(generation_table(history))
    print()
    print(species_table(history[0]))
    print()
    print(species_table(history[-1]))
    print()
    print(genome_table(state["best"]["genome"]))
    r = history[-1]
    print(distance_table(r["genomes"], cfg, r["species"]))
    if plots:
        plot_overview(history)
        draw_genome(state["best"]["genome"], title=f"Champion, fitness {state['best']['fitness']:.4f}")
        plot_distances(history[0]["genomes"], cfg)
        plot_distances(history[-1]["genomes"], cfg)
        plt.show()
        
        

    return state, history

###############################################################################
# helpers
###############################################################################

def n_hidden(genome):
    return sum(1 for node in genome["nodes"].values() if node["type"] == "hidden")


def sigmoid_only_config():
    cfg = xor_config()
    cfg["activation_mutate_prob"] = 0.0
    return cfg


###############################################################################
# default config
###############################################################################

def test_xor_is_solved_with_the_default_config():
    for seed in (0, 1, 2, 3, 4):
        state, history = solve_xor(seed)
        assert state["best"]["fitness"] >= 3.9, seed
        assert len(history) < 100, seed


def test_the_winner_classifies_all_four_cases():
    state, history = solve_xor(0)
    network = compile_genome(state["best"]["genome"])
    for inputs, target in CASES:
        output = activate(network, inputs)[0]
        assert (output > 0.5) == (target == 1.0), (inputs, output)


def test_reported_fitness_matches_the_winner():
    state, history = solve_xor(1)
    assert abs(xor_fitness(state["best"]["genome"]) - state["best"]["fitness"]) < 1e-12


def test_history_best_ever_never_decreases_and_ends_solved():
    state, history = solve_xor(2)
    values = [s["best_ever_fitness"] for s in history]
    for before, after in zip(values, values[1:]):
        assert after >= before
    assert values[-1] >= 3.9
    assert values[0] < values[-1]

###############################################################################
# sigmoid only: hidden nodes are required
###############################################################################

def test_xor_needs_and_grows_hidden_nodes_with_sigmoid_only():
    for seed in (0, 1):
        state, history = solve_xor(seed, sigmoid_only_config())
        best = state["best"]
        assert best["fitness"] >= 3.9, seed
        assert n_hidden(best["genome"]) >= 1, seed
        assert max(s["n_species"] for s in history) > 1, seed


def test_all_activations_stay_sigmoid_when_activation_mutation_is_off():
    state, history = solve_xor(0, sigmoid_only_config())
    for genome in state["genomes"]:
        for node in genome["nodes"].values():
            if node["type"] != "input":
                assert node["activation"] == "sigmoid"

###############################################################################
# reproducibility
###############################################################################

def test_same_seed_same_run():
    a, history_a = solve_xor(3)
    b, history_b = solve_xor(3)
    assert history_a == history_b
    assert a["best"] == b["best"]


def test_different_seeds_different_runs():
    a, history_a = solve_xor(3)
    b, history_b = solve_xor(4)
    assert history_a != history_b

###############################################################################
# Direct run
###############################################################################

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if "--diag" in sys.argv:
        state, history = diagnose(seed, sigmoid_only_config() if "--sigmoid" in sys.argv else None)
    else:
        state, history = solve_xor(seed)
    best = state["best"]
    print(f"seed {seed}: best fitness {best['fitness']:.4f} after {len(history)} generations")
    hidden = sum(1 for n in best["genome"]["nodes"].values() if n["type"] == "hidden")
    enabled = sum(1 for c in best["genome"]["connections"].values() if c["enabled"])
    print(f"winner: {hidden} hidden nodes, {enabled} enabled connections")
    network = compile_genome(best["genome"])
    for inputs, target in CASES:
        print(inputs, "->", round(activate(network, inputs)[0], 3), "(want", target, ")")


def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"xor: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()
    state, history = diagnose(0, sigmoid_only_config())
    plot_overview(history, "xor_overview.png")