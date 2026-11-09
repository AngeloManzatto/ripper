
"""
Created on Fri Oct 23 14:45:00 2026

@author: Angelo Antonio Manzatto

First real run of the Player: evolve NEAT genomes on goal rate.

Run this file (F5 in Spyder). With the settings below it plays about
POPULATION x N_EPISODES games per generation (about 30 s per generation at
150 x 20). At the end it prints the diagnostics tables, shows the plots, tests
the champion on layouts it never trained on, and saves the champion genome.

Fitness is the pure goal rate; the opponents are the engine's own.
"""

###############################################################################
# Libraries
###############################################################################

import time

import numpy as np
import matplotlib.pyplot as plt

from agents.neat_agent.agent import N_INPUTS
from core.actions import N_ACTIONS
from experiments.neat_player.fitness import evaluate_population, genome_fitness
from neat.checkpoint import save_genome
from neat.config import default_config
from neat.evolution import init_population
from neat.diagnostics.record import run_recorded
from neat.diagnostics.table import generation_table, species_table
from neat.diagnostics.plots import plot_overview

###############################################################################
# Settings (change here)
###############################################################################

POPULATION = 150
N_EPISODES = 20            # games per genome, same layouts for everybody
MAX_GENERATIONS = 30
SEED = 200_000             # training layouts: must stay away from the benchmark seeds
FRESH_SEED = SEED + 10_000 # layouts the champion never saw (also away from the benchmark)
N_FRESH = 100
RUN_SEED = 0               # seed of the evolution itself
CHAMPION_PATH = "neat_player_champion.json"

###############################################################################
# Run
###############################################################################

def main():
    cfg = default_config()
    cfg["population_size"] = POPULATION
    cfg["max_generations"] = MAX_GENERATIONS
    cfg["fitness_threshold"] = None            # always run all generations in this first look

    rng = np.random.default_rng(RUN_SEED)
    state = init_population(N_INPUTS, N_ACTIONS, rng, cfg)

    generation = [0]
    start = time.perf_counter()

    def evaluate(genomes):
        fitnesses = evaluate_population(genomes, N_EPISODES, SEED)
        print(f"generation {generation[0]:3d}   best {max(fitnesses):.3f}   "
              f"mean {np.mean(fitnesses):.3f}   {time.perf_counter() - start:6.0f} s elapsed")
        generation[0] += 1
        return fitnesses

    state, history = run_recorded(state, evaluate, rng, cfg)

    print()
    print(generation_table(history))
    print()
    print(species_table(history[-1]))

    champion = state["best"]
    hidden = sum(1 for n in champion["genome"]["nodes"].values() if n["type"] == "hidden")
    enabled = sum(1 for c in champion["genome"]["connections"].values() if c["enabled"])
    fresh = genome_fitness(champion["genome"], N_FRESH, FRESH_SEED)
    print()
    print(f"champion from generation {champion['generation']}: {hidden} hidden nodes, {enabled} enabled connections")
    print(f"goal rate on its training layouts: {champion['fitness']:.3f}")
    print(f"goal rate on {N_FRESH} new layouts:        {fresh:.3f}")

    save_genome(CHAMPION_PATH, champion["genome"])
    print(f"champion saved to {CHAMPION_PATH}")

    plot_overview(history)
    plt.show()

    return state, history


if __name__ == "__main__":
    state, history = main()