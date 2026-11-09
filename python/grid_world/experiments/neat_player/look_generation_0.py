"""
Created on Fri Oct 23 14:30:00 2026

@author: Angelo Antonio Manzatto

Look at generation 0 of the Player before any evolution: how many random genomes
reach the goal at all, and how the others fail.

Run this file (F5 in Spyder). It takes about POPULATION x 0.2 s. It evaluates one
random population on the training layouts and prints the distribution of goal
rates and the outcome counts. Nothing here is a test.
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

from agents.neat_agent.agent import N_INPUTS
from core.actions import N_ACTIONS
from experiments.neat_player.fitness import genome_summary
from neat.config import default_config
from neat.evolution import init_population

###############################################################################
# Settings (change here)
###############################################################################

POPULATION = 150
N_EPISODES = 50
SEED = 200_000           # training seed: must stay away from the benchmark seeds
GENOME_SEED = 0          # seed of the random genomes

###############################################################################
# Run
###############################################################################

def main():
    cfg = default_config()
    cfg["population_size"] = POPULATION
    rng = np.random.default_rng(GENOME_SEED)
    genomes = init_population(N_INPUTS, N_ACTIONS, rng, cfg)["genomes"]

    summaries = [genome_summary(g, N_EPISODES, SEED) for g in genomes]
    rates = [s.rate("GoalReached") for s in summaries]

    # goal rates
    print(f"{POPULATION} random genomes, {N_EPISODES} games each\n")
    print(f"genomes that reached the goal at least once: {sum(r > 0 for r in rates)} of {POPULATION}")
    print(f"best goal rate {max(rates):.3f}   mean {np.mean(rates):.3f}\n")
    print("goal rate -> number of genomes")
    for value in sorted(set(rates)):
        print(f"  {value:.3f} -> {rates.count(value)}")

    # outcomes over all games of all genomes
    total = {}
    for s in summaries:
        for name, count in s.outcome_counts.items():
            total[name] = total.get(name, 0) + count
    games = sum(total.values())
    print(f"\noutcomes over {games} games")
    for name, count in total.items():
        print(f"  {name:12s} {count:6d}  {100 * count / games:5.1f}%")
    print(f"mean steps per game: {np.mean([s.mean_steps for s in summaries]):.1f}")

    plt.figure(figsize=(7, 3.5))
    plt.hist(rates, bins=np.linspace(0, max(max(rates), 0.05) + 0.025, 21), edgecolor="white")
    plt.xlabel("goal rate of a random genome")
    plt.ylabel("genomes")
    plt.title("Generation 0: goal rate of random genomes")
    plt.show()


if __name__ == "__main__":
    main()