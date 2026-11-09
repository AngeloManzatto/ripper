
"""
Created on Fri Oct 23 11:20:00 2026

@author: Angelo Antonio Manzatto

How long does one genome take to evaluate, and where does the time go?

Run this file (F5 in Spyder). It builds a population of random starting genomes,
times genome_fitness on a few of them, and then profiles one evaluation so you can
see which functions eat the time. Nothing here is a test: it only prints numbers.
"""

###############################################################################
# Libraries
###############################################################################

import cProfile
import pstats
import time

import numpy as np

from agents.neat_agent.agent import N_INPUTS
from core.actions import N_ACTIONS
from experiments.neat_player.fitness import genome_fitness
from neat.config import default_config
from neat.evolution import init_population

###############################################################################
# Settings (change here)
###############################################################################

N_EPISODES = 20          # layouts per genome
SEED = 200_000           # training seed: must stay away from the benchmark seeds
N_TIMED = 5              # how many genomes to time
POPULATION = 150         # for the estimate at the end
GENERATIONS = 100        # for the estimate at the end

###############################################################################
# Run
###############################################################################

def main():
    cfg = default_config()
    cfg["population_size"] = N_TIMED
    rng = np.random.default_rng(0)
    state = init_population(N_INPUTS, N_ACTIONS, rng, cfg)
    genomes = state["genomes"]

    genome_fitness(genomes[0], 2, SEED)          # warm-up, not timed

    seconds = []
    for genome in genomes:
        start = time.perf_counter()
        fitness = genome_fitness(genome, N_EPISODES, SEED)
        seconds.append(time.perf_counter() - start)
        print(f"fitness {fitness:.3f}   {seconds[-1]:.3f} s")

    per_genome = sum(seconds) / len(seconds)
    total_hours = per_genome * POPULATION * GENERATIONS / 3600
    print()
    print(f"{N_EPISODES} episodes per genome: {per_genome:.3f} s on average")
    print(f"{POPULATION} genomes x {GENERATIONS} generations, one process: about {total_hours:.1f} hours")

    print("\nWhere the time goes (one genome, top 12 by own time):\n")
    profiler = cProfile.Profile()
    profiler.enable()
    genome_fitness(genomes[0], N_EPISODES, SEED)
    profiler.disable()
    pstats.Stats(profiler).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":
    main()