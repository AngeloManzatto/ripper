"""
Created on Fri Oct 23 15:05:00 2026

@author: Angelo Antonio Manzatto

Run of the Player with rotating layouts and honest scoring.

  - Every generation plays NEW layouts (seed = SEED + generation * N_EPISODES), so a
    genome cannot win by memorising a fixed set of games.
  - After each generation, that generation's best genome plays N_VAL fixed layouts it
    never trained on. The printed "val" is the generalization curve.
  - The champion is the genome with the best val score (the best training score of a
    generation is partly luck). It is then tested ONCE at the end on a third set of
    layouts (TEST_SEED) that played no part in any choice.
  - A steadier measure than the champion: the mean test score of the TOP_K best genomes of the
    last generation ("top"), and of a fixed random sample of SAMPLE_K genomes of it ("sample").
    "top" is less ruled by the luck of one genome; "sample" says how good a typical genome is.
  - GENERATION_KWARGS sets the world (here: an easy one, to see whether NEAT learns).

Run this file (F5 in Spyder) for one run, or use run_seeds.py to repeat it with
several seeds. About 26 s per generation at 150 x 20, plus 1 s of validation.
"""

###############################################################################
# Libraries
###############################################################################

import copy
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
N_EPISODES = 20              # games per genome, new layouts every generation
MAX_GENERATIONS = 30
SEED = 200_000               # training layouts start here: must stay away from the benchmark seeds
VAL_SEED = SEED + 10_000     # validation layouts, fixed
N_VAL = 100
TEST_SEED = SEED + 20_000    # test layouts, used once at the end
N_TEST = 200
RUN_SEED = 0                 # seed of the evolution itself (initial genomes and mutations)
TOP_K = 20                   # best genomes of the last generation (by training score) tested on the test layouts
SAMPLE_K = 20                # random genomes of the last generation tested on the test layouts
SAMPLE_SEED = 12345          # which genomes: the same positions in every run
CHAMPION_PATH = "neat_player_champion_{tag}_seed{seed}.json"

# Layout settings, passed to the engine's GenerationConfig for training, validation and test.
# None = the engine defaults.
GENERATION_KWARGS = {
    "wall_density": 0.0,
    "num_enemies": 0,
    "num_traps": 0,
}

###############################################################################
# One run
###############################################################################

def run_once(run_seed=RUN_SEED, verbose=True, overrides=None, tag="easy"):
    """One complete evolution. Returns a dict with everything worth comparing between runs:
    run_seed, state, history, train_best, val_scores (one per generation), champion
    (val, generation, genome), test_champion, test_lucky, top_test, sample_test, seconds.

    overrides : dict of NEAT config values to change for this run (names as in default_config).
    tag       : label used in the name of the saved champion file."""
    if TOP_K > POPULATION or SAMPLE_K > POPULATION:
        raise ValueError("TOP_K and SAMPLE_K cannot be larger than POPULATION")
    if SEED + MAX_GENERATIONS * N_EPISODES > VAL_SEED:
        raise ValueError("training layouts would run into the validation layouts: "
                         "lower MAX_GENERATIONS or N_EPISODES, or move VAL_SEED")

    cfg = default_config()
    cfg["population_size"] = POPULATION
    cfg["max_generations"] = MAX_GENERATIONS
    cfg["fitness_threshold"] = None            # always run all generations
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise ValueError(f"unknown config key {key!r}")
        cfg[key] = value

    if verbose:
        print(f"layouts: {GENERATION_KWARGS if GENERATION_KWARGS else 'engine defaults'}   run seed {run_seed}   "
              f"config changes: {overrides if overrides else 'none'}\n")

    rng = np.random.default_rng(run_seed)
    state = init_population(N_INPUTS, N_ACTIONS, rng, cfg)

    generation = [0]
    last = {}                                  # genomes and fitnesses of the last evaluated generation
    train_best, val_scores = [], []
    champion = {"val": -1.0, "genome": None, "generation": None}
    start = time.perf_counter()

    def evaluate(genomes):
        fitnesses = evaluate_population(genomes, N_EPISODES, SEED + generation[0] * N_EPISODES, GENERATION_KWARGS)

        best = int(np.argmax(fitnesses))
        val = genome_fitness(genomes[best], N_VAL, VAL_SEED, GENERATION_KWARGS)
        if val > champion["val"]:
            champion.update(val=val, genome=copy.deepcopy(genomes[best]), generation=generation[0])

        last["genomes"], last["fitnesses"] = genomes, fitnesses
        train_best.append(fitnesses[best])
        val_scores.append(val)
        if verbose:
            print(f"seed {run_seed}  generation {generation[0]:3d}   best {fitnesses[best]:.3f}   "
                  f"mean {np.mean(fitnesses):.3f}   val {val:.3f}   {time.perf_counter() - start:6.0f} s elapsed")
        generation[0] += 1
        return fitnesses

    state, history = run_recorded(state, evaluate, rng, cfg)

    lucky = state["best"]
    test_champion = genome_fitness(champion["genome"], N_TEST, TEST_SEED, GENERATION_KWARGS)
    test_lucky = genome_fitness(lucky["genome"], N_TEST, TEST_SEED, GENERATION_KWARGS)
    save_genome(CHAMPION_PATH.format(tag=tag, seed=run_seed), champion["genome"])

    # steadier measures: the last evaluated generation, tested on the test layouts
    order = np.argsort(-np.asarray(last["fitnesses"]), kind="stable")
    top = [last["genomes"][i] for i in order[:TOP_K]]
    positions = np.random.default_rng(SAMPLE_SEED).choice(len(last["genomes"]), size=SAMPLE_K, replace=False)
    sample = [last["genomes"][i] for i in positions]
    top_test = float(np.mean([genome_fitness(g, N_TEST, TEST_SEED, GENERATION_KWARGS) for g in top]))
    sample_test = float(np.mean([genome_fitness(g, N_TEST, TEST_SEED, GENERATION_KWARGS) for g in sample]))

    return {
        "run_seed": run_seed, "tag": tag, "overrides": overrides, "state": state, "history": history,
        "train_best": train_best, "val_scores": val_scores,
        "champion": champion, "lucky": lucky,
        "test_champion": test_champion, "test_lucky": test_lucky,
        "top_test": top_test, "sample_test": sample_test,
        "seconds": time.perf_counter() - start,
    }

###############################################################################
# Run
###############################################################################

def main():
    result = run_once()
    history = result["history"]

    print()
    print(generation_table(history))
    print()
    print(species_table(history[-1]))

    champion, lucky = result["champion"], result["lucky"]
    print()
    print(f"champion by validation: generation {champion['generation']}, val {champion['val']:.3f}, "
          f"test on {N_TEST} unseen layouts {result['test_champion']:.3f}")
    print(f"best training score ever: generation {lucky['generation']}, train {lucky['fitness']:.3f}, "
          f"test {result['test_lucky']:.3f}")
    print(f"last generation on {N_TEST} unseen layouts: best {TOP_K} genomes {result['top_test']:.3f}, "
          f"random {SAMPLE_K} genomes {result['sample_test']:.3f}")
    print(f"champion saved to {CHAMPION_PATH.format(tag=result['tag'], seed=result['run_seed'])}")

    plot_overview(history)
    plt.figure(figsize=(7, 3.5))
    plt.plot(result["train_best"], label="best training score of the generation")
    plt.plot(result["val_scores"], label="same genome on validation layouts")
    plt.xlabel("generation")
    plt.ylabel("goal rate")
    plt.legend(frameon=False)
    plt.title("Does the best genome of each generation generalize?")
    plt.show()

    return result


if __name__ == "__main__":
    result = main()