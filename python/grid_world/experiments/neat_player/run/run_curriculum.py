"""
Created on Sat Oct 24 06:40:00 2026

@author: Angelo Antonio Manzatto

Curriculum for the Player: train in easy worlds first and carry the SAME population into harder
ones, and compare with training in the target world from the start for the same total number of
generations (the control).

  curriculum : free arena -> walls 0.1 -> walls 0.2 -> walls 0.2 + 1 enemy   (30 generations each)
  control    : the target world (walls 0.2 + 1 enemy) for all four periods     (30 generations each)

Both start from the same random population (same seed) and see the same training layouts per
generation number, so they are compared seed by seed. After every period the script tests the
last generation on layouts that took no part in any choice (see run_rotating.py):
  top    : mean of the best TOP_K genomes
  sample : mean of a fixed random sample of SAMPLE_K genomes
both in the world of that period ("own") and in the target world ("target").

The question: after 120 generations, is the curriculum's "target" score above the control's?

Run this file (F5 in Spyder). About 1 hour per sequence, so about 2 hours per seed; seeds run one
after the other and a comparison is printed after each one, so you can stop after the first.
"""

###############################################################################
# Libraries
###############################################################################

import copy
import time

import numpy as np

from agents.neat_agent.agent import N_INPUTS
from core.actions import N_ACTIONS
from experiments.neat_player.fitness import evaluate_population, genome_fitness
from neat.checkpoint import save_checkpoint, save_genome
from neat.config import default_config
from neat.evolution import init_population
from neat.diagnostics.record import run_recorded
import experiments.neat_player.run_rotate as rr

###############################################################################
# Settings (change here)
###############################################################################

RUN_SEEDS = [0, 1, 2]

TARGET = {"wall_density": 0.2, "num_enemies": 1, "num_traps": 0}

# (name, layout settings, generations)
CURRICULUM = [
    ("free arena",          {"wall_density": 0.0, "num_enemies": 0, "num_traps": 0}, 30),
    ("walls 0.1",           {"wall_density": 0.1, "num_enemies": 0, "num_traps": 0}, 30),
    ("walls 0.2",           {"wall_density": 0.2, "num_enemies": 0, "num_traps": 0}, 30),
    ("walls 0.2 + 1 enemy", TARGET, 30),
]
CONTROL = [(f"target, period {k + 1}", TARGET, n) for k, (_, _, n) in enumerate(CURRICULUM)]

# Config changes against default_config. weight_mutate_prob 0.1 is the setting that won the last
# experiment. max_stagnation is raised because there is one species and a new period lowers its
# scores, which could otherwise get that species removed.
OVERRIDES = {"weight_mutate_prob": 0.10, "max_stagnation": 1000}

CHAMPION_PATH = "neat_player_{mode}_period{period}_seed{seed}.json"

###############################################################################
# One sequence of periods
###############################################################################

def steady_tests(last, kwargs):
    """(top, sample): mean test score of the best TOP_K and of a fixed random sample of SAMPLE_K
    genomes of the last evaluated generation, on the test layouts of the world `kwargs`."""
    order = np.argsort(-np.asarray(last["fitnesses"]), kind="stable")
    top = [last["genomes"][i] for i in order[:rr.TOP_K]]
    positions = np.random.default_rng(rr.SAMPLE_SEED).choice(len(last["genomes"]), size=rr.SAMPLE_K, replace=False)
    sample = [last["genomes"][i] for i in positions]
    top_test = float(np.mean([genome_fitness(g, rr.N_TEST, rr.TEST_SEED, kwargs) for g in top]))
    sample_test = float(np.mean([genome_fitness(g, rr.N_TEST, rr.TEST_SEED, kwargs) for g in sample]))
    return top_test, sample_test


def run_sequence(periods, run_seed, mode, overrides=None, checkpoint_template=None, score_population=None):
    """Train through the periods one after the other with the same population.
    Returns one dict per period: name, kwargs, val_scores, champion_val, champion_target,
    top_own, sample_own, top_target, sample_target, minutes.

    overrides           : config changes for this sequence; None uses OVERRIDES above.
    checkpoint_template : e.g. "ckpt_{mode}_period{period}_seed{seed}.pkl". When given, the whole
                          evolution state is saved after every period, so the population can be
                          inspected later (neat.checkpoint.load_checkpoint).
    score_population    : function (genomes, layout_seed, kwargs) -> (fitnesses, goal_rates). The
                          fitnesses go to the evolution; the goal rates are what is reported, what
                          picks the generation's best genome for validation, and what ranks the
                          "top" genomes. None plays the games with evaluate_population and uses the
                          goal rates as the fitnesses (the original behaviour)."""
    total = sum(n for _, _, n in periods)
    if rr.SEED + total * rr.N_EPISODES > rr.VAL_SEED:
        raise ValueError("training layouts would run into the validation layouts: lower the generations "
                         "or move VAL_SEED")

    if score_population is None:
        def score_population(genomes, layout_seed, kwargs):
            goal_rates = evaluate_population(genomes, rr.N_EPISODES, layout_seed, kwargs)
            return goal_rates, goal_rates

    cfg = default_config()
    cfg["population_size"] = rr.POPULATION
    cfg["fitness_threshold"] = None
    for key, value in (OVERRIDES if overrides is None else overrides).items():
        if key not in cfg:
            raise ValueError(f"unknown config key {key!r}")
        cfg[key] = value

    rng = np.random.default_rng(run_seed)
    state = init_population(N_INPUTS, N_ACTIONS, rng, cfg)

    counter = [0]                                    # generations done so far, over all periods
    results = []
    start = time.perf_counter()

    for index, (name, kwargs, n_generations) in enumerate(periods):
        cfg["max_generations"] = counter[0] + n_generations
        last = {}
        val_scores = []
        champion = {"val": -1.0, "genome": None}

        def evaluate(genomes):
            fitnesses, goal_rates = score_population(genomes, rr.SEED + counter[0] * rr.N_EPISODES, kwargs)
            best = int(np.argmax(goal_rates))
            val = genome_fitness(genomes[best], rr.N_VAL, rr.VAL_SEED, kwargs)
            if val > champion["val"]:
                champion.update(val=val, genome=copy.deepcopy(genomes[best]))
            last["genomes"], last["fitnesses"] = genomes, goal_rates
            val_scores.append(val)
            print(f"seed {run_seed} {mode} period {index + 1}/{len(periods)} ({name})  generation {counter[0]:3d}   "
                  f"best {goal_rates[best]:.3f}   mean {np.mean(goal_rates):.3f}   val {val:.3f}   "
                  f"{(time.perf_counter() - start) / 60:5.1f} min")
            counter[0] += 1
            return fitnesses

        state, _ = run_recorded(state, evaluate, rng, cfg)

        top_own, sample_own = steady_tests(last, kwargs)
        if kwargs == TARGET:
            top_target, sample_target = top_own, sample_own
        else:
            top_target, sample_target = steady_tests(last, TARGET)
        champion_target = genome_fitness(champion["genome"], rr.N_TEST, rr.TEST_SEED, TARGET)
        save_genome(CHAMPION_PATH.format(mode=mode, period=index + 1, seed=run_seed), champion["genome"])
        if checkpoint_template is not None:
            save_checkpoint(checkpoint_template.format(mode=mode, period=index + 1, seed=run_seed), state, rng, cfg)

        results.append({
            "name": name, "kwargs": kwargs, "val_scores": val_scores,
            "champion_val": champion["val"], "champion_target": champion_target,
            "top_own": top_own, "sample_own": sample_own,
            "top_target": top_target, "sample_target": sample_target,
            "minutes": (time.perf_counter() - start) / 60,
        })
        print(f"\nseed {run_seed} {mode} period {index + 1} done ({name}):  own top {top_own:.3f} sample {sample_own:.3f}"
              f"   target top {top_target:.3f} sample {sample_target:.3f}   champion on target {champion_target:.3f}\n")

    return results

###############################################################################
# Summary
###############################################################################

def comparison_table(curriculum, control):
    """Per period: both sequences side by side (target-world scores), and the difference.
    curriculum, control: the lists returned by run_sequence for the same seed. Returns text."""
    head = (f"{'period':>6}  {'curriculum world':<22} {'own top':>8} {'own smp':>8} {'TARGET top':>11} {'smp':>6}"
            f"   {'control top':>12} {'smp':>6}   {'d top':>7} {'d smp':>7}")
    lines = [head, "-" * len(head)]
    for k, (c, n) in enumerate(zip(curriculum, control)):
        lines.append(f"{k + 1:>6}  {c['name']:<22} {c['top_own']:>8.3f} {c['sample_own']:>8.3f} "
                     f"{c['top_target']:>11.3f} {c['sample_target']:>6.3f}   {n['top_target']:>12.3f} "
                     f"{n['sample_target']:>6.3f}   {c['top_target'] - n['top_target']:>+7.3f} "
                     f"{c['sample_target'] - n['sample_target']:>+7.3f}")
    return "\n".join(lines)

###############################################################################
# Run
###############################################################################

def main():
    everything = {}
    for seed in RUN_SEEDS:
        curriculum = run_sequence(CURRICULUM, seed, "curriculum")
        control = run_sequence(CONTROL, seed, "control")
        everything[seed] = {"curriculum": curriculum, "control": control}

        print(f"\nseed {seed}: target-world scores of the last generation after each period\n")
        print(comparison_table(curriculum, control))
        print("\n(last row = same total number of generations; d = curriculum minus control)\n")

    if len(everything) > 1:
        d_top = [v["curriculum"][-1]["top_target"] - v["control"][-1]["top_target"] for v in everything.values()]
        d_smp = [v["curriculum"][-1]["sample_target"] - v["control"][-1]["sample_target"] for v in everything.values()]
        print(f"final period, curriculum minus control over {len(d_top)} seeds: "
              f"top {np.mean(d_top):+.3f} (positive on {sum(x > 0 for x in d_top)}), "
              f"sample {np.mean(d_smp):+.3f} (positive on {sum(x > 0 for x in d_smp)})")

    return everything


if __name__ == "__main__":
    everything = main()