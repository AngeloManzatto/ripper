"""
Created on Sat Oct 24 22:30:00 2026

@author: Angelo Antonio Manzatto

First look at novelty in the Player's fitness.

The evolution is told that a genome is worth

    (1 - NOVELTY_WEIGHT) * goal rate  +  NOVELTY_WEIGHT * novelty

where novelty says how differently the genome moves, compared with the other genomes of the same
generation, on the same 20 layouts (see novelty.py: the cells it visits in each game, compared with
the K_NEIGHBOURS closest genomes). Everything else is the baseline of run_long.py: the target world,
weight_mutate_prob 0.1, 150 genomes, 20 games each, the same training layouts per generation.

What is reported does not change: the printed best / mean / val are goal rates, and top / sample /
champion are goal rates on the unseen test layouts. Only the choice of who breeds uses the mix.

Checks before believing anything:
  - generation 0 must print exactly the numbers of the baseline for the same seed (seed 1: best 0.150
    mean 0.029 val 0.010), because the first population and the first layouts are the same and only
    the breeding depends on the weight;
  - NOVELTY_WEIGHT = 0.0 must reproduce the baseline completely (seed 1 at generation 120: top 0.500
    sample 0.336). The script says SAME RUN or DIFFERENT. It costs a full run, so do it only if the
    generation 0 line looks wrong or you want the proof.

The comparison is with long_baseline_seed<seed>.json (written by run_long.py), over the same number of
generations. Remember that one seed differs from another by about +-0.2 in top: look at the difference
on the same seed, and ask for more seeds before trusting it.

About 55 minutes per seed for 120 generations (novelty adds a second or two per generation).
"""

###############################################################################
# Libraries
###############################################################################

import json
import os

from experiments.neat_player.fitness import evaluate_population_novelty
from experiments.neat_player.long_report import curve_text, compare_text, summary_text
import experiments.neat_player.run_curriculum as rc
import experiments.neat_player.run_rotating as rr

###############################################################################
# Settings (change here)
###############################################################################

NOVELTY_WEIGHT = 0.2           # 0 = the baseline; 1 = novelty only
K_NEIGHBOURS = 10              # novelty = mean distance to this many closest genomes
RUN_SEEDS = [1]

PERIOD = 30
N_PERIODS = 4                  # 4 * 30 = 120 generations, the length of the earlier comparisons

TARGET = rc.TARGET
PERIODS = [(f"target, gens {k * PERIOD + 1}-{(k + 1) * PERIOD}", TARGET, PERIOD) for k in range(N_PERIODS)]

# top and sample of the baseline at generation 120 (rounded to 3 decimals), for the weight 0 check
BASELINE_AT_120 = {0: (0.377, 0.224), 1: (0.500, 0.336), 2: (0.353, 0.223)}

# The baseline after generations 30, 60, 90 and 120: (top, sample) in the target world. These are the
# control runs of run_curriculum.py, which are the same run as the baseline of run_long.py (seed 1
# was checked: SAME RUN). Used when there is no long_baseline_seed<seed>.json.
BASELINE_CURVES = {
    0: [(0.068, 0.044), (0.148, 0.105), (0.291, 0.172), (0.377, 0.224)],
    1: [(0.121, 0.095), (0.232, 0.176), (0.486, 0.337), (0.500, 0.336)],
    2: [(0.067, 0.043), (0.174, 0.098), (0.320, 0.257), (0.353, 0.223)],
}

MODE = "novelty{weight:g}"
RESULT_PATH = "novelty_w{weight:g}_seed{seed}.json"
CHECKPOINT_PATH = "ckpt_{mode}_period{period}_seed{seed}.pkl"

###############################################################################
# Run
###############################################################################

def score_population(genomes, layout_seed, kwargs):
    """(fitnesses for the evolution, goal rates for the report) of one generation."""
    return evaluate_population_novelty(genomes, rr.N_EPISODES, layout_seed, NOVELTY_WEIGHT, K_NEIGHBOURS, kwargs)


def run_one_seed(seed):
    mode = MODE.format(weight=NOVELTY_WEIGHT)
    results = rc.run_sequence(PERIODS, seed, mode, checkpoint_template=CHECKPOINT_PATH,
                              score_population=score_population)
    with open(RESULT_PATH.format(weight=NOVELTY_WEIGHT, seed=seed), "w") as file:
        json.dump(results, file, indent=1)
    return results


def check_weight_zero(seed, results):
    if NOVELTY_WEIGHT != 0.0 or seed not in BASELINE_AT_120:
        return
    last = results[-1]
    got = (round(last["top_target"], 3), round(last["sample_target"], 3))
    want = BASELINE_AT_120[seed]
    print(f"\nseed {seed} at generation 120: top {got[0]:.3f} sample {got[1]:.3f}   baseline: "
          f"top {want[0]:.3f} sample {want[1]:.3f}   -> {'SAME RUN' if got == want else 'DIFFERENT'}")


def baseline_for(seed, n_periods):
    """The baseline of a seed as a list of periods (the format of run_sequence results), first n_periods.
    From long_baseline_seed<seed>.json if it exists, else from BASELINE_CURVES."""
    path = f"long_baseline_seed{seed}.json"
    if os.path.exists(path):
        with open(path) as file:
            return json.load(file)[:n_periods]
    return [{"val_scores": [0.0] * PERIOD, "top_target": top, "sample_target": sample}
            for top, sample in BASELINE_CURVES[seed][:n_periods]]


def report(seed):
    """Novelty against the baseline of the same seed, over the same generations."""
    with open(RESULT_PATH.format(weight=NOVELTY_WEIGHT, seed=seed)) as file:
        novelty = json.load(file)

    print(f"\nseed {seed}: baseline against novelty weight {NOVELTY_WEIGHT:g}\n")
    print(compare_text(baseline_for(seed, len(novelty)), novelty, "baseline", "novelty"))


def report_all():
    """Over every seed with a finished novelty file: mean difference at the last generation."""
    novelty_by_seed = {}
    for seed in sorted(BASELINE_CURVES):
        path = RESULT_PATH.format(weight=NOVELTY_WEIGHT, seed=seed)
        if os.path.exists(path):
            with open(path) as file:
                novelty_by_seed[seed] = json.load(file)
    if len(novelty_by_seed) < 2:
        return
    baseline_by_seed = {seed: baseline_for(seed, N_PERIODS) for seed in novelty_by_seed}
    print()
    print(summary_text(baseline_by_seed, novelty_by_seed, N_PERIODS * PERIOD, "novelty", "baseline"))


def main():
    for seed in RUN_SEEDS:
        results = run_one_seed(seed)
        check_weight_zero(seed, results)
        print(f"\nseed {seed}, novelty weight {NOVELTY_WEIGHT:g}\n")
        print(curve_text(results))
        report(seed)
    report_all()


if __name__ == "__main__":
    main()