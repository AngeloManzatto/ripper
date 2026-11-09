"""
Created on Fri Oct 23 22:55:00 2026

@author: Angelo Antonio Manzatto

Repeat runs with several seeds, to tell a result that is real from one that was lucky,
and compare config changes on the SAME seeds.

The first experiment in EXPERIMENTS is the baseline; every later one is compared with it seed
by seed (paired). Every run uses the settings of run_rotating.py plus the experiment's own config
changes. Only the seed of the evolution changes between runs.

Measures on layouts that played no part in any choice (see run_rotating.py):
  champion : the genome with the best validation score        (the noisiest: one genome)
  top      : mean of the best TOP_K genomes of the last generation
  sample   : mean of a fixed random sample of the last generation (how good is a typical genome)

Run this file (F5 in Spyder). About 12 minutes per seed and experiment, plus about a minute of tests.
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

import experiments.neat_player.run_rotate as rr

###############################################################################
# Settings (change here)
###############################################################################

RUN_SEEDS = [0, 1, 2]
WINDOW = 5                   # generations averaged at the start and at the end of a run

# Config changes to try, one experiment at a time: name -> {config key: value}.
# The FIRST one is the baseline. Change ONE thing per experiment, so a difference has one cause.
EXPERIMENTS = {
    "default": {},
    # Default weight mutation touches 80% of the ~133 weights in every child, each by about 0.5.
    # Here only about 10% of the weights are touched.
    "weight_mutate_0.1": {"weight_mutate_prob": 0.10},
}

###############################################################################
# Summary
###############################################################################

def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def summary_table(results, window=WINDOW, baseline=None):
    """One row per seed, then the spread over seeds. With baseline (the results of the baseline
    experiment, same seeds) it adds the paired differences for champion, top and sample. Returns text."""
    head = (f"{'seed':>4} {'val start':>10} {'val end':>8} {'champion':>9} {'top':>7} {'sample':>7} {'minutes':>8}")
    if baseline:
        head += f" {'d champ':>8} {'d top':>7} {'d sample':>9}"
    lines = [head, "-" * len(head)]
    by_seed = {r["run_seed"]: r for r in baseline} if baseline else {}
    diffs = {"test_champion": [], "top_test": [], "sample_test": []}
    for r in results:
        start, end = float(np.mean(r["val_scores"][:window])), float(np.mean(r["val_scores"][-window:]))
        row = (f"{r['run_seed']:>4} {start:>10.3f} {end:>8.3f} {r['test_champion']:>9.3f} {r['top_test']:>7.3f} "
               f"{r['sample_test']:>7.3f} {r['seconds'] / 60:>8.1f}")
        if baseline:
            b = by_seed[r["run_seed"]]
            for key in diffs:
                diffs[key].append(r[key] - b[key])
            row += f" {diffs['test_champion'][-1]:>+8.3f} {diffs['top_test'][-1]:>+7.3f} {diffs['sample_test'][-1]:>+9.3f}"
        lines.append(row)
    lines.append("-" * len(head))
    for key, label in (("test_champion", "champion"), ("top_test", "top"), ("sample_test", "sample")):
        values = [r[key] for r in results]
        lines.append(f"{label:>9} over seeds: mean {np.mean(values):.3f}   min {min(values):.3f}   max {max(values):.3f}")
    if baseline:
        for key, label in (("test_champion", "champion"), ("top_test", "top"), ("sample_test", "sample")):
            d = diffs[key]
            lines.append(f"{label:>9} difference to baseline: mean {np.mean(d):+.3f}, "
                         f"positive on {sum(x > 0 for x in d)} of {len(d)} seeds")
    return "\n".join(lines)

###############################################################################
# Run
###############################################################################

def main():
    all_results = {}
    baseline = None
    for name, overrides in EXPERIMENTS.items():
        results = []
        for seed in RUN_SEEDS:
            result = rr.run_once(seed, overrides=overrides, tag=name)
            results.append(result)
            print(f"\n{name}, seed {seed} done: champion {result['test_champion']:.3f}  "
                  f"top {result['top_test']:.3f}  sample {result['sample_test']:.3f}\n")
        all_results[name] = results

        print(f"\nexperiment {name}: {overrides if overrides else 'default config'}\n")
        print(summary_table(results, baseline=baseline))
        if baseline is None:
            baseline = results

        plt.figure(figsize=(8, 4))
        for r in results:
            plt.plot(moving_average(r["val_scores"], WINDOW), label=f"seed {r['run_seed']}")
        plt.xlabel(f"generation (average of {WINDOW})")
        plt.ylabel("goal rate on validation layouts")
        plt.title(f"Validation curve per seed: {name}")
        plt.legend(frameon=False)
        plt.show()

    return all_results


if __name__ == "__main__":
    all_results = main()