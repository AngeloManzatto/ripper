"""
Created on Sat Oct 24 19:30:00 2026

@author: Angelo Antonio Manzatto

Long runs in the target world, to answer two questions:

  2. Does the Player keep improving after 120 generations, or does it hit a ceiling?
  1. Does keeping several species (so the population cannot collapse into one genome) help?

Each run trains straight in the target world (walls 0.2 + 1 enemy, weight_mutate_prob 0.1, the
"control" of run_curriculum.py) for N_PERIODS * PERIOD generations. After every period the last
generation is tested on unseen layouts (top and sample, as before), a checkpoint of the whole
population is saved, and the numbers are written to long_<variant>_seed<seed>.json.

VARIANT "baseline": the settings of the earlier control. Its first 120 generations must give
    exactly the numbers of the earlier control runs (seed 1: top 0.500 sample 0.336; seed 2:
    top 0.353 sample 0.223). The script checks this and says so. If it does not match, something
    changed and the long run cannot be compared with the earlier ones.
VARIANT "species": the same, with compatibility_threshold = SPECIES_THRESHOLD. Run look_species.py
    on a baseline checkpoint first and choose the threshold from what it shows.

Seeds are run one after the other. A table is printed after every seed; run report() at any time
to print the tables from the files written so far, also while another console is still running.
Time: about 26 minutes per 60 generations, so about 1 h 45 min per seed of 240 generations.
"""

###############################################################################
# Libraries
###############################################################################

import json
import os

from experiments.neat_player.long_report import curve_text, compare_text, summary_text
import experiments.neat_player.run_curriculum as rc

###############################################################################
# Settings (change here)
###############################################################################

VARIANT = "baseline"           # "baseline" or "species"
RUN_SEEDS = [1]                # start with one seed; add 2 and 0 afterwards
SPECIES_THRESHOLD = None       # fill in after look_species.py, only needed for "species"

PERIOD = 30                    # generations between two test points
N_PERIODS = 8                  # 8 * 30 = 240 generations

TARGET = rc.TARGET
PERIODS = [(f"target, gens {k * PERIOD + 1}-{(k + 1) * PERIOD}", TARGET, PERIOD) for k in range(N_PERIODS)]

# top and sample of the earlier control runs at generation 120 (rounded to 3 decimals)
EARLIER_CONTROL_AT_120 = {1: (0.500, 0.336), 2: (0.353, 0.223)}

RESULT_PATH = "long_{variant}_seed{seed}.json"
CHECKPOINT_PATH = "ckpt_{mode}_period{period}_seed{seed}.pkl"      # mode = "long-<variant>"

###############################################################################
# Run
###############################################################################

def overrides_for(variant):
    """Config changes of a variant: the earlier control's changes, plus the species threshold."""
    overrides = dict(rc.OVERRIDES)
    if variant == "baseline":
        return overrides
    if variant == "species":
        if SPECIES_THRESHOLD is None:
            raise ValueError("set SPECIES_THRESHOLD first (look at a baseline checkpoint with look_species.py)")
        overrides["compatibility_threshold"] = SPECIES_THRESHOLD
        return overrides
    raise ValueError(f"unknown variant {variant!r}")


def run_one_seed(variant, seed):
    results = rc.run_sequence(PERIODS, seed, f"long-{variant}", overrides_for(variant), CHECKPOINT_PATH)
    with open(RESULT_PATH.format(variant=variant, seed=seed), "w") as file:
        json.dump(results, file, indent=1)
    return results


def check_against_earlier(variant, seed, results):
    """Print whether the numbers at generation 120 equal the earlier control run (baseline only)."""
    if variant != "baseline" or seed not in EARLIER_CONTROL_AT_120:
        return
    ends = 0
    for r in results:
        ends += len(r["val_scores"])
        if ends == 120:
            got = (round(r["top_target"], 3), round(r["sample_target"], 3))
            want = EARLIER_CONTROL_AT_120[seed]
            print(f"\nseed {seed} at generation 120: top {got[0]:.3f} sample {got[1]:.3f}   earlier control: "
                  f"top {want[0]:.3f} sample {want[1]:.3f}   -> {'SAME RUN' if got == want else 'DIFFERENT, do not compare'}")
            return


def main():
    for seed in RUN_SEEDS:
        results = run_one_seed(VARIANT, seed)
        check_against_earlier(VARIANT, seed, results)
        print(f"\nseed {seed}, {VARIANT}\n")
        print(curve_text(results))
    report()

###############################################################################
# Report from the files
###############################################################################

def load_all(variant):
    """{seed: results} for every long_<variant>_seed<seed>.json found in the current folder."""
    found = {}
    for name in sorted(os.listdir(".")):
        prefix, suffix = f"long_{variant}_seed", ".json"
        if name.startswith(prefix) and name.endswith(suffix):
            with open(name) as file:
                found[int(name[len(prefix):-len(suffix)])] = json.load(file)
    return found


def report():
    baseline = load_all("baseline")
    species = load_all("species")

    for seed, results in sorted(baseline.items()):
        print(f"\nbaseline, seed {seed}\n")
        print(curve_text(results))

    for seed in sorted(set(baseline) & set(species)):
        print(f"\nseed {seed}: one species against several species\n")
        print(compare_text(baseline[seed], species[seed], "baseline", "species"))

    common = sorted(set(baseline) & set(species))
    if common:
        ends = sorted({sum(len(r["val_scores"]) for r in baseline[common[0]][:k + 1]) for k in range(len(baseline[common[0]]))})
        print()
        for generation in (g for g in sorted({120, ends[-1]}) if g in ends):
            print(summary_text(baseline, species, generation, "species", "baseline"))


if __name__ == "__main__":
    main()