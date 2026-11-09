"""
Created on Sat Oct 24 19:45:00 2026

@author: Angelo Antonio Manzatto

How many species would a trained population split into, for different compatibility thresholds?

Genomes are put in the same species when their compatibility distance is below the threshold.
In the runs so far the weights are the only thing that evolves, every genome has the same
connections, and the distance is only 0.4 * (average weight difference): at most about 1 in a
random population and less in a trained one, against a threshold of 3. So everything was one
species and NEAT's protection of diversity never acted.

This script loads the population saved by run_long.py after a period (the genomes of the next
generation, before they are evaluated), applies the real speciation to it with a series of
thresholds, and prints how many species each gives and how big they are. No games are played,
it takes a few seconds.

Reading it: you want a handful of species (say 3 to 10) of reasonable size, not 1 and not dozens
of tiny ones. A threshold that gives 40 species of 3 or 4 genomes cuts one crowd into pieces.

Run this file (F5 in Spyder). Change CHECKPOINT to look at another period or seed.
"""

###############################################################################
# Libraries
###############################################################################

import copy

from neat.checkpoint import load_checkpoint
from neat.diagnostics.distances import pairwise_distances, distance_summary
from neat.speciation import init_species, speciate

###############################################################################
# Settings (change here)
###############################################################################

CHECKPOINT = "ckpt_long-baseline_period4_seed1.pkl"        # generation 120 of the baseline run, seed 1
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 3.0]
SHOW_SIZES = 10                                              # sizes of the biggest species to print

###############################################################################
# Species for several thresholds
###############################################################################

def species_by_threshold(genomes, cfg, thresholds, speciate_fn=speciate, new_species_set=init_species):
    """
    Speciate `genomes` from scratch once per threshold.

    Returns one dict per threshold, in the order given: threshold, n_species and sizes (the
    number of genomes of each species, biggest first). Neither genomes nor cfg are changed
    (cfg is copied before its compatibility_threshold is set).
    """
    if len(genomes) == 0:
        raise ValueError("need at least one genome")
    if len(thresholds) == 0:
        raise ValueError("need at least one threshold")

    rows = []
    for threshold in thresholds:
        trial = copy.copy(cfg)
        trial["compatibility_threshold"] = threshold
        species_set = speciate_fn(genomes, new_species_set(), trial)
        sizes = sorted((len(s["members"]) for s in species_set["species"].values()), reverse=True)
        rows.append({"threshold": threshold, "n_species": len(sizes), "sizes": sizes})
    return rows


def species_text(rows, show=10):
    """Text table: threshold, number of species, sizes of the `show` biggest (and how many more)."""
    head = f"{'threshold':>9} {'species':>8}   sizes (biggest first)"
    lines = [head, "-" * len(head)]
    for r in rows:
        sizes = " ".join(str(n) for n in r["sizes"][:show])
        more = len(r["sizes"]) - show
        lines.append(f"{r['threshold']:>9g} {r['n_species']:>8}   {sizes}" + (f"  (+{more} more)" if more > 0 else ""))
    return "\n".join(lines)

###############################################################################
# Run
###############################################################################

def main():
    state, _, cfg = load_checkpoint(CHECKPOINT)
    genomes = state["genomes"]

    d = distance_summary(pairwise_distances(genomes, cfg), cfg["compatibility_threshold"])
    print(f"{CHECKPOINT}: generation {state['generation']}, {len(genomes)} genomes")
    print(f"distances: min {d['min']:.3f}  p10 {d['p10']:.3f}  median {d['median']:.3f}  "
          f"p90 {d['p90']:.3f}  max {d['max']:.3f}\n")

    print(species_text(species_by_threshold(genomes, cfg, THRESHOLDS), SHOW_SIZES))


if __name__ == "__main__":
    main()