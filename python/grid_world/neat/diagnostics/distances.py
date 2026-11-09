"""
Created on Fri Oct 23 08:38:42 2026

@author: Angelo Antonio Manzatto

Diagnostics step 4: how far apart are the genomes of a population?

Speciation puts a genome into a species when its compatibility distance to the species
representative is below cfg["compatibility_threshold"]. These functions show the whole
distribution of distances between genomes, with the threshold marked, so you can judge
whether the threshold fits the population.

"""

###############################################################################
# libraries
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

from neat.compatibility import compatibility_distance
from neat.config import default_config
from neat.diagnostics.plots import SURFACE, INK, INK_SOFT, BLUE, ORANGE, _style


def pairwise_distances(genomes, cfg=None):
    """Compatibility distance of every pair of genomes (each pair once). Needs 2 genomes or more."""
    if len(genomes) < 2:
        raise ValueError("need at least two genomes to compare")
    if cfg is None:
        cfg = default_config()
    return [compatibility_distance(genomes[i], genomes[j], cfg)
            for i in range(len(genomes)) for j in range(i + 1, len(genomes))]


def share_below(distances, threshold):
    """Fraction of the distances that are below the threshold (strictly)."""
    if not distances:
        raise ValueError("no distances")
    return sum(1 for d in distances if d < threshold) / len(distances)


def plot_distances(genomes, cfg=None, ax=None, bins=30):
    """Histogram of all pairwise distances with the compatibility threshold as a vertical line.

    The share of pairs below the threshold is in the title. It is not the same as 'share in
    the same species' (species compare against a representative) but it tells you how crowded
    the space under the threshold is.
    """
    if cfg is None:
        cfg = default_config()
    d = pairwise_distances(genomes, cfg)
    threshold = cfg["compatibility_threshold"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE, layout="constrained")
    ax.hist(d, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(threshold, color=ORANGE, linewidth=2, linestyle="--", label=f"threshold {threshold:g}")
    _style(ax, f"Genome distances: {100 * share_below(d, threshold):.0f}% of {len(d)} pairs below the threshold", "pairs")
    ax.set_xlabel("compatibility distance", color=INK_SOFT)
    ax.legend(frameon=False, labelcolor=INK_SOFT, loc="upper right")
    return ax


def distance_summary(distances, threshold):
    """Summary of a list of distances: count, mean, min, percentiles, max and the share below the threshold."""
    if not distances:
        raise ValueError("no distances")
    p10, p25, p50, p75, p90 = np.percentile(distances, [10, 25, 50, 75, 90])
    return {
        "pairs": len(distances), "mean": float(np.mean(distances)),
        "min": min(distances), "p10": float(p10), "p25": float(p25), "median": float(p50),
        "p75": float(p75), "p90": float(p90), "max": max(distances),
        "share_below": share_below(distances, threshold),
    }


def species_distances(genomes, species, cfg=None):
    """Per species: distances inside it and to the closest other species.

    species : list of rows with "id" and "members" (record["species"] of one generation).
    Returns one dict per species: id, size, within_mean, within_max (None for a single
    member), nearest (id of the species whose members are closest on average, None if there
    is only one species) and nearest_mean.
    """
    if cfg is None:
        cfg = default_config()
    rows = []
    for s in species:
        inside = [compatibility_distance(genomes[i], genomes[j], cfg)
                  for a, i in enumerate(s["members"]) for j in s["members"][a + 1:]]
        best_id, best_mean = None, None
        for other in species:
            if other["id"] == s["id"]:
                continue
            between = [compatibility_distance(genomes[i], genomes[j], cfg)
                       for i in s["members"] for j in other["members"]]
            mean = sum(between) / len(between)
            if best_mean is None or mean < best_mean:
                best_id, best_mean = other["id"], mean
        rows.append({
            "id": s["id"], "size": len(s["members"]),
            "within_mean": sum(inside) / len(inside) if inside else None,
            "within_max": max(inside) if inside else None,
            "nearest": best_id, "nearest_mean": best_mean,
        })
    return rows


def distance_table(genomes, cfg=None, species=None):
    """Text view of the distances: the overall distribution, and (given species) how separated they are.

    How to read the species part: a species is well separated when within_mean is clearly
    smaller than nearest_mean. If they are close, the species are cut through one crowd of
    similar genomes, and the compatibility threshold is probably too low.
    """
    if cfg is None:
        cfg = default_config()
    threshold = cfg["compatibility_threshold"]
    d = distance_summary(pairwise_distances(genomes, cfg), threshold)
    lines = [f"compatibility distances of {len(genomes)} genomes ({d['pairs']} pairs), threshold {threshold:g}",
             f"  min {d['min']:.2f}  p10 {d['p10']:.2f}  p25 {d['p25']:.2f}  median {d['median']:.2f}  "
             f"p75 {d['p75']:.2f}  p90 {d['p90']:.2f}  max {d['max']:.2f}  mean {d['mean']:.2f}",
             f"  {100 * d['share_below']:.1f}% of pairs are below the threshold"]
    if species:
        def f(x):
            return "-" if x is None else f"{x:.2f}"
        lines += ["", f"{'id':>4} {'size':>5} {'within_mean':>12} {'within_max':>11} {'nearest':>8} {'nearest_mean':>13}",
                  "-" * 60]
        for r in species_distances(genomes, species, cfg):
            near = "-" if r["nearest"] is None else str(r["nearest"])
            lines.append(f"{r['id']:>4} {r['size']:>5} {f(r['within_mean']):>12} {f(r['within_max']):>11} "
                         f"{near:>8} {f(r['nearest_mean']):>13}")
    return "\n".join(lines)