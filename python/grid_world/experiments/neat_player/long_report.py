"""
Created on Sat Oct 24 19:00:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Tables for the long runs
###############################################################################

"""
A "results" list is what run_curriculum.run_sequence returns: one dict per period with at
least val_scores (one number per generation of the period), top_target and sample_target (mean
test score of the best TOP_K genomes and of a fixed random sample of the last generation of
the period, in the target world). Periods may have different lengths.
"""

def period_ends(results: list) -> list:
    """Generation number at the end of each period: the running total of generations played."""
    ends, total = [], 0
    for r in results:
        total += len(r["val_scores"])
        ends.append(total)
    return ends


def curve_text(results: list) -> str:
    """One line per period: generations played, top, sample and gap = top - sample.
    A gap near zero means the population became nearly one genome."""
    if not results:
        raise ValueError("curve_text needs at least one period")
    head = f"{'gens':>5} {'top':>7} {'sample':>7} {'gap':>7}"
    lines = [head, "-" * len(head)]
    for end, r in zip(period_ends(results), results):
        lines.append(f"{end:>5} {r['top_target']:>7.3f} {r['sample_target']:>7.3f} "
                     f"{r['top_target'] - r['sample_target']:>7.3f}")
    return "\n".join(lines)


def compare_text(base: list, other: list, base_name: str, other_name: str) -> str:
    """Two runs side by side, period by period, and other - base for top and sample.
    Both must have the same periods (same generations at the end of each)."""
    if not base or not other:
        raise ValueError("compare_text needs at least one period in each run")
    if period_ends(base) != period_ends(other):
        raise ValueError(f"the runs have different periods: {period_ends(base)} against {period_ends(other)}")
    head = (f"{'gens':>5}  {base_name + ' top':>16} {'smp':>6}   {other_name + ' top':>16} {'smp':>6}"
            f"   {'d top':>7} {'d smp':>7}")
    lines = [head, "-" * len(head)]
    for end, b, o in zip(period_ends(base), base, other):
        lines.append(f"{end:>5}  {b['top_target']:>16.3f} {b['sample_target']:>6.3f}   "
                     f"{o['top_target']:>16.3f} {o['sample_target']:>6.3f}   "
                     f"{o['top_target'] - b['top_target']:>+7.3f} {o['sample_target'] - b['sample_target']:>+7.3f}")
    return "\n".join(lines)


def summary_text(base_by_seed: dict, other_by_seed: dict, generation: int, other_name: str, base_name: str) -> str:
    """other minus base at the period that ends at `generation`, over the seeds both runs have:
    the mean, how many seeds are positive, and the smallest and largest difference."""
    seeds = sorted(set(base_by_seed) & set(other_by_seed))
    if not seeds:
        raise ValueError("no seed has both runs")

    def at(results, seed):
        for end, r in zip(period_ends(results), results):
            if end == generation:
                return r
        raise ValueError(f"seed {seed}: no period ends at generation {generation}")

    d_top = [at(other_by_seed[s], s)["top_target"] - at(base_by_seed[s], s)["top_target"] for s in seeds]
    d_smp = [at(other_by_seed[s], s)["sample_target"] - at(base_by_seed[s], s)["sample_target"] for s in seeds]

    def part(name, d):
        return (f"{name} {sum(d) / len(d):+.3f} (positive on {sum(x > 0 for x in d)} of {len(d)}, "
                f"from {min(d):+.3f} to {max(d):+.3f})")

    return (f"generation {generation}, {other_name} minus {base_name}, seeds {seeds}:  "
            f"{part('top', d_top)};  {part('sample', d_smp)}")