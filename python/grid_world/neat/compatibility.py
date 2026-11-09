"""
Created on Mon Oct 19 08:26:13 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from neat.config import default_config

###############################################################################
# Compatibility distance
###############################################################################

def compatibility_distance(genome_a, genome_b, cfg=None):
    """Return the NEAT compatibility distance between two genomes.

    Only the connection genes are compared, matched by innovation number:

        distance = c1 * E / N  +  c2 * D / N  +  c3 * W

    - matching gene: present in both genomes (same innovation number);
    - excess gene: present in one genome only, with a number HIGHER than the
      biggest number of the other genome;
    - disjoint gene: present in one genome only, with a number inside the
      other genome's range;
    - E and D: how many excess and disjoint genes, counting both genomes;
    - W: average |weight_a - weight_b| over the matching genes
      (0.0 when there are none);
    - N: number of connection genes of the bigger genome (disabled genes
      count), or 1 when that number is below cfg["compatibility_n_min"].

    The enabled flag and the nodes (bias, activation) are ignored. The result
    is symmetric and 0.0 for two identical genomes. Nothing is modified.

    Parameters
    ----------
    genome_a, genome_b : dict
        The genomes to compare.
    cfg : dict, optional
        Config from default_config(). Uses compatibility_c1 (excess),
        compatibility_c2 (disjoint), compatibility_c3 (weights) and
        compatibility_n_min. Defaults to default_config().

    Returns
    -------
    float
        The distance, 0.0 or more.
    """
    if cfg is None:
        cfg = default_config()

    conns_a = genome_a["connections"]
    conns_b = genome_b["connections"]

    # biggest innovation number of each genome (-1 when it has no genes)
    max_a = max(conns_a) if conns_a else -1
    max_b = max(conns_b) if conns_b else -1

    excess = 0
    disjoint = 0
    matching = 0
    weight_diff = 0.0

    for key in conns_a:
        if key in conns_b:
            matching += 1
            weight_diff += abs(conns_a[key]["weight"] - conns_b[key]["weight"])
        elif key > max_b:
            excess += 1
        else:
            disjoint += 1

    for key in conns_b:
        if key in conns_a:
            continue
        if key > max_a:
            excess += 1
        else:
            disjoint += 1

    if matching > 0:
        avg_weight_diff = weight_diff / matching
    else:
        avg_weight_diff = 0.0

    # size of the bigger genome; small genomes are not normalised
    n = max(len(conns_a), len(conns_b))
    if n < cfg["compatibility_n_min"] or n == 0:
        n = 1

    distance = (cfg["compatibility_c1"] * excess / n
                + cfg["compatibility_c2"] * disjoint / n
                + cfg["compatibility_c3"] * avg_weight_diff)

    return float(distance)