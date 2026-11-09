"""
Created on Sat Oct 24 21:30:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Novelty: how differently does a genome behave from the others?
###############################################################################

"""
The behaviour of a genome in one generation is the list of sets of cells it stood on, one set
per game. All genomes of a generation play the same layouts, so game i of one genome can be
compared with game i of another.

    distance between two genomes = mean over the games of the Jaccard distance of their sets
    novelty of a genome          = mean distance to its k nearest genomes in the same generation
    fitness                      = (1 - weight) * goal rate + weight * novelty / biggest novelty

With weight 0 the fitness is exactly the goal rate.
"""

def jaccard_distance(a, b) -> float:
    """1 - |a and b| / |a or b| for two sets of cells: 0.0 for equal sets, 1.0 for sets with no cell
    in common. Two empty sets are equal, so their distance is 0.0."""
    a, b = set(a), set(b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


def behaviour_distance(visits_a: list, visits_b: list) -> float:
    """Mean Jaccard distance over the games. visits_x is the list of cell sets of one genome, one
    per game, on the same layouts. Raises ValueError for lists of different length or empty lists."""
    if len(visits_a) != len(visits_b):
        raise ValueError(f"the genomes played {len(visits_a)} and {len(visits_b)} games")
    if len(visits_a) == 0:
        raise ValueError("behaviour_distance needs at least one game")
    return sum(jaccard_distance(a, b) for a, b in zip(visits_a, visits_b)) / len(visits_a)


def novelty_scores(behaviours: list, k: int) -> list:
    """Novelty of every genome: the mean distance to its k nearest other genomes.

    behaviours: one list of cell sets per genome (see behaviour_distance).
    If there are fewer than k other genomes, all of them are used.
    Raises ValueError for fewer than 2 genomes or k < 1."""
    n = len(behaviours)
    if n < 2:
        raise ValueError("novelty needs at least two genomes")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    distance = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance[i][j] = distance[j][i] = behaviour_distance(behaviours[i], behaviours[j])

    k = min(k, n - 1)
    scores = []
    for i in range(n):
        others = sorted(distance[i][j] for j in range(n) if j != i)
        scores.append(sum(others[:k]) / k)
    return scores


def mix_fitness(goal_rates: list, novelty: list, weight: float) -> list:
    """(1 - weight) * goal rate + weight * novelty / biggest novelty, for every genome.

    Novelty is divided by the biggest one of the generation, so it lies in [0, 1] (all zeros if no
    genome is novel). weight must be in [0, 1]; with 0 the result is the goal rates themselves.
    Raises ValueError for lists of different length, empty lists or a weight outside [0, 1]."""
    if len(goal_rates) != len(novelty):
        raise ValueError(f"{len(goal_rates)} goal rates but {len(novelty)} novelty scores")
    if len(goal_rates) == 0:
        raise ValueError("mix_fitness needs at least one genome")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")

    biggest = max(novelty)
    scaled = [x / biggest for x in novelty] if biggest > 0 else [0.0] * len(novelty)
    return [(1.0 - weight) * g + weight * s for g, s in zip(goal_rates, scaled)]