"""
Created on Mon Oct 19 09:26:27 2026

@author: Angelo Antonio Manzatto

Offspring counts: how many children each species gets next generation.

This only decides the NUMBERS. Choosing the parents and building the children
(selection, crossover, mutation) is done by reproduction.

Species are rewarded by their MEAN fitness (fitness sharing): a big species
does not win just because it is big, and a small species with good genomes
still gets children.

"""

###############################################################################
# libraries
###############################################################################

import math

from neat.config import default_config


def compute_offspring(species_set, fitnesses, cfg=None):
    """Divide the next population among the species and return the counts.

    1. Each species gets a score: the mean fitness of its members minus the
       lowest member fitness among all species in the set. Only genomes that
       belong to a species in the set count; fitnesses of other genomes (for
       example from species removed as stagnant) are ignored. Because of the
       shift, adding a constant to every fitness, or multiplying all of them
       by a positive constant, changes nothing.
    2. cfg["population_size"] is divided in proportion to the scores. If every
       score is 0 (all fitnesses equal) it is divided equally.
    3. The division is exact. Each species gets the whole part of its share and
       the children left over go one by one to the species with the biggest
       fractional part (on a tie, the lower species id first), so the counts
       always add up to population_size.
    4. A species with score 0 gets no children (unless everything is equal).
       There is no minimum; keeping the best genome alive is the job of
       reproduction.

    No randomness is used and nothing is modified.

    Parameters
    ----------
    species_set : dict
        The species set (after stagnant species were removed).
    fitnesses : list of float
        Fitness of every genome, by position (the list given to speciate).
    cfg : dict, optional
        Config from default_config(). Uses population_size. Defaults to
        default_config().

    Returns
    -------
    dict
        {species_id: count}, Python ints, ascending species ids.

    Raises
    ------
    ValueError
        If the species set is empty or a fitness is NaN.
    IndexError
        If a member has no entry in fitnesses.
    """
    if cfg is None:
        cfg = default_config()

    species = species_set["species"]
    if not species:
        raise ValueError("no species to give offspring to")

    # mean fitness of every species, and the lowest member fitness of all
    means = {}
    lowest = None
    for species_id in sorted(species):
        members = species[species_id]["members"]
        total = 0.0
        for index in members:
            fitness = fitnesses[index]
            if math.isnan(fitness):
                raise ValueError("fitness is NaN")
            total += fitness
            if lowest is None or fitness < lowest:
                lowest = fitness
        means[species_id] = total / len(members)

    # score = mean fitness shifted so the lowest member fitness is 0
    scores = {}
    for species_id in sorted(species):
        scores[species_id] = means[species_id] - lowest
    score_total = sum(scores.values())

    # everything equal: every species counts the same
    if score_total <= 0.0:
        for species_id in scores:
            scores[species_id] = 1.0
        score_total = float(len(scores))

    # whole part of each share, remember the fractional parts
    population_size = cfg["population_size"]
    counts = {}
    fractions = []
    for species_id in sorted(species):
        quota = scores[species_id] / score_total * population_size
        whole = int(math.floor(quota))
        counts[species_id] = whole
        fractions.append((quota - whole, species_id))

    # children left over go to the biggest fractions (tie: lower id first)
    leftover = population_size - sum(counts.values())
    fractions.sort(key=lambda item: (-item[0], item[1]))
    for position in range(leftover):
        counts[fractions[position][1]] += 1

    return counts