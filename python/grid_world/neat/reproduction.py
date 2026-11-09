"""
Created on Mon Oct 19 11:16:02 2026

@author: Angelo Antonio Manzatto

compute_offspring already decided HOW MANY children each species gets (counts).
Reproduction decides WHICH children: elites copied unchanged, and the rest bred
from the best members of the species by crossover and mutation.

Per generation:
    speciate -> update_stagnation -> find_stagnant -> remove_species
    -> compute_offspring -> reproduce
    
"""

###############################################################################
# libraries
###############################################################################

import copy
import math

from neat.config import default_config
from neat.crossover import crossover
from neat.mutation import mutate


def rank_members(members, fitnesses):
    """Return the member indices sorted from best to worst fitness.

    Equal fitness: the lower index first. Nothing is modified.

    Parameters
    ----------
    members : list of int
        Positions of the members in the population list.
    fitnesses : list of float
        Fitness of every genome, by position.

    Returns
    -------
    list of int
        The members, best first (Python ints).

    Raises
    ------
    ValueError
        If a member's fitness is NaN.
    """
    for index in members:
        if math.isnan(fitnesses[index]):
            raise ValueError("fitness is NaN")
    return [int(i) for i in sorted(members, key=lambda index: (-fitnesses[index], index))]


def parent_pool(ranked, cfg=None):
    """Return the members that are allowed to become parents.

    The first ceil(survival_threshold * size) members of `ranked` (best first),
    but never fewer than 2 (or the whole species when it has fewer) and never
    more than the species. The product is rounded to 9 decimals before
    rounding up, so float noise such as 0.07 * 100 = 7.000000000000001 does not
    add a parent.

    Parameters
    ----------
    ranked : list of int
        Members of one species, best first (from rank_members).
    cfg : dict, optional
        Config from default_config(). Uses survival_threshold. Defaults to
        default_config().

    Returns
    -------
    list of int
        The parent pool, best first. `ranked` is not modified.
    """
    if cfg is None:
        cfg = default_config()

    size = len(ranked)
    cutoff = int(math.ceil(round(cfg["survival_threshold"] * size, 9)))
    cutoff = max(cutoff, min(2, size))
    cutoff = min(cutoff, size)

    return list(ranked[:cutoff])


def find_champion(species_set, fitnesses):
    """Return (index, species_id) of the best genome among all species members.

    Equal fitness: the lowest index. Genomes that are in no species are ignored.

    Parameters
    ----------
    species_set : dict
        The species set (after stagnant species were removed).
    fitnesses : list of float
        Fitness of every genome, by position.

    Returns
    -------
    tuple
        (index, species_id), or (None, None) if there are no members.

    Raises
    ------
    ValueError
        If a member's fitness is NaN.
    """
    champion = None
    champion_species = None
    for species_id in sorted(species_set["species"]):
        for index in species_set["species"][species_id]["members"]:
            fitness = fitnesses[index]
            if math.isnan(fitness):
                raise ValueError("fitness is NaN")
            if champion is None or fitness > fitnesses[champion] or (
                    fitness == fitnesses[champion] and index < champion):
                champion = index
                champion_species = species_id
    return champion, champion_species


def ensure_champion_slot(species_set, fitnesses, counts):
    """Make sure the species of the champion has at least one child.

    If it has 0, one child is taken from the species with the most children
    (lowest id on a tie) and given to it. The total does not change. `counts`
    is not modified; a new dict is returned.

    Parameters
    ----------
    species_set : dict
        The species set.
    fitnesses : list of float
        Fitness of every genome, by position.
    counts : dict
        {species_id: children} from compute_offspring.

    Returns
    -------
    dict
        The corrected counts.
    """
    new_counts = dict(counts)

    champion, champion_species = find_champion(species_set, fitnesses)
    if champion is None or new_counts[champion_species] >= 1:
        return new_counts

    donor = None
    for species_id in sorted(new_counts):
        if donor is None or new_counts[species_id] > new_counts[donor]:
            donor = species_id
    if new_counts[donor] < 1:
        return new_counts

    new_counts[donor] -= 1
    new_counts[champion_species] += 1
    return new_counts


def reproduce(genomes, fitnesses, species_set, counts, innovations, rng, cfg=None):
    """Build the next population and return it as a list of new genomes.

    The champion of the whole population always survives unchanged (see
    ensure_champion_slot). Then, for every species with at least one child, in
    ascending id order:

    - ELITES: the best elitism_count members are copied unchanged, if the
      species has at least elitism_min_species_size members (never more than
      the species' children). The champion's species always keeps its
      champion, even if it is small. Elites count as children of the species.
    - The other children are bred from the species' parent pool:
        * pool of 2 or more: two different parents are drawn; the fitter one is
          parent_a of crossover (equal fitness: random order). Then, unless a
          roll below mate_only_prob says "mate only", the child is mutated.
        * pool of 1: the child is a mutated clone of that parent.

    The registry `innovations` is updated in place by the mutations. The given
    genomes, fitnesses, species_set, counts and cfg are never modified, and the
    children share nothing with the parents. The same seed gives the same
    population.

    Parameters
    ----------
    genomes : list of dict
        The current population.
    fitnesses : list of float
        Fitness of every genome, by position.
    species_set : dict
        The species set (after stagnant species were removed).
    counts : dict
        {species_id: children} from compute_offspring. Must have exactly the
        species ids of the species set and no negative value.
    innovations : dict
        Shared innovation registry.
    rng : np.random.Generator
        Source of randomness.
    cfg : dict, optional
        Config from default_config(). Uses elitism_count,
        elitism_min_species_size, survival_threshold, mate_only_prob and
        everything mutate uses. Defaults to default_config().

    Returns
    -------
    list of dict
        The new population, sum(counts) genomes (after the champion slot
        correction the total is unchanged).

    Raises
    ------
    ValueError
        If counts do not match the species ids, a count is negative, or a
        fitness is NaN.
    """
    if cfg is None:
        cfg = default_config()

    if set(counts) != set(species_set["species"]):
        raise ValueError("counts and species ids do not match")
    for species_id in counts:
        if counts[species_id] < 0:
            raise ValueError("negative offspring count")

    counts = ensure_champion_slot(species_set, fitnesses, counts)
    champion, champion_species = find_champion(species_set, fitnesses)

    population = []
    for species_id in sorted(species_set["species"]):
        quota = counts[species_id]
        if quota == 0:
            continue

        members = species_set["species"][species_id]["members"]
        ranked = rank_members(members, fitnesses)

        # elites: copied unchanged, they count as children of this species
        n_elites = 0
        if len(members) >= cfg["elitism_min_species_size"]:
            n_elites = cfg["elitism_count"]
        if species_id == champion_species:
            n_elites = max(n_elites, 1)
        n_elites = min(n_elites, quota)
        for index in ranked[:n_elites]:
            population.append(copy.deepcopy(genomes[index]))

        # the other children are bred from the best members
        pool = parent_pool(ranked, cfg)
        for _ in range(quota - n_elites):
            if len(pool) >= 2:
                picks = rng.choice(len(pool), size=2, replace=False)
                first, second = int(min(picks)), int(max(picks))
                parent_a, parent_b = pool[first], pool[second]      # parent_a is fitter
                if fitnesses[parent_a] == fitnesses[parent_b] and rng.random() < 0.5:
                    parent_a, parent_b = parent_b, parent_a
                child = crossover(genomes[parent_a], genomes[parent_b], rng)
                if rng.random() >= cfg["mate_only_prob"]:
                    child = mutate(child, rng, innovations, cfg)
            else:
                child = mutate(genomes[pool[0]], rng, innovations, cfg)
            population.append(child)

    return population