"""
Created on Mon Oct 19 08:54:19 2026

@author: Angelo Antonio Manzatto

Speciation: group a population into species of similar genomes.

Species protect new structure. A genome with a new node or connection usually
scores worse at first, so it competes only against genomes that are similar to
it, not against the whole population.

The species set is plain data, created once by init_species and then replaced
by the result of speciate every generation:

    {
        "species": {species_id: {"representative": genome,
                                 "members": [index, ...]}},
        "next_id": int,    # id the next new species receives
    }

"members" are positions in the list of genomes given to speciate (this
generation only). "representative" is a stored copy of one genome; the next
generation compares its genomes to it.

"""

###############################################################################
# libraries
###############################################################################

import copy

from neat.config import default_config
from neat.compatibility import compatibility_distance


def init_species():
    """Return an empty species set (no species yet, first id is 0).

    Returns
    -------
    dict
        {"species": {}, "next_id": 0}
    """
    return {"species": {}, "next_id": 0}


def speciate(genomes, species_set, cfg=None):
    """Assign every genome to a species and return the new species set.

    Steps:

    1. Genomes are taken in list order. Each one is compared with the species
       representatives, in increasing species-id order, and joins the FIRST
       species whose representative is at a distance STRICTLY LESS than
       cfg["compatibility_threshold"] (compatibility_distance, same cfg).
    2. A genome that fits no species founds a new one: its id is next_id (which
       then grows by one) and the genome is the representative right away, so
       later genomes of the same call can join it.
    3. Species left without members are dropped (their ids are never reused).
       Each surviving species gets a new representative: its member closest to
       the OLD representative (the first one in case of a tie), stored as a
       deep copy.

    Every genome ends in exactly one species. Members are Python ints in
    ascending order. Nothing is modified: the genomes, species_set and cfg are
    left as they were, and a new species set is returned.

    Parameters
    ----------
    genomes : list of dict
        The current population.
    species_set : dict
        The species set of the previous generation (init_species() at the
        start).
    cfg : dict, optional
        Config from default_config(). Uses compatibility_threshold and the
        compatibility_* coefficients. Defaults to default_config().

    Returns
    -------
    dict
        The new species set.
    """
    if cfg is None:
        cfg = default_config()

    old_species = species_set["species"]
    next_id = species_set["next_id"]

    # representatives to compare with, and the members found so far
    representatives = {}
    members = {}
    for species_id in sorted(old_species):
        representatives[species_id] = old_species[species_id]["representative"]
        members[species_id] = []

    # step 1 and 2: place every genome
    for index, genome in enumerate(genomes):
        placed = False
        for species_id in sorted(representatives):
            distance = compatibility_distance(genome, representatives[species_id], cfg)
            if distance < cfg["compatibility_threshold"]:
                members[species_id].append(index)
                placed = True
                break
        if not placed:
            species_id = next_id
            next_id += 1
            representatives[species_id] = genome
            members[species_id] = [index]

    # step 3: drop empty species, choose the new representatives
    new_species = {}
    for species_id in sorted(members):
        if not members[species_id]:
            continue

        best = members[species_id][0]
        best_distance = compatibility_distance(genomes[best], representatives[species_id], cfg)
        for index in members[species_id][1:]:
            distance = compatibility_distance(genomes[index], representatives[species_id], cfg)
            if distance < best_distance:
                best = index
                best_distance = distance

        new_species[species_id] = {
            "representative": copy.deepcopy(genomes[best]),
            "members": members[species_id],
        }

    return {"species": new_species, "next_id": next_id}