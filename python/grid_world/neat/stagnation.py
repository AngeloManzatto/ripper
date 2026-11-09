"""
Created on Mon Oct 19 09:08:52 2026

@author: Angelo Antonio Manzatto

Stagnation: find species that stopped improving, so they can be removed.

A species is STAGNANT when its best fitness has not improved for too long.
Removing it stops the population from spending offspring on a dead end.

The stagnation record is plain data, kept next to the species set (not inside
it) and replaced every generation by the result of update_stagnation:

    {species_id: {"best_fitness": float, "last_improved": generation}}

"best_fitness" is the best value EVER seen by that species (a record that only
goes up) and "last_improved" is the generation where the record was last beaten.
Fitness is always MAXIMISED (to minimise something, negate it).

Per generation: speciate -> update_stagnation -> find_stagnant -> remove_species.

"""

###############################################################################
# libraries
###############################################################################

import copy
import math

from neat.config import default_config


def update_stagnation(stagnation, species_set, fitnesses, generation):
    """Update the best-fitness record of every species and return it.

    - The best of a species is the maximum fitness over its members
      (fitnesses[i] is the fitness of genome i, the same list that was given
      to speciate).
    - A species that is not in the record yet starts with that best and
      last_improved = generation.
    - A known species improves only if its best is STRICTLY above its record;
      then both fields are updated, otherwise the old record is kept.
    - Species that are no longer in the species set are dropped from the record.

    Nothing is modified; a new record is returned.

    Parameters
    ----------
    stagnation : dict
        Record of the previous generation ({} at the start).
    species_set : dict
        The current species set, from speciate.
    fitnesses : list of float
        Fitness of every genome, by position.
    generation : int
        The current generation number.

    Returns
    -------
    dict
        The new record, with best_fitness as a Python float.

    Raises
    ------
    ValueError
        If a member's fitness is NaN.
    IndexError
        If a member has no entry in fitnesses.
    """
    new_stagnation = {}

    for species_id in sorted(species_set["species"]):
        members = species_set["species"][species_id]["members"]

        best = None
        for index in members:
            fitness = fitnesses[index]
            if math.isnan(fitness):
                raise ValueError("fitness is NaN")
            if best is None or fitness > best:
                best = fitness
        best = float(best)

        if species_id not in stagnation:
            new_stagnation[species_id] = {"best_fitness": best, "last_improved": generation}
        else:
            old = stagnation[species_id]
            if best > old["best_fitness"]:
                new_stagnation[species_id] = {"best_fitness": best, "last_improved": generation}
            else:
                new_stagnation[species_id] = {"best_fitness": old["best_fitness"],
                                              "last_improved": old["last_improved"]}

    return new_stagnation


def find_stagnant(stagnation, generation, cfg=None):
    """Return the ids of the species that should be removed.

    A species is stagnant when generation - last_improved is at least
    cfg["max_stagnation"]. But at least cfg["min_species_to_keep"] species are
    always kept: if removing every stagnant species would leave fewer, the
    stagnant species with the best records are spared (the worst records are
    removed first; on a tie the higher id, the newer species, goes first).

    Nothing is modified.

    Parameters
    ----------
    stagnation : dict
        The record from update_stagnation.
    generation : int
        The current generation number.
    cfg : dict, optional
        Config from default_config(). Uses max_stagnation and
        min_species_to_keep. Defaults to default_config().

    Returns
    -------
    list of int
        Species ids to remove, in ascending order.
    """
    if cfg is None:
        cfg = default_config()

    # worst record first; on a tie the newer (higher id) species first
    order = sorted(stagnation, key=lambda sid: (stagnation[sid]["best_fitness"], -sid))

    remaining = len(stagnation)
    stagnant = []
    for species_id in order:
        if remaining <= cfg["min_species_to_keep"]:
            break
        waited = generation - stagnation[species_id]["last_improved"]
        if waited >= cfg["max_stagnation"]:
            stagnant.append(species_id)
            remaining -= 1

    return sorted(stagnant)


def remove_species(species_set, species_ids):
    """Return a new species set without the given species.

    The remaining species are deep copies and next_id is unchanged. The
    input is not modified.

    Parameters
    ----------
    species_set : dict
        The species set from speciate.
    species_ids : list of int
        Ids to remove (usually the result of find_stagnant).

    Returns
    -------
    dict
        The new species set.

    Raises
    ------
    ValueError
        If an id is not in the species set.
    """
    for species_id in species_ids:
        if species_id not in species_set["species"]:
            raise ValueError(f"unknown species id {species_id}")

    new_species = {}
    for species_id, species in species_set["species"].items():
        if species_id in species_ids:
            continue
        new_species[species_id] = copy.deepcopy(species)

    return {"species": new_species, "next_id": species_set["next_id"]}