"""
Created on Sat Jun  6 19:05:48 2026

@author: Angelo Antonio Manzatto
---------------
Speciation for NEAT.

Why speciation exists
---------------------
Structural mutations (add node, add connection) temporarily reduce
fitness — weights haven't been tuned for the new topology yet.
Without protection, structural innovations are eliminated before
they mature. Speciation groups similar genomes so they only compete
within their group, giving innovations time to develop.

How it works
------------
Each genome is assigned to the first species whose representative
genome is within compatibility_threshold distance. If no species
matches, a new one is created with this genome as its representative.

Representatives are chosen once per generation from the previous
generation's members — before new genomes are assigned. This
separation is critical: you cannot use new members as representatives
for assigning those same members.

Fitness sharing
---------------
Each genome's raw fitness is divided by its species size:
    shared_fitness = raw_fitness / species_size

This prevents large species from dominating by making it harder to
accumulate offspring slots. It is the primary balancing mechanism
across species.

Stagnation
----------
Each species tracks its historical best fitness and how many
generations have passed without improvement. Stagnant species
(stagnation_count >= max_stagnation) are removed, unless doing so
would drop below min_species_to_keep.

Generation lifecycle
--------------------
1. select_representatives()  — choose reps from current members
2. assign_genomes(population) — assign genomes to species
3. apply_fitness_sharing()   — divide raw fitness by species size
4. update_stagnation()       — check for improvement, increment counter
5. remove_stagnant(...)      — eliminate stagnant species
"""

###############################################################################
# libraries
###############################################################################


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.neat.genome import Genome
from src.neat.config import NEATConfig


###############################################################################
# Species
###############################################################################

@dataclass
class Species:
    """
    A group of genomes with similar topology.

    Attributes
    ----------
    species_id : int
        Unique identifier. Assigned at creation, never changes.
    representative : Genome
        The genome used for compatibility distance comparisons.
        Chosen from the previous generation's members.
        Must not be None after initialisation.
    members : List[Genome]
        Genomes assigned to this species in the current generation.
        Populated by assign_genomes(), empty between generations.
    best_fitness : Optional[float]
        Best fitness ever seen in this species.
        None until at least one evaluation.
    stagnation_count : int
        Generations without improvement in best_fitness.
        Incremented by update_stagnation().
    generation_created : int
        Generation index when this species was created.
    """

    species_id:         int
    representative:     Genome
    members:            List[Genome]       = field(default_factory=list)
    best_fitness:       Optional[float]    = field(default=None)
    stagnation_count:   int                = field(default=0)
    generation_created: int                = field(default=0)

    # ── Member management ─────────────────────────────────────────────────────

    def add_member(self, genome: Genome) -> None:
        """Add a genome to this species and set its species_id."""
        genome.species_id = self.species_id
        self.members.append(genome)

    def clear_members(self) -> None:
        """Remove all members. Called before re-assignment each generation."""
        self.members.clear()

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_empty(self) -> bool:
        return len(self.members) == 0

    # ── Fitness ───────────────────────────────────────────────────────────────

    @property
    def mean_fitness(self) -> Optional[float]:
        """Mean fitness of current members. None if no members or no fitness."""
        scores = [m.fitness for m in self.members if m.fitness is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def max_fitness(self) -> Optional[float]:
        """Best fitness among current members."""
        scores = [m.fitness for m in self.members if m.fitness is not None]
        if not scores:
            return None
        return max(scores)

    def best_genome(self) -> Optional[Genome]:
        """Return the member with the highest fitness, or None."""
        scored = [m for m in self.members if m.fitness is not None]
        if not scored:
            return None
        return max(scored, key=lambda g: g.fitness)

    # ── Stagnation ────────────────────────────────────────────────────────────

    def update_stagnation(self, maximize: bool = True) -> bool:
        """
        Update stagnation counter based on current best fitness.

        Parameters
        ----------
        maximize : bool
            If True, improvement means higher fitness.
            If False, improvement means lower fitness.

        Returns
        -------
        bool
            True if this generation showed improvement (stagnation reset).
        """
        current = self.max_fitness
        if current is None:
            self.stagnation_count += 1
            return False

        if self.best_fitness is None:
            self.best_fitness    = current
            self.stagnation_count = 0
            return True

        improved = (
            current > self.best_fitness if maximize
            else current < self.best_fitness
        )

        if improved:
            self.best_fitness     = current
            self.stagnation_count = 0
            return True

        self.stagnation_count += 1
        return False

    @property
    def is_stagnant(self) -> bool:
        """True if stagnation_count >= 1. Use with max_stagnation threshold."""
        return self.stagnation_count > 0

    # ── Representative ────────────────────────────────────────────────────────

    def select_representative(self, rng: np.random.Generator) -> None:
        """
        Choose a new representative from current members.

        Called before clearing members for the next generation.
        Uses random selection from current members.

        Raises
        ------
        RuntimeError
            If species has no members (should not happen in normal use).
        """
        if not self.members:
            raise RuntimeError(
                f"Species {self.species_id} has no members — "
                "cannot select representative."
            )
        idx = int(rng.integers(len(self.members)))
        self.representative = self.members[idx]

    def select_best_representative(self) -> None:
        """
        Choose the best-fitness genome as representative.

        Alternative to random selection. Tends to produce more
        stable species over time but may reduce diversity.

        Falls back to the first member if no fitness is assigned.
        """
        best = self.best_genome()
        if best is not None:
            self.representative = best
        elif self.members:
            self.representative = self.members[0]
        else:
            raise RuntimeError(
                f"Species {self.species_id} has no members — "
                "cannot select representative."
            )

    # ── Serialization ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Species("
            f"id={self.species_id}, "
            f"size={self.size}, "
            f"best={self.best_fitness}, "
            f"stagnation={self.stagnation_count})"
        )


###############################################################################
# Speciation operations
###############################################################################

def assign_genomes(
    population:  List[Genome],
    species:     List[Species],
    cfg:         NEATConfig,
    generation:  int,
    next_species_id: int,
) -> Tuple[List[Species], int]:
    """
    Assign each genome in population to a species.

    Algorithm
    ---------
    For each genome (processed in genome_id order for reproducibility):
        1. Compute compatibility distance to each species representative.
        2. Assign to first species within compatibility_threshold.
        3. If no species matches, create a new species with this genome
           as its representative.

    Parameters
    ----------
    population : List[Genome]
        All genomes to assign. Fitness is not required.
    species : List[Species]
        Existing species with representatives already selected.
        Members should be cleared before calling this function.
    cfg : NEATConfig
        Provides compatibility_threshold and distance coefficients.
    generation : int
        Current generation index. Used when creating new species.
    next_species_id : int
        The next species id to assign when creating new species.

    Returns
    -------
    (species, next_species_id) : Tuple[List[Species], int]
        Updated species list and the next available species id.
    """
    # process in deterministic order
    sorted_pop = sorted(population, key=lambda g: g.genome_id)

    for genome in sorted_pop:
        assigned = False

        for sp in species:
            dist = genome.compatibility_distance(sp.representative, cfg)
            if dist <= cfg.compatibility_threshold:
                sp.add_member(genome)
                assigned = True
                break

        if not assigned:
            # create a new species with this genome as its representative
            new_sp = Species(
                species_id         = next_species_id,
                representative     = genome,
                generation_created = generation,
            )
            new_sp.add_member(genome)
            species.append(new_sp)
            next_species_id += 1

    return species, next_species_id


def select_representatives(
    species: List[Species],
    rng:     np.random.Generator,
    *,
    use_best: bool = False,
) -> None:
    """
    Choose representatives from current members for all species.

    Must be called before clear_members() and before assign_genomes().
    Modifies species in place.

    Parameters
    ----------
    species : List[Species]
        Species with members from the current generation.
    rng : np.random.Generator
        Used for random representative selection.
    use_best : bool
        If True, select the best-fitness member as representative.
        If False (default), select randomly.
    """
    for sp in species:
        if sp.is_empty:
            continue
        if use_best:
            sp.select_best_representative()
        else:
            sp.select_representative(rng)


def clear_members(species: List[Species]) -> None:
    """
    Clear all member lists in preparation for re-assignment.

    Must be called after select_representatives() and before
    assign_genomes().
    """
    for sp in species:
        sp.clear_members()


def remove_empty_species(species: List[Species]) -> List[Species]:
    """
    Remove species that received no members after assignment.

    Returns a new list — does not modify in place.
    """
    return [sp for sp in species if not sp.is_empty]


def apply_fitness_sharing(species: List[Species]) -> None:
    """
    Divide each genome's fitness by its species size in place.

    This prevents large species from dominating by reducing their
    per-genome fitness contribution.

    Genomes with fitness=None are skipped.
    Species with size=0 are skipped.
    """
    for sp in species:
        if sp.size == 0:
            continue
        for genome in sp.members:
            if genome.fitness is not None:
                genome.fitness = genome.fitness / sp.size


def update_stagnation(
    species:  List[Species],
    maximize: bool = True,
) -> None:
    """
    Update stagnation counters for all species in place.

    Parameters
    ----------
    species : List[Species]
    maximize : bool
        Direction of fitness improvement. Matches NEATConfig.fitness_maximize.
    """
    for sp in species:
        sp.update_stagnation(maximize=maximize)


def remove_stagnant_species(
    species:              List[Species],
    max_stagnation:       int,
    min_species_to_keep:  int,
) -> List[Species]:
    """
    Remove stagnant species while keeping a minimum number alive.

    Algorithm
    ---------
    1. Separate species into non-stagnant and stagnant.
    2. If non-stagnant count >= min_species_to_keep: return non-stagnant only.
    3. Otherwise: supplement with the best stagnant species (by best_fitness)
       until min_species_to_keep is reached.

    Parameters
    ----------
    species : List[Species]
    max_stagnation : int
        Matches NEATConfig.max_stagnation.
    min_species_to_keep : int
        Matches NEATConfig.min_species_to_keep.

    Returns
    -------
    List[Species]
        Surviving species. Returns a new list.
    """
    non_stagnant = [
        sp for sp in species
        if sp.stagnation_count < max_stagnation
    ]
    stagnant = [
        sp for sp in species
        if sp.stagnation_count >= max_stagnation
    ]

    if len(non_stagnant) >= min_species_to_keep:
        return non_stagnant

    # need to supplement from stagnant — pick best by historical fitness
    needed = min_species_to_keep - len(non_stagnant)
    rescued = sorted(
        stagnant,
        key=lambda sp: sp.best_fitness if sp.best_fitness is not None
                       else float('-inf'),
        reverse=True,
    )[:needed]

    return non_stagnant + rescued


def species_summary(species: List[Species]) -> Dict[int, dict]:
    """
    Return a summary dict for reporting/logging.

    Returns
    -------
    Dict[species_id → {size, best, mean, stagnation}]
    """
    return {
        sp.species_id: {
            "size":       sp.size,
            "best":       sp.best_fitness,
            "mean":       sp.mean_fitness,
            "stagnation": sp.stagnation_count,
        }
        for sp in species
    }


###############################################################################
# Type hint fix
###############################################################################
# Tuple imported at module level for the return type annotation
from typing import Tuple, Dict  # noqa: E402 — moved here to avoid circular