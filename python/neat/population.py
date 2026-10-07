"""
Created on Sat Jun  6 19:49:32 2026

@author: Angelo Antonio Manzatto
------------------
Population: the evolution loop for NEAT.

This is where all previous modules come together.

Lifecycle per generation
------------------------
1.  registry.next_generation()         — reset innovation tracking if configured
2.  fitness_fn.on_generation_start()   — allow fitness fn to update data
3.  evaluate population                — assign raw fitness to every genome
4.  speciate                           — assign genomes to species
5.  apply_fitness_sharing              — divide fitness by species size
6.  update_stagnation                  — track improvement per species
7.  remove_stagnant_species            — eliminate stuck species
8.  remove_empty_species               — clean up after removal
9.  check stopping criterion           — solved or max_generations reached
10. compute offspring quotas           — proportional to mean shared fitness
11. generate offspring                 — elitism + crossover + mutation
12. fitness_fn.on_generation_end()     — allow fitness fn to log/save

Offspring generation
--------------------
Each species receives a quota of offspring proportional to its mean
shared fitness. Within that quota:
    - elitism_count best genomes are copied unchanged (if species ≥ elitism_min)
    - remaining slots: with probability mate_only_prob → crossover only
                       otherwise → crossover (or clone) + mutation

Genome id management
--------------------
Population owns a counter and assigns unique ids to every genome
including offspring. Ids are monotonically increasing within a run.

Usage
-----
    cfg       = NEATConfig(population_size=150, max_generations=300)
    fitness   = MyFitnessFunction(data)
    stopping  = ThresholdFitness(fitness, threshold=0.9)
    pop       = Population(n_inputs=22, n_outputs=5, cfg=cfg,
                           fitness_fn=stopping)
    result    = pop.run()

    # or step-by-step:
    pop.initialise()
    for gen in range(300):
        stats = pop.step()
        print(stats)
        if pop.is_solved():
            break
    best = pop.best_genome()
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.neat.config import NEATConfig
from src.neat.genes import NodeGene, ConnectionGene, NodeType
from src.neat.genome import Genome, make_minimal_genome
from src.neat.innovation import InnovationRegistry
from src.neat.fitness import FitnessFunction, ThresholdFitness
from src.neat.species import (
    Species,
    assign_genomes,
    select_representatives,
    clear_members,
    remove_empty_species,
    apply_fitness_sharing,
    update_stagnation,
    remove_stagnant_species,
    species_summary,
)


###############################################################################
# Per-generation statistics
###############################################################################

@dataclass
class GenerationStats:
    """
    Statistics captured at the end of one generation.

    Attributes
    ----------
    generation : int
        Zero-based generation index.
    best_fitness : Optional[float]
        Highest fitness in the population (raw, before sharing).
    mean_fitness : Optional[float]
        Mean fitness across all evaluated genomes (raw).
    n_species : int
        Number of species after stagnation removal.
    n_stagnant : int
        Number of species whose stagnation_count > 0.
    population_size : int
        Number of genomes in the population this generation.
    best_genome_id : Optional[int]
        genome_id of the best genome.
    solved : bool
        Whether the stopping criterion was met this generation.
    """
    generation:     int
    best_fitness:   Optional[float]
    mean_fitness:   Optional[float]
    n_species:      int
    n_stagnant:     int
    population_size: int
    best_genome_id: Optional[int]
    solved:         bool

    def __repr__(self) -> str:
        return (
            f"Gen {self.generation:>4} | "
            f"best={self.best_fitness:.4f} | "
            f"mean={self.mean_fitness:.4f} | "
            f"species={self.n_species} | "
            f"stagnant={self.n_stagnant} | "
            f"{'SOLVED' if self.solved else ''}"
        ) if self.best_fitness is not None else (
            f"Gen {self.generation:>4} | no fitness"
        )


###############################################################################
# Evolution result
###############################################################################

@dataclass
class EvolutionResult:
    """
    The outcome of a completed evolution run.

    Attributes
    ----------
    best_genome : Optional[Genome]
        The genome with the highest fitness found during the run.
    best_fitness : Optional[float]
        Its fitness value.
    solved : bool
        Whether the stopping criterion was met.
    generations_run : int
        How many generations were executed.
    history : List[GenerationStats]
        Per-generation statistics for the entire run.
    """
    best_genome:     Optional[Genome]
    best_fitness:    Optional[float]
    solved:          bool
    generations_run: int
    history:         List[GenerationStats] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "SOLVED" if self.solved else "not solved"
        return (
            f"EvolutionResult({status}, "
            f"gens={self.generations_run}, "
            f"best={self.best_fitness})"
        )


###############################################################################
# Population
###############################################################################

class Population:
    """
    Manages the NEAT evolution loop.

    Parameters
    ----------
    n_inputs : int
        Number of input nodes in every genome.
    n_outputs : int
        Number of output nodes in every genome.
    cfg : NEATConfig
        All hyperparameters.
    fitness_fn : FitnessFunction or ThresholdFitness
        Evaluates genomes. If ThresholdFitness, used for solved() check.
    seed : Optional[int]
        Overrides cfg.seed if provided. None uses cfg.seed.
    """

    def __init__(
        self,
        n_inputs:   int,
        n_outputs:  int,
        cfg:        NEATConfig,
        fitness_fn: FitnessFunction,
        seed:       Optional[int] = None,
    ) -> None:
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")

        self.n_inputs   = n_inputs
        self.n_outputs  = n_outputs
        self.cfg        = cfg
        self.fitness_fn = fitness_fn

        # random number generator — one per population, seeded once
        _seed = seed if seed is not None else cfg.seed
        self.rng = np.random.default_rng(_seed)

        # innovation registry — lives for the full run
        self.registry = InnovationRegistry(
            reset_every_generation=False  # standard NEAT
        )
        self.registry.initialise(n_inputs, n_outputs)

        # state
        self.population:  List[Genome]   = []
        self.species:     List[Species]  = []
        self.generation:  int            = 0
        self.history:     List[GenerationStats] = []

        # genome id counter
        self._next_genome_id: int = 0

        # best genome seen across all generations
        self._all_time_best: Optional[Genome] = None

        self._initialised = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """
        Create the initial population of minimal genomes.

        All genomes start with the same minimal topology (inputs fully
        connected to outputs, no hidden nodes) but with different
        random weights.

        Must be called before step() or run().
        Safe to call again — resets all state.
        """
        self.population  = []
        self.species     = []
        self.generation  = 0
        self.history     = []
        self._all_time_best = None
        self._next_genome_id = 0

        # fresh registry
        self.registry = InnovationRegistry(reset_every_generation=False)
        self.registry.initialise(self.n_inputs, self.n_outputs)

        for _ in range(self.cfg.population_size):
            g = make_minimal_genome(
                genome_id = self._next_genome_id,
                n_inputs  = self.n_inputs,
                n_outputs = self.n_outputs,
                cfg       = self.cfg,
                registry  = self.registry,
                rng       = self.rng,
            )
            self.population.append(g)
            self._next_genome_id += 1

        self._initialised = True

    # ── Single generation step ────────────────────────────────────────────────

    def step(self) -> GenerationStats:
        """
        Execute one complete generation.

        Returns
        -------
        GenerationStats
            Statistics for this generation.

        Raises
        ------
        RuntimeError
            If initialise() has not been called.
        """
        if not self._initialised:
            raise RuntimeError(
                "Population not initialised. Call initialise() first."
            )

        # 1. innovation registry new generation
        self.registry.next_generation()

        # 2. fitness function lifecycle hook
        self.fitness_fn.on_generation_start(self.generation)

        # 3. evaluate — assign raw fitness to every genome
        scores = self.fitness_fn.evaluate_population(self.population)
        for genome, score in zip(self.population, scores):
            genome.fitness = float(score)

        # 4. speciate — assign genomes to species
        #    first generation: no existing species
        if self.generation == 0:
            self.species = []
            self.species, _ = assign_genomes(
                self.population, self.species, self.cfg,
                generation=0, next_species_id=1
            )
            self._next_species_id = max(
                (sp.species_id for sp in self.species), default=0
            ) + 1
        else:
            # select reps from current members before clearing
            select_representatives(self.species, self.rng)
            clear_members(self.species)
            self.species, self._next_species_id = assign_genomes(
                self.population, self.species, self.cfg,
                generation=self.generation,
                next_species_id=self._next_species_id,
            )

        # 5. fitness sharing — divide by species size
        #    save raw fitness for reporting before sharing modifies it
        raw_fitnesses = [g.fitness for g in self.population
                         if g.fitness is not None]
        apply_fitness_sharing(self.species)

        # 6. update stagnation
        update_stagnation(self.species, maximize=self.cfg.fitness_maximize)

        # 7. remove stagnant species
        self.species = remove_stagnant_species(
            self.species,
            self.cfg.max_stagnation,
            self.cfg.min_species_to_keep,
        )

        # 8. remove empty species (after stagnation removal may leave empties)
        self.species = remove_empty_species(self.species)

        # 9. check stopping criterion
        solved = self._check_solved()

        # track all-time best using RAW scores (before sharing modified them)
        best = self._best_genome_raw(scores)
        if best is not None:
            raw_best_fitness = scores[self.population.index(best)]
            current_best_fitness = (self._all_time_best.fitness
                                    if self._all_time_best else None)
            if self._is_better(raw_best_fitness, current_best_fitness):
                self._all_time_best = best.copy(best.genome_id)
                self._all_time_best.fitness = raw_best_fitness

        # build stats before generating offspring (population still current gen)
        stats = self._build_stats(raw_fitnesses, solved)
        self.history.append(stats)

        # report
        if self.cfg.report_every > 0 and \
                self.generation % self.cfg.report_every == 0:
            print(stats)

        # 10-11. generate offspring (skip if solved — caller decides whether
        #        to continue)
        if not solved:
            self.population = self._generate_offspring()

        # 12. fitness function lifecycle hook
        self.fitness_fn.on_generation_end(self.generation, self.population)

        self.generation += 1
        return stats

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self) -> EvolutionResult:
        """
        Execute the full evolution until solved or max_generations.

        Calls initialise() if not already called.

        Returns
        -------
        EvolutionResult
            Final result including best genome and history.
        """
        if not self._initialised:
            self.initialise()

        solved = False
        for _ in range(self.cfg.max_generations):
            stats = self.step()
            if stats.solved:
                solved = True
                break

        return EvolutionResult(
            best_genome     = self._all_time_best,
            best_fitness    = (self._all_time_best.fitness
                               if self._all_time_best else None),
            solved          = solved,
            generations_run = self.generation,
            history         = list(self.history),
        )

    # ── Public accessors ──────────────────────────────────────────────────────

    def best_genome(self) -> Optional[Genome]:
        """Return the best genome seen across all generations."""
        return self._all_time_best

    def is_solved(self) -> bool:
        """Return True if the last generation met the stopping criterion."""
        return bool(self.history and self.history[-1].solved)

    def current_population(self) -> List[Genome]:
        """Return the current population (read-only view)."""
        return list(self.population)

    def __repr__(self) -> str:
        return (
            f"Population("
            f"n_inputs={self.n_inputs}, "
            f"n_outputs={self.n_outputs}, "
            f"size={self.cfg.population_size}, "
            f"generation={self.generation}, "
            f"species={len(self.species)})"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_solved(self) -> bool:
        """Return True if any genome meets the stopping criterion."""
        if isinstance(self.fitness_fn, ThresholdFitness):
            return self.fitness_fn.solved_any(self.population)
        # no threshold — never auto-stop
        return False

    def _is_better(self, a: Optional[float], b: Optional[float]) -> bool:
        """Return True if fitness a is better than b."""
        if a is None:
            return False
        if b is None:
            return True
        return a > b if self.cfg.fitness_maximize else a < b

    def _best_genome_raw(self, raw_scores: List[float]) -> Optional[Genome]:
        """Return the genome with the best raw fitness this generation."""
        if not self.population or not raw_scores:
            return None
        if self.cfg.fitness_maximize:
            idx = int(np.argmax(raw_scores))
        else:
            idx = int(np.argmin(raw_scores))
        return self.population[idx]

    def _build_stats(
        self,
        raw_fitnesses: List[float],
        solved: bool,
    ) -> GenerationStats:
        best_fit = max(raw_fitnesses) if raw_fitnesses else None
        mean_fit = (sum(raw_fitnesses) / len(raw_fitnesses)
                    if raw_fitnesses else None)
        n_stagnant = sum(1 for sp in self.species
                         if sp.stagnation_count > 0)
        best_g = None
        if raw_fitnesses and self.population:
            if self.cfg.fitness_maximize:
                best_g = max(self.population,
                             key=lambda g: g.fitness or float('-inf'))
            else:
                best_g = min(self.population,
                             key=lambda g: g.fitness or float('inf'))

        return GenerationStats(
            generation      = self.generation,
            best_fitness    = best_fit,
            mean_fitness    = mean_fit,
            n_species       = len(self.species),
            n_stagnant      = n_stagnant,
            population_size = len(self.population),
            best_genome_id  = (best_g.genome_id if best_g else None),
            solved          = solved,
        )

    # ── Offspring generation ──────────────────────────────────────────────────

    def _generate_offspring(self) -> List[Genome]:
        """
        Generate the next generation of genomes.

        Steps:
        1. Compute per-species offspring quotas.
        2. For each species, produce elites + bred offspring.
        3. If total < population_size, pad with mutations of best genomes.
        4. If total > population_size, trim to exact size.

        Returns
        -------
        List[Genome]
            New population of exactly population_size genomes.
        """
        quotas = self._compute_quotas()
        offspring: List[Genome] = []

        for sp, quota in zip(self.species, quotas):
            if quota == 0 or sp.is_empty:
                continue

            # sort members by fitness descending
            scored = [m for m in sp.members if m.fitness is not None]
            if not scored:
                continue
            scored.sort(
                key=lambda g: g.fitness,
                reverse=self.cfg.fitness_maximize,
            )

            # elitism — copy best genomes unchanged
            n_elites = 0
            if sp.size >= self.cfg.elitism_min_species_size:
                n_elites = min(self.cfg.elitism_count, quota, len(scored))
                for elite in scored[:n_elites]:
                    child = elite.copy(self._next_genome_id)
                    self._next_genome_id += 1
                    offspring.append(child)

            # bred offspring
            n_breed = quota - n_elites
            # breeding pool: top survival_threshold fraction
            pool_size = max(1, int(len(scored) * self.cfg.survival_threshold))
            pool = scored[:pool_size]

            for _ in range(n_breed):
                if len(pool) >= 2 and self.rng.random() > self.cfg.mate_only_prob:
                    # crossover
                    idxs = self.rng.choice(len(pool), size=2, replace=False)
                    pa, pb = pool[int(idxs[0])], pool[int(idxs[1])]
                    child = Genome.crossover(pa, pb, self._next_genome_id,
                                            self.rng)
                else:
                    # clone best
                    child = pool[0].copy(self._next_genome_id)

                self._next_genome_id += 1

                # mutate unless mate_only
                if self.rng.random() > self.cfg.mate_only_prob:
                    child.mutate(self.cfg, self.rng, self.registry)

                offspring.append(child)

        # pad if we're short (can happen due to integer quota rounding)
        while len(offspring) < self.cfg.population_size:
            # clone + mutate a random genome from all species
            all_members = [m for sp in self.species for m in sp.members
                           if m.fitness is not None]
            if not all_members:
                break
            src = all_members[int(self.rng.integers(len(all_members)))]
            child = src.copy(self._next_genome_id)
            self._next_genome_id += 1
            child.mutate(self.cfg, self.rng, self.registry)
            offspring.append(child)

        # trim to exact size
        return offspring[:self.cfg.population_size]

    def _compute_quotas(self) -> List[int]:
        """
        Compute offspring quota per species proportional to mean shared fitness.

        Handles negative fitness by shifting all values to be non-negative.
        Species with no evaluated members get quota 0.

        Returns
        -------
        List[int]
            One quota per species, summing to approximately population_size.
        """
        mean_fitnesses: List[float] = []

        for sp in self.species:
            scored = [m.fitness for m in sp.members if m.fitness is not None]
            if scored:
                mean_fitnesses.append(sum(scored) / len(scored))
            else:
                mean_fitnesses.append(0.0)

        # shift to non-negative if any values are negative
        min_f = min(mean_fitnesses) if mean_fitnesses else 0.0
        if min_f < 0:
            mean_fitnesses = [f - min_f for f in mean_fitnesses]

        total = sum(mean_fitnesses)

        if total <= 0:
            # equal quotas when all species have zero fitness
            base = self.cfg.population_size // max(len(self.species), 1)
            quotas = [base] * len(self.species)
        else:
            raw = [f / total * self.cfg.population_size
                   for f in mean_fitnesses]
            quotas = [int(r) for r in raw]

            # distribute remainder to highest-fraction species
            remainder = self.cfg.population_size - sum(quotas)
            fractions = [(r - int(r), i) for i, r in enumerate(raw)]
            fractions.sort(reverse=True)
            for _, i in fractions[:remainder]:
                quotas[i] += 1

        return quotas