"""
Created on Sat Jun  6 19:57:33 2026

@author: Angelo Antonio Manzatto
-----------------
Observation and reporting layer for NEAT evolution.

Design
------
Reporter is an abstract base class with optional lifecycle hooks.
Subclass it and override only the hooks you need.

ReporterSet manages a collection of reporters and forwards all
Population events to each one. The Population holds one ReporterSet.

Concrete reporters provided
---------------------------
ConsoleReporter
    Prints a formatted summary to stdout each generation.
    Configurable verbosity: minimal (fitness only) or full (species detail).

StatisticsReporter
    Accumulates GenerationStats history.
    Computes aggregates (best ever, mean over window).
    Exports to CSV for external analysis.

CheckpointReporter
    Saves the best genome to JSON every N generations.
    Also saves on evolution completion.

Usage
-----
    reporters = ReporterSet([
        ConsoleReporter(verbose=True),
        StatisticsReporter(),
        CheckpointReporter(save_dir="checkpoints", every_n=10),
    ])
    pop = Population(..., reporters=reporters)

    # or add after construction:
    pop.reporters.add(MyCustomReporter())
"""

###############################################################################
# libraries
###############################################################################

import csv
import json
import os
from abc import ABC
from pathlib import Path
from typing import List, Optional

from src.neat.genome import Genome
from src.neat.population import GenerationStats, EvolutionResult


###############################################################################
# Abstract base
###############################################################################

class Reporter(ABC):
    """
    Abstract base for NEAT evolution observers.

    All methods have no-op defaults — override only what you need.

    Hook call order per generation
    --------------------------------
    1. on_generation_start(generation)     — before evaluation
    2. on_generation_end(stats, population, best_genome)  — after offspring
    3. on_run_end(result)                  — once, after run() completes

    None of these methods should modify the population or genomes.
    """

    def on_generation_start(self, generation: int) -> None:
        """Called at the start of each generation, before evaluation."""

    def on_generation_end(
        self,
        stats:      GenerationStats,
        population: List[Genome],
        best_genome: Optional[Genome],
    ) -> None:
        """
        Called at the end of each generation, after offspring are generated.

        Parameters
        ----------
        stats : GenerationStats
            Pre-computed statistics for this generation.
        population : List[Genome]
            The NEW offspring population (fitness=None until next step).
        best_genome : Optional[Genome]
            The all-time best genome seen so far (may have been from a
            previous generation). fitness is its raw fitness at the time
            it was recorded.
        """

    def on_run_end(self, result: EvolutionResult) -> None:
        """Called once when run() completes (solved or max_generations hit)."""


###############################################################################
# Reporter set
###############################################################################

class ReporterSet:
    """
    Manages a collection of reporters and forwards events to all of them.

    The Population holds one ReporterSet instance.
    Adding or removing reporters never touches Population code.

    Parameters
    ----------
    reporters : List[Reporter]
        Initial set of reporters. May be empty.
    """

    def __init__(self, reporters: Optional[List[Reporter]] = None) -> None:
        self._reporters: List[Reporter] = list(reporters or [])

    def add(self, reporter: Reporter) -> None:
        """Add a reporter to the set."""
        if not isinstance(reporter, Reporter):
            raise TypeError(
                f"Expected Reporter, got {type(reporter)}"
            )
        self._reporters.append(reporter)

    def remove(self, reporter: Reporter) -> None:
        """Remove a reporter from the set. No-op if not present."""
        try:
            self._reporters.remove(reporter)
        except ValueError:
            pass

    def on_generation_start(self, generation: int) -> None:
        for r in self._reporters:
            r.on_generation_start(generation)

    def on_generation_end(
        self,
        stats:       GenerationStats,
        population:  List[Genome],
        best_genome: Optional[Genome],
    ) -> None:
        for r in self._reporters:
            r.on_generation_end(stats, population, best_genome)

    def on_run_end(self, result: EvolutionResult) -> None:
        for r in self._reporters:
            r.on_run_end(result)

    def __len__(self) -> int:
        return len(self._reporters)

    def __repr__(self) -> str:
        names = [r.__class__.__name__ for r in self._reporters]
        return f"ReporterSet({names})"


###############################################################################
# Console reporter
###############################################################################

class ConsoleReporter(Reporter):
    """
    Prints a formatted summary to stdout each generation.

    Parameters
    ----------
    verbose : bool
        If True, print species detail alongside the generation summary.
        If False, print one line per generation.
    show_every : int
        Print every N generations. 1 = every generation (default).
        0 = never print during evolution, only on run end.
    """

    def __init__(
        self,
        verbose:    bool = False,
        show_every: int  = 1,
    ) -> None:
        self._verbose    = verbose
        self._show_every = max(0, int(show_every))

    def on_generation_end(
        self,
        stats:       GenerationStats,
        population:  List[Genome],
        best_genome: Optional[Genome],
    ) -> None:
        if self._show_every == 0:
            return
        if stats.generation % self._show_every != 0:
            return

        best_f = f"{stats.best_fitness:.5f}" \
                 if stats.best_fitness is not None else "N/A"
        mean_f = f"{stats.mean_fitness:.5f}" \
                 if stats.mean_fitness is not None else "N/A"

        line = (
            f"[Gen {stats.generation:>5}] "
            f"best={best_f}  mean={mean_f}  "
            f"species={stats.n_species}  "
            f"stagnant={stats.n_stagnant}  "
            f"pop={stats.population_size}"
        )
        if stats.solved:
            line += "  *** SOLVED ***"
        print(line)

        if self._verbose:
            self._print_species_detail(population)

    def _print_species_detail(self, population: List[Genome]) -> None:
        from collections import Counter
        species_sizes = Counter(
            g.species_id for g in population
            if g.species_id is not None
        )
        for sid, size in sorted(species_sizes.items()):
            print(f"  Species {sid:>3}: {size} members")

    def on_run_end(self, result: EvolutionResult) -> None:
        status = "SOLVED" if result.solved else "not solved"
        print(
            f"\n=== Evolution complete ({status}) ==="
            f"\n  Generations : {result.generations_run}"
            f"\n  Best fitness: {result.best_fitness}"
            f"\n  Best genome : {result.best_genome}"
        )


###############################################################################
# Statistics reporter
###############################################################################

class StatisticsReporter(Reporter):
    """
    Accumulates GenerationStats and computes aggregate metrics.

    Provides:
    - Full history as List[GenerationStats]
    - Best fitness ever seen
    - Mean fitness over a rolling window
    - CSV export

    Parameters
    ----------
    window : int
        Window size for rolling mean computation. Default 10.
    """

    def __init__(self, window: int = 10) -> None:
        self._window:  int                  = max(1, int(window))
        self._history: List[GenerationStats] = []
        self._best_fitness: Optional[float]  = None
        self._best_generation: Optional[int] = None

    # ── Reporter hooks ────────────────────────────────────────────────────────

    def on_generation_end(
        self,
        stats:       GenerationStats,
        population:  List[Genome],
        best_genome: Optional[Genome],
    ) -> None:
        self._history.append(stats)

        if stats.best_fitness is not None:
            if (self._best_fitness is None or
                    stats.best_fitness > self._best_fitness):
                self._best_fitness    = stats.best_fitness
                self._best_generation = stats.generation

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def history(self) -> List[GenerationStats]:
        """All GenerationStats in order."""
        return list(self._history)

    @property
    def best_fitness_ever(self) -> Optional[float]:
        """Highest fitness seen across all generations."""
        return self._best_fitness

    @property
    def best_fitness_generation(self) -> Optional[int]:
        """Generation index where best_fitness_ever was achieved."""
        return self._best_generation

    @property
    def n_generations(self) -> int:
        """Number of generations recorded."""
        return len(self._history)

    def rolling_mean_fitness(self, n: Optional[int] = None) -> Optional[float]:
        """
        Mean best_fitness over the last n generations.

        Parameters
        ----------
        n : int, optional
            Window size. Defaults to self._window.

        Returns
        -------
        Optional[float]
            Rolling mean, or None if no data.
        """
        n = n or self._window
        recent = [
            s.best_fitness for s in self._history[-n:]
            if s.best_fitness is not None
        ]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def fitness_series(self) -> List[Optional[float]]:
        """Best fitness per generation as a list."""
        return [s.best_fitness for s in self._history]

    def mean_fitness_series(self) -> List[Optional[float]]:
        """Mean fitness per generation as a list."""
        return [s.mean_fitness for s in self._history]

    def species_count_series(self) -> List[int]:
        """Number of species per generation as a list."""
        return [s.n_species for s in self._history]

    # ── Export ────────────────────────────────────────────────────────────────

    def save_csv(self, path: str) -> None:
        """
        Export generation statistics to a CSV file.

        Columns: generation, best_fitness, mean_fitness,
                 n_species, n_stagnant, population_size, solved

        Parameters
        ----------
        path : str
            Output file path. Parent directories are created if needed.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "generation", "best_fitness", "mean_fitness",
                "n_species", "n_stagnant", "population_size", "solved",
            ])
            writer.writeheader()
            for s in self._history:
                writer.writerow({
                    "generation":     s.generation,
                    "best_fitness":   s.best_fitness,
                    "mean_fitness":   s.mean_fitness,
                    "n_species":      s.n_species,
                    "n_stagnant":     s.n_stagnant,
                    "population_size": s.population_size,
                    "solved":         s.solved,
                })

    def __repr__(self) -> str:
        return (
            f"StatisticsReporter("
            f"generations={self.n_generations}, "
            f"best={self._best_fitness})"
        )


###############################################################################
# Checkpoint reporter
###############################################################################

class CheckpointReporter(Reporter):
    """
    Saves the best genome to JSON at regular intervals.

    Saves:
    - Every `every_n` generations (when a new all-time best is found)
    - Always at run end

    File naming: {prefix}_{generation:05d}.json
    The latest checkpoint is also written to {prefix}_latest.json

    Parameters
    ----------
    save_dir : str
        Directory to save checkpoint files. Created if it doesn't exist.
    every_n : int
        Save every N generations. Default 10.
    prefix : str
        Filename prefix. Default "checkpoint".
    save_on_improvement_only : bool
        If True, only save when a new all-time best fitness is achieved.
        If False, save every every_n generations regardless.
        Default True.
    """

    def __init__(
        self,
        save_dir:                str  = "checkpoints",
        every_n:                 int  = 10,
        prefix:                  str  = "checkpoint",
        save_on_improvement_only: bool = True,
    ) -> None:
        self._save_dir   = Path(save_dir)
        self._every_n    = max(1, int(every_n))
        self._prefix     = str(prefix)
        self._improve_only = bool(save_on_improvement_only)
        self._last_saved_fitness: Optional[float] = None

    def on_generation_end(
        self,
        stats:       GenerationStats,
        population:  List[Genome],
        best_genome: Optional[Genome],
    ) -> None:
        if best_genome is None:
            return
        if stats.generation % self._every_n != 0:
            return

        improved = (
            self._last_saved_fitness is None or
            (best_genome.fitness or float('-inf')) > self._last_saved_fitness
        )

        if self._improve_only and not improved:
            return

        self._save(best_genome, stats.generation)
        self._last_saved_fitness = best_genome.fitness

    def on_run_end(self, result: EvolutionResult) -> None:
        if result.best_genome is not None:
            self._save(result.best_genome, result.generations_run,
                       final=True)

    def _save(
        self,
        genome:     Genome,
        generation: int,
        final:      bool = False,
    ) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)

        suffix   = "final" if final else f"{generation:05d}"
        filename = self._save_dir / f"{self._prefix}_{suffix}.json"
        latest   = self._save_dir / f"{self._prefix}_latest.json"

        data = {
            "generation": generation,
            "fitness":    genome.fitness,
            "genome":     genome.to_dict(),
        }

        payload = json.dumps(data, indent=2)
        filename.write_text(payload)
        latest.write_text(payload)

    @staticmethod
    def load(path: str) -> Genome:
        """
        Load a genome from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to a checkpoint JSON file.

        Returns
        -------
        Genome
            The loaded genome with fitness restored.
        """
        from src.neat.genome import Genome
        data = json.loads(Path(path).read_text())
        genome = Genome.from_dict(data["genome"])
        genome.fitness = data.get("fitness")
        return genome

    def __repr__(self) -> str:
        return (
            f"CheckpointReporter("
            f"save_dir={self._save_dir}, "
            f"every_n={self._every_n})"
        )