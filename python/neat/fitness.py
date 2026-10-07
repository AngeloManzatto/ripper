"""
Created on Sat Jun  6 19:01:32 2026

@author: Angelo Antonio Manzatto
------------------
Modular fitness evaluation for NEAT.

Architecture
------------

FitnessFunction (abstract)
    The single required contract: evaluate(genome) -> float.
    All other fitness components inherit from this.

DataDrivenFitness (abstract, extends FitnessFunction)
    For fitness functions that evaluate genomes against a dataset.
    Adds dataset management and train/validation split support.
    Subclass this for supervised or self-supervised tasks.

CompositeFitness (concrete, extends FitnessFunction)
    Weighted linear combination of multiple FitnessFunction instances.
    This is the primary composition mechanism:

        fitness = w1 * f1(genome) + w2 * f2(genome) + ...

    Use this to combine reconstruction quality, symbolic agreement,
    novelty, and any other terms without coupling them together.

ThresholdFitness (concrete wrapper)
    Wraps any FitnessFunction and adds a solved() predicate.
    Decouples stopping criterion from fitness computation.

Lifecycle hooks
---------------
Both FitnessFunction and DataDrivenFitness expose optional hooks:
    on_generation_start(generation)
    on_generation_end(generation, genomes)

CompositeFitness forwards these calls to all components automatically.

Usage example (transform detection)
-------------------------------------
    reconstruction = ReconstructionFitness(pairs)      # subclass of DataDrivenFitness
    symbolic_agree = SymbolicAgreementFitness(library)  # subclass of DataDrivenFitness
    novelty        = NoveltyFitness(archive)            # subclass of FitnessFunction

    combined = CompositeFitness([
        (reconstruction, 1.0),
        (symbolic_agree, 0.5),
        (novelty,        0.2),
    ])

    threshold = ThresholdFitness(combined, threshold=0.85)
    # use threshold in Population
"""

###############################################################################
# libraries
###############################################################################

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.neat.genome import Genome
from src.neat.activations import ActivationRegistry, DEFAULT_REGISTRY

###############################################################################
# Base class
###############################################################################

class FitnessFunction(ABC):
    """
    Abstract base class for all NEAT fitness evaluators.

    One method is required: evaluate(genome) -> float.

    Conventions
    -----------
    - Return values are raw scores. Higher is not necessarily better —
      NEATConfig.fitness_maximize controls the direction.
    - Return a finite float always. Use large penalties (-1e6) for
      invalid genomes rather than NaN or inf.
    - Do not modify the genome.
    - evaluate() may be called from multiple threads if the Population
      uses parallel evaluation — implementations must be thread-safe
      or document that they are not.
    """

    @abstractmethod
    def evaluate(self, genome: Genome) -> float:
        """
        Evaluate a single genome.

        Parameters
        ----------
        genome : Genome
            The genome to evaluate. Must not be modified.

        Returns
        -------
        float
            Raw fitness score. Must be finite.
        """
        ...

    def evaluate_population(self, genomes: List[Genome]) -> List[float]:
        """
        Evaluate all genomes and return scores in the same order.

        Default: sequential loop over evaluate().
        Override for parallel evaluation or competitive fitness.

        Parameters
        ----------
        genomes : List[Genome]

        Returns
        -------
        List[float]
            One score per genome, same order as input.
        """
        return [self.evaluate(g) for g in genomes]

    def on_generation_start(self, generation: int) -> None:
        """
        Called before evaluation each generation.

        Override to update data, anneal parameters, or log stats.
        Default implementation does nothing.
        """

    def on_generation_end(
        self,
        generation: int,
        genomes: List[Genome],
    ) -> None:
        """
        Called after fitness is assigned each generation.

        Override to save checkpoints, compute validation metrics,
        or adjust curriculum.
        Default implementation does nothing.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


###############################################################################
# Data-driven base
###############################################################################

class DataDrivenFitness(FitnessFunction, ABC):
    """
    Abstract base for fitness functions that evaluate against a dataset.

    Adds structured dataset management on top of FitnessFunction:
    - Training and optional validation split
    - Thread-safe data replacement via update_data()
    - Activation registry injection

    Subclass this for any supervised or self-supervised fitness task.
    The only required override is _evaluate_on_sample().

    Parameters
    ----------
    data : Sequence of samples
        Training data. Format is task-specific — subclasses define
        what a "sample" is (e.g. (inputs, targets) for regression,
        (mask_a, mask_b) for transform detection).
    validation_data : optional Sequence of samples
        Held-out data for reporting. Never used for fitness computation,
        only for on_generation_end() reporting if implemented.
    registry : ActivationRegistry
        Used to resolve activation functions during forward pass.
    allow_recurrent : bool
        Passed to genome.forward(). Default False.
    penalty : float
        Fitness returned when genome.forward() raises. Default -1e6.
    """

    def __init__(
        self,
        data:             Sequence,
        *,
        validation_data:  Optional[Sequence] = None,
        registry:         ActivationRegistry = DEFAULT_REGISTRY,
        allow_recurrent:  bool               = False,
        penalty:          float              = -1e6,
    ) -> None:
        if not data:
            raise ValueError("data must not be empty")
        self._data            = list(data)
        self._validation_data = list(validation_data) if validation_data else []
        self._registry        = registry
        self._allow_recurrent = allow_recurrent
        self._penalty         = float(penalty)

    # ── Required ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _evaluate_on_sample(
        self,
        genome:  Genome,
        sample,
    ) -> float:
        """
        Compute error/score for one data sample.

        Called once per sample per genome.
        Must return a finite float.
        Must not modify genome.

        Parameters
        ----------
        genome : Genome
        sample : task-specific type matching elements of self._data

        Returns
        -------
        float
            Per-sample score. Sign convention is task-specific —
            document whether higher or lower is better.
        """
        ...

    def _aggregate(self, per_sample_scores: List[float]) -> float:
        """
        Aggregate per-sample scores into a single fitness value.

        Default: mean of all scores.
        Override to use sum, median, worst-case, or other aggregation.

        Parameters
        ----------
        per_sample_scores : List[float]
            One score per training sample.

        Returns
        -------
        float
            Aggregated fitness value.
        """
        if not per_sample_scores:
            return self._penalty
        return sum(per_sample_scores) / len(per_sample_scores)

    # ── FitnessFunction implementation ────────────────────────────────────────

    def evaluate(self, genome: Genome) -> float:
        """
        Evaluate genome against all training samples.

        Returns penalty if any forward pass raises.
        """
        scores: List[float] = []

        for sample in self._data:
            try:
                score = self._evaluate_on_sample(genome, sample)
            except Exception:
                return self._penalty
            scores.append(score)

        return self._aggregate(scores)

    # ── Data management ───────────────────────────────────────────────────────

    def update_data(
        self,
        data:            Sequence,
        validation_data: Optional[Sequence] = None,
    ) -> None:
        """
        Replace training (and optionally validation) data.

        Call from on_generation_start() for incremental or curriculum
        learning. Thread safety: caller must ensure no concurrent
        evaluate() calls during update.

        Parameters
        ----------
        data : non-empty Sequence
        validation_data : optional Sequence
        """
        if not data:
            raise ValueError("data must not be empty")
        self._data = list(data)
        if validation_data is not None:
            self._validation_data = list(validation_data)

    @property
    def n_samples(self) -> int:
        """Number of training samples."""
        return len(self._data)

    @property
    def n_validation_samples(self) -> int:
        """Number of validation samples (0 if none provided)."""
        return len(self._validation_data)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_train={self.n_samples}, "
            f"n_val={self.n_validation_samples})"
        )


###############################################################################
# Composite fitness
###############################################################################

class CompositeFitness(FitnessFunction):
    """
    Weighted linear combination of multiple fitness functions.

        fitness = sum(weight_i * component_i.evaluate(genome))

    This is the primary composition mechanism. Each term is an
    independent FitnessFunction — they can be DataDrivenFitness
    subclasses, novelty measures, or anything else.

    Lifecycle hooks (on_generation_start, on_generation_end) are
    forwarded to all components automatically.

    Parameters
    ----------
    components : Sequence of (FitnessFunction, weight) pairs
        At least one component required. Weights are raw floats —
        they are not normalised automatically.

    Example
    -------
        combined = CompositeFitness([
            (reconstruction_fitness, 1.0),
            (symbolic_agreement,     0.5),
            (novelty_bonus,          0.2),
        ])
    """

    def __init__(
        self,
        components: Sequence[Tuple[FitnessFunction, float]],
    ) -> None:
        if not components:
            raise ValueError("components must not be empty")
        for i, (fn, w) in enumerate(components):
            if not isinstance(fn, FitnessFunction):
                raise TypeError(
                    f"components[{i}][0] must be a FitnessFunction, "
                    f"got {type(fn)}"
                )
            if not isinstance(w, (int, float)) or not np.isfinite(w):
                raise ValueError(
                    f"components[{i}][1] weight must be a finite float, "
                    f"got {w!r}"
                )
        self._components: List[Tuple[FitnessFunction, float]] = list(components)

    # ── FitnessFunction implementation ────────────────────────────────────────

    def evaluate(self, genome: Genome) -> float:
        """
        Compute weighted sum of all component fitness scores.

        Returns sum(weight_i * component_i.evaluate(genome)).
        """
        total = 0.0
        for fn, weight in self._components:
            total += weight * fn.evaluate(genome)
        return total

    def on_generation_start(self, generation: int) -> None:
        for fn, _ in self._components:
            fn.on_generation_start(generation)

    def on_generation_end(
        self,
        generation: int,
        genomes: List[Genome],
    ) -> None:
        for fn, _ in self._components:
            fn.on_generation_end(generation, genomes)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def n_components(self) -> int:
        return len(self._components)

    def weights(self) -> List[float]:
        """Return list of weights in component order."""
        return [w for _, w in self._components]

    def component_scores(self, genome: Genome) -> List[float]:
        """
        Return individual component scores before weighting.

        Useful for diagnostics — see which term dominates.
        """
        return [fn.evaluate(genome) for fn, _ in self._components]

    def __repr__(self) -> str:
        parts = [
            f"{fn.__class__.__name__}*{w}"
            for fn, w in self._components
        ]
        return f"CompositeFitness([{', '.join(parts)}])"


###############################################################################
# Threshold wrapper
###############################################################################

class ThresholdFitness:
    """
    Wraps a FitnessFunction and adds a solved() predicate.

    Decouples the stopping criterion from fitness computation.
    The Population uses solved() to decide when to stop early.

    This is not a FitnessFunction subclass — it is a wrapper that
    delegates all FitnessFunction calls to the inner function while
    adding the threshold check.

    Parameters
    ----------
    fitness_fn : FitnessFunction
        The fitness function to wrap.
    threshold : float
        A genome is "solved" when evaluate(genome) >= threshold
        (or <= threshold if maximize=False).
    maximize : bool
        If True (default), solved when score >= threshold.
        If False, solved when score <= threshold.
        Should match NEATConfig.fitness_maximize.

    Example
    -------
        combined = CompositeFitness([...])
        stopping = ThresholdFitness(combined, threshold=0.90)

        # In the evolution loop:
        if stopping.solved(best_genome):
            break
    """

    def __init__(
        self,
        fitness_fn: FitnessFunction,
        threshold:  float,
        maximize:   bool = True,
    ) -> None:
        if not isinstance(fitness_fn, FitnessFunction):
            raise TypeError(
                f"fitness_fn must be a FitnessFunction, "
                f"got {type(fitness_fn)}"
            )
        if not np.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold!r}")
        self._fn        = fitness_fn
        self._threshold = float(threshold)
        self._maximize  = bool(maximize)

    # ── Delegate FitnessFunction interface ────────────────────────────────────

    def evaluate(self, genome: Genome) -> float:
        return self._fn.evaluate(genome)

    def evaluate_population(self, genomes: List[Genome]) -> List[float]:
        return self._fn.evaluate_population(genomes)

    def on_generation_start(self, generation: int) -> None:
        self._fn.on_generation_start(generation)

    def on_generation_end(
        self,
        generation: int,
        genomes: List[Genome],
    ) -> None:
        self._fn.on_generation_end(generation, genomes)

    # ── Threshold check ───────────────────────────────────────────────────────

    def solved(self, genome: Genome) -> bool:
        """
        Return True if genome meets the fitness threshold.

        Parameters
        ----------
        genome : Genome
            The genome to check.

        Returns
        -------
        bool
            True if the genome is considered solved.
        """
        score = self.evaluate(genome)
        if self._maximize:
            return score >= self._threshold
        return score <= self._threshold

    def solved_any(self, genomes: List[Genome]) -> bool:
        """Return True if any genome in the list is solved."""
        return any(self.solved(g) for g in genomes)

    def best_solved(self, genomes: List[Genome]) -> Optional[Genome]:
        """
        Return the best solved genome, or None if none are solved.

        Parameters
        ----------
        genomes : List[Genome]
            Genomes with fitness already assigned.

        Returns
        -------
        Optional[Genome]
            The solved genome with the best fitness, or None.
        """
        solved = [g for g in genomes if self.solved(g)]
        if not solved:
            return None
        return max(solved, key=lambda g: g.fitness or float('-inf'))

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def maximize(self) -> bool:
        return self._maximize

    def __repr__(self) -> str:
        direction = ">=" if self._maximize else "<="
        return (
            f"ThresholdFitness("
            f"{self._fn!r}, "
            f"threshold{direction}{self._threshold})"
        )