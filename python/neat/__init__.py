"""
Created on Sat Jun  6 19:55:55 2026

@author: Angelo Antonio Manzatto
-----
A modular, professional NEAT implementation.

Quick start
-----------
    from neat import (
        NEATConfig,
        Population,
        ThresholdFitness,
        ConsoleReporter,
        StatisticsReporter,
        CheckpointReporter,
        ReporterSet,
    )

Modules
-------
    config      — NEATConfig frozen dataclass
    activations — ActivationRegistry and DEFAULT_REGISTRY
    genes       — NodeGene, ConnectionGene, NodeType
    innovation  — InnovationRegistry
    genome      — Genome, make_minimal_genome
    fitness     — FitnessFunction, DataDrivenFitness,
                  CompositeFitness, ThresholdFitness
    species     — Species and speciation functions
    population  — Population, GenerationStats, EvolutionResult
    reporters   — Reporter, ReporterSet, ConsoleReporter,
                  StatisticsReporter, CheckpointReporter
"""
###############################################################################
# libraries
###############################################################################

from src.neat.config      import NEATConfig
from src.neat.activations import ActivationRegistry, DEFAULT_REGISTRY, ActivationNotFoundError
from src.neat.genes       import NodeGene, ConnectionGene, NodeType
from src.neat.innovation  import InnovationRegistry
from src.neat.genome      import Genome, make_minimal_genome
from src.neat.fitness     import (
    FitnessFunction,
    DataDrivenFitness,
    CompositeFitness,
    ThresholdFitness,
)
from src.neat.species     import Species
from src.neat.population  import Population, GenerationStats, EvolutionResult
from src.neat.reporters   import (
    Reporter,
    ReporterSet,
    ConsoleReporter,
    StatisticsReporter,
    CheckpointReporter,
)

__all__ = [
    "NEATConfig",
    "ActivationRegistry", "DEFAULT_REGISTRY", "ActivationNotFoundError",
    "NodeGene", "ConnectionGene", "NodeType",
    "InnovationRegistry",
    "Genome", "make_minimal_genome",
    "FitnessFunction", "DataDrivenFitness", "CompositeFitness", "ThresholdFitness",
    "Species",
    "Population", "GenerationStats", "EvolutionResult",
    "Reporter", "ReporterSet",
    "ConsoleReporter", "StatisticsReporter", "CheckpointReporter",
]