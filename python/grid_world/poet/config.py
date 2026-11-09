"""
Created on Mon Nov  2 07:33:56 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass, field
 
from neat.config import default_config

###############################################################################
# Validation
###############################################################################
 
def _check(condition, message):
    if not condition:
        raise ValueError(message)
 
###############################################################################
# Train Config
###############################################################################
 
@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """How one population evolves (see run_generations).
 
    neat            : the NEAT settings (neat.config.default_config()); population_size below
                      replaces the one inside, see neat_config().
    population_size : genomes per population.
    n_generations   : generations in one call (one round of optimisation, per role).
    n_games         : games per genome per generation. Generation g plays the seeds
                      train_seed + g * n_games ...
    train_seed      : first seed of the training games.
    valid_seed      : first seed of the validation games (never trained on).
    n_valid_games   : games of the duel the table reports for the best genome.
    novelty_weight  : share of the fitness that comes from novelty, in [0, 1].
    k_neighbours    : neighbours the novelty score compares with.
    report_every    : a table row with the duel every this many generations.
    """
    neat: dict = field(default_factory=default_config)
    population_size: int = 50
    n_generations: int = 10
    n_games: int = 20
    train_seed: int = 200_000
    valid_seed: int = 210_000
    n_valid_games: int = 50
    novelty_weight: float = 0.2
    k_neighbours: int = 10
    report_every: int = 10
 
    def __post_init__(self):
        _check(isinstance(self.neat, dict), f"neat must be a dict, got {type(self.neat).__name__}")
        _check(self.population_size >= 2, f"population_size must be >= 2, got {self.population_size}")
        _check(self.n_generations >= 0, f"n_generations must be >= 0, got {self.n_generations}")
        _check(self.n_games >= 1, f"n_games must be >= 1, got {self.n_games}")
        _check(self.n_valid_games >= 1, f"n_valid_games must be >= 1, got {self.n_valid_games}")
        _check(self.train_seed >= 0 and self.valid_seed >= 0, "seeds must be >= 0")
        _check(self.train_seed != self.valid_seed, "train_seed and valid_seed must differ")
        _check(0.0 <= self.novelty_weight <= 1.0, f"novelty_weight must be in [0, 1], got {self.novelty_weight}")
        _check(self.k_neighbours >= 1, f"k_neighbours must be >= 1, got {self.k_neighbours}")
        _check(self.report_every >= 1, f"report_every must be >= 1, got {self.report_every}")
 
    def neat_config(self):
        """A copy of the NEAT settings with this population_size. self.neat is not modified."""
        return {**self.neat, "population_size": self.population_size}

###############################################################################
# Champion Config
###############################################################################
 
@dataclass(frozen=True, kw_only=True)
class ChampionConfig:
    """How the champion of a role is chosen after a round (see pick_champion).
 
    seed          : first seed of the games used to choose; later rounds move on from it.
    n_finalists   : genomes of the population that play the final, besides the old champion.
    n_score_games : games of stage 1 (the population is scored, the best become finalists).
    n_final_games : games of stage 2 (the finalists are measured again, fresh games).
    """
    seed: int = 240_000
    n_finalists: int = 5
    n_score_games: int = 20
    n_final_games: int = 100
 
    def __post_init__(self):
        _check(self.seed >= 0, f"seed must be >= 0, got {self.seed}")
        _check(self.n_finalists >= 1, f"n_finalists must be >= 1, got {self.n_finalists}")
        _check(self.n_score_games >= 1, f"n_score_games must be >= 1, got {self.n_score_games}")
        _check(self.n_final_games >= 1, f"n_final_games must be >= 1, got {self.n_final_games}")
 
 
###############################################################################
# Minimal Criterion Config
###############################################################################
 
@dataclass(frozen=True, kw_only=True)
class CriterionConfig:
    """The minimal criterion and the games it is checked on (see passes_minimal_criterion).
 
    low, high   : both roles must reach their goal in a share of the games inside [low, high].
    timeout_cap : at most this share of the games may run out of time.
    n_games     : games of every check (eligibility of a parent, acceptance of a child).
    first_seed  : first seed of those games; training never uses them.
    """
    low: float = 0.2
    high: float = 0.7
    timeout_cap: float = 0.3
    n_games: int = 100
    first_seed: int = 230_000
 
    def __post_init__(self):
        _check(0.0 <= self.low <= self.high <= 1.0,
               f"need 0 <= low <= high <= 1, got low={self.low}, high={self.high}")
        _check(0.0 <= self.timeout_cap <= 1.0, f"timeout_cap must be in [0, 1], got {self.timeout_cap}")
        _check(self.n_games >= 1, f"n_games must be >= 1, got {self.n_games}")
        _check(self.first_seed >= 0, f"first_seed must be >= 0, got {self.first_seed}")
 
 
###############################################################################
# Reproduction Config
###############################################################################
 
@dataclass(frozen=True, kw_only=True)
class ReproductionConfig:
    """How the pool makes children (see run_poet).
 
    mutation   : the engine's MutationConfig settings, as a dict (without the seed).
    n_mutate   : reproduce every this many iterations.
    n_children : attempts per eligible parent.
    max_new    : children admitted to the pool per reproduction step, at most.
    max_size   : the pool never holds more entries than this; the oldest leave first.
    """
    mutation: dict
    n_mutate: int = 1
    n_children: int = 5
    max_new: int = 2
    max_size: int = 4
 
    def __post_init__(self):
        _check(isinstance(self.mutation, dict), f"mutation must be a dict, got {type(self.mutation).__name__}")
        _check("seed" not in self.mutation, "mutation must not hold a seed: every attempt gets its own")
        _check(self.n_mutate >= 1, f"n_mutate must be >= 1, got {self.n_mutate}")
        _check(self.n_children >= 1, f"n_children must be >= 1, got {self.n_children}")
        _check(self.max_new >= 1, f"max_new must be >= 1, got {self.max_new}")
        _check(self.max_size >= 1, f"max_size must be >= 1, got {self.max_size}")
 
 
###############################################################################
# Poet Config
###############################################################################
 
@dataclass(frozen=True, kw_only=True)
class PoetConfig:
    """The four groups of settings of one POET run."""
    reproduction: ReproductionConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    champion: ChampionConfig = field(default_factory=ChampionConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
 
    def __post_init__(self):
        for name, kind in (("train", TrainConfig), ("champion", ChampionConfig),
                           ("criterion", CriterionConfig), ("reproduction", ReproductionConfig)):
            _check(isinstance(getattr(self, name), kind),
                   f"{name} must be a {kind.__name__}, got {type(getattr(self, name)).__name__}")