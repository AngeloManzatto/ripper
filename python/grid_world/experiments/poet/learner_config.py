"""
Created on Thu Nov  5 06:30:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Settings of the NEAT learner (experiments/poet/neat_learner.py).

Two small frozen dataclasses, one per job. The NEAT library's own settings (mutation rates,
speciation, ...) stay in a plain dict that its check_config validates.

    TrainConfig    : how one role evolves in one call of train()
    ChampionConfig : how the champion is chosen after that

Seeds: three ranges of games must never overlap, or a genome would be judged on games it has seen:
    training games    train_seed ...              moving forward with the generation
    validation games  valid_seed ... + n_valid_games
    champion games    champion_seed ...           moving forward with the generation
check_seed_ranges checks the fixed part of that. The training range grows with the generations, so
TrainConfig.last_generation tells how far it may go before it would reach valid_seed.
"""

from dataclasses import dataclass


def _at_least(name, value, minimum):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")

###############################################################################
# Train Config
###############################################################################

@dataclass(frozen=True)
class TrainConfig:
    """population_size : genomes in the population.
    n_generations   : generations of evolution in ONE call of train().
    n_games         : games that score each genome in one generation.
    n_valid_games   : games of the validation duel (never trained on).
    train_seed      : first seed of the training games. Generation g plays the seeds
                      train_seed + g * n_games ... + n_games - 1.
    valid_seed      : first seed of the validation games.
    novelty_weight  : share of novelty in the fitness, 0 = only the goal rate.
    k_neighbours    : neighbours that measure novelty (fewer than population_size).
    report_every    : every this many generations the table row gets the validation duel;
                      None = never (no validation games are played).
    """
    population_size: int
    n_generations: int
    n_games: int
    n_valid_games: int
    train_seed: int = 200_000
    valid_seed: int = 210_000
    novelty_weight: float = 0.2
    k_neighbours: int = 10
    report_every: int | None = 10

    def __post_init__(self):
        _at_least("population_size", self.population_size, 2)
        _at_least("n_generations", self.n_generations, 1)
        _at_least("n_games", self.n_games, 1)
        _at_least("n_valid_games", self.n_valid_games, 1)
        _at_least("train_seed", self.train_seed, 0)
        _at_least("valid_seed", self.valid_seed, 0)
        _at_least("k_neighbours", self.k_neighbours, 1)

        if self.report_every is not None:
            _at_least("report_every", self.report_every, 1)
        if not 0 <= self.novelty_weight <= 1:
            raise ValueError(f"novelty_weight must be in [0, 1], got {self.novelty_weight}")
        if self.k_neighbours >= self.population_size:
            raise ValueError(f"k_neighbours ({self.k_neighbours}) must be smaller than "
                             f"population_size ({self.population_size})")
        if self.valid_seed < self.train_seed + self.n_games:
            raise ValueError("valid_seed must come after the first generation's training games: "
                             f"train_seed + n_games = {self.train_seed + self.n_games}, valid_seed = {self.valid_seed}")

    def last_generation(self):
        """The highest generation whose training games still stay below valid_seed."""
        return (self.valid_seed - self.train_seed) // self.n_games - 1

###############################################################################
# Champion Config
###############################################################################

@dataclass(frozen=True)
class ChampionConfig:
    """How the champion of a population is chosen (see pick_champion).

    champion_seed  : first seed of the games that choose a champion; every round moves forward.
    n_finalists    : genomes (besides the current champion) that reach the second stage.
    n_score_games  : games of the first stage, to find the finalists.
    n_final_games  : games of the second stage, to pick the champion among the finalists.
    """
    champion_seed: int = 240_000
    n_finalists: int = 5
    n_score_games: int = 20
    n_final_games: int = 100

    def __post_init__(self):
        _at_least("champion_seed", self.champion_seed, 0)
        _at_least("n_finalists", self.n_finalists, 1)
        _at_least("n_score_games", self.n_score_games, 1)
        _at_least("n_final_games", self.n_final_games, 1)

    @property
    def games_per_round(self):
        """Seeds one round of choosing uses (both stages)."""
        return self.n_score_games + self.n_final_games


def check_seed_ranges(train, champion):
    """The champion games must start after the validation games. Raises ValueError otherwise."""
    valid_end = train.valid_seed + train.n_valid_games
    if champion.champion_seed < valid_end:
        raise ValueError(f"champion_seed ({champion.champion_seed}) is inside the validation games "
                         f"{train.valid_seed} ... {valid_end - 1}")