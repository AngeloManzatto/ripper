"""
Created on Fri Nov  6 08:56:30 2026

@author: Angelo Antonio Manzatto

The NEAT learner: one role (player or enemy) that POET can train, with a NEAT population inside.

POET only sees agent(), train() and clone() (poet.learner.Learner). Everything NEAT stays here: the
population, the number of generations of one train() call, the games that score a genome, and how
the champion is chosen.

    TrainConfig     how one train() call evolves the population (experiments/poet/learner_config.py)
    ChampionConfig  how the champion is chosen afterwards
    neat_config     the NEAT library's own settings (a plain dict, see neat.config.default_config)
    
"""

###############################################################################
# libraries
###############################################################################

import copy

import numpy as np

from agents.neat_agent.agent import N_INPUTS, NeatAgent
from core.actions import N_ACTIONS
from neat.evolution import evolve_one_generation, init_population

from experiments.poet.learner_config import ChampionConfig, check_seed_ranges
from experiments.poet.neat_champion import pick_champion
from experiments.poet.neat_scoring import score_population
from poet.entry import OPPONENT_OF

###############################################################################
# Neat Learner
###############################################################################

class NeatLearner:
    """A NEAT population that learns one role, and the champion that plays for it.

    role            : "player" or "enemy_0".
    neat_config     : the NEAT library's settings. Its population_size is replaced by the one of
                      train_config; the dict you pass is not modified.
    train_config    : a TrainConfig. Every train() call evolves train_config.n_generations
                      generations. Generation g is scored on the seeds
                      train_seed + g * n_games ..., and g keeps counting across calls.
    champion_config : a ChampionConfig.
    seed            : makes the first population. The same seed gives the same learner.
    make_agent      : genome -> the agent that plays it (NeatAgent).

    The champion starts as the first genome and changes only when pick_champion finds a genome that
    does better on fresh games. After the last train() call:
        history        : one dict per generation scored, with generation, goal_mean, goal_best,
                         novelty and cells (what the old run_generations table showed)
        champion_rate  : the champion's goal rate in the last choice (None before any training)
    """

    def __init__(self, role, neat_config, train_config, champion_config=None, seed=0, make_agent=NeatAgent):
        if role not in OPPONENT_OF:
            raise ValueError(f"role must be one of {tuple(OPPONENT_OF)}, got {role!r}")

        champion_config = champion_config if champion_config is not None else ChampionConfig()
        check_seed_ranges(train_config, champion_config)

        self.role = role
        self.train_config = train_config
        self.champion_config = champion_config
        self.make_agent = make_agent
        self.neat_config = {**neat_config, "population_size": train_config.population_size}

        self.state = init_population(N_INPUTS, N_ACTIONS, np.random.default_rng(seed), self.neat_config)
        self.champion = self.state["genomes"][0]
        self.champion_rate = None
        self.history = []

    def agent(self):
        """A new agent that plays with the champion. Playing with it never changes the learner."""
        return self.make_agent(self.champion)

    def train(self, layout, opponent, rng):
        """Evolve train_config.n_generations generations on this layout against `opponent`, then
        choose the champion. `rng` (a random.Random) is the only source of randomness.

        The opponent is only played against (every genome meets its own clone of it).
        Raises ValueError, before anything changes, when the generations would reach the training
        seeds into the validation games (see TrainConfig.last_generation). When anything fails
        halfway the learner stays exactly as it was. Returns None.
        """
        train = self.train_config
        first = self.state["generation"]
        last = first + train.n_generations - 1

        if last > train.last_generation():
            raise ValueError(
                f"training would reach generation {last}, but generations above {train.last_generation()} "
                f"would play on the validation seeds (valid_seed = {train.valid_seed}); "
                f"raise valid_seed or lower n_games")

        others = {OPPONENT_OF[self.role]: opponent}
        neat_rng = np.random.default_rng(rng.getrandbits(64))

        state, history = self.state, []
        for _ in range(train.n_generations):
            generation = state["generation"]
            rates, novelty, fitness, cells = score_population(
                state["genomes"], self.role, others, layout, train.n_games,
                train.train_seed + generation * train.n_games,
                train.novelty_weight, train.k_neighbours, make_agent=self.make_agent)

            history.append({
                "generation": generation,
                "goal_mean": sum(rates) / len(rates),
                "goal_best": max(rates),
                "novelty": sum(novelty) / len(novelty),
                "cells": sum(cells) / len(cells),
            })
            state, _ = evolve_one_generation(state, fitness, neat_rng, self.neat_config)

        # The seeds of this choice follow the generation, so no call chooses on the games of another
        first_seed = self.champion_config.champion_seed + state["generation"] * self.champion_config.games_per_round
        champion, rate = pick_champion(
            state["genomes"], self.champion, self.role, others, layout, first_seed,
            train.novelty_weight, train.k_neighbours, self.champion_config, make_agent=self.make_agent)

        self.state, self.champion, self.champion_rate = state, champion, rate
        self.history = self.history + history

    def clone(self):
        """An independent copy: same population, same champion, same history. One deepcopy, so a
        champion that is also in the population stays one object in the copy."""
        return copy.deepcopy(self)