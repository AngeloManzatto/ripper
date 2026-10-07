"""
Created on Mon Oct  5 08:55:10 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Base agent contract for RIPPER.

An agent is a pure policy:
    - it sees ONLY its own EntityObservation (never the full Observation)
    - it returns an Action
    - it never steps the world, never manages episodes, never runs loops
      (that is the job of the play / evaluate pipeline)

Randomness: every agent owns a seeded rng, separate from the engine's rng,
so agent seed and level seed can vary independently.
"""

###############################################################################
# Libraries
###############################################################################

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from engine import Action, EntityObservation

###############################################################################
# Base Agent
###############################################################################


class BaseAgent(ABC):
    """
    Contract used by every experiment:

        agent.reset()                                   # once per episode
        action = agent.act(obs)                         # once per step
        agent.on_transition(obs, action, next_obs,      # once per step, after
                            terminated, truncated)      # the world has stepped

    Subclasses implement `select_action` (required) and may override
    `on_reset` and `on_transition` (optional).
    """

    name: str = "base"

    def __init__(self, seed: int = 42, history_size: int = 8):
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")

        # Own rng. reset() does NOT reseed it: one agent seed fixes a whole
        # evaluation run, while episodes still differ from each other.
        self.rng = np.random.default_rng(seed)

        # Last observations THIS agent has seen in the current episode.
        # history[-1] is the current obs, history[-2] the previous one, ...
        self.history: deque[EntityObservation] = deque(maxlen=history_size)

    ###########################################################################
    # Episode / step API (called by the pipeline)
    ###########################################################################

    def act(self, obs: EntityObservation) -> Action:
        """Record the observation, then delegate the decision to the subclass."""
        self.history.append(obs)

        action = self.select_action(obs)

        if not isinstance(action, Action):
            raise TypeError(
                f"{type(self).__name__}.select_action must return an Action, "
                f"got {type(action).__name__}"
            )
        return action

    def reset(self) -> None:
        """Called before every episode. Clears per-episode memory."""
        self.history.clear()
        self.on_reset()

    ###########################################################################
    # Hooks for subclasses
    ###########################################################################

    @abstractmethod
    def select_action(self, obs: EntityObservation) -> Action:
        """Choose an action. `self.history[-1]` is `obs`."""
        raise NotImplementedError

    def on_reset(self) -> None:
        """Clear any extra per-episode state (recurrent state, plans, ...)."""
        return None

    def on_transition(
        self,
        obs: EntityObservation,
        action: Action,
        next_obs: EntityObservation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learning hook. Default: do nothing (a pure policy ignores it).

        terminated: the episode ended by a game outcome (goal / trapped / caught).
        truncated:  the episode was cut by the clock. A learner should still
                    bootstrap here, unlike on `terminated`.
        """
        return None

    ###########################################################################
    # Convenience
    ###########################################################################

    @property
    def previous_obs(self) -> EntityObservation | None:
        """The observation before the current one, or None on the first step."""
        if len(self.history) < 2:
            return None
        return self.history[-2]