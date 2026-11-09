"""
Created on Mon Oct  5 08:55:10 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import copy
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from core.actions import ACTIONS
from engine import Action, EntityObservation
from core.transition import Transition

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

    Persistence and copies, also part of the contract:

        agent.save(path)       # write what the agent has learned to a file
        agent.load(path)       # read it back into this agent
        other = agent.clone()  # an independent copy that plays the same way

    Every agent also records its own episode, whatever it is:

        agent.positions   # the cell it stood on at each decision of the episode, in order
        agent.actions     # the action it chose at each of those decisions, as an index into
                          # core.actions.ACTIONS (a plain int: engine actions cannot be pickled)

    Both are emptied by `reset()`, so after an episode they hold that episode until the
    next one starts. Read them, do not change them.

    An agent with something to store overrides `save` and `load`. The defaults
    raise NotImplementedError, so asking an agent that cannot do it fails loudly.
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
        
        # Training / Evaluation flag
        self.training = True

        # Record of the current episode: one entry per decision, never truncated
        self.positions: list[tuple] = []
        self.actions: list[int] = []

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

        # Recorded together and only after a valid action, so the lists stay aligned.
        # The action is stored as its index in ACTIONS: ints and tuples can be pickled and
        # saved, the engine's Action objects cannot. ValueError if it is not one of ACTIONS.
        index = ACTIONS.index(action)
        self.positions.append(tuple(obs.position))
        self.actions.append(index)

        return action

    def reset(self) -> None:
        """Called before every episode. Clears per-episode memory."""
        self.history.clear()
        self.positions.clear()
        self.actions.clear()
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
        t: Transition
    ) -> None:
        """
        Learning hook. Default: do nothing (a pure policy ignores it).

        t.terminated: the episode ended by a game outcome (goal / trapped / caught).
        t.truncated:  the episode was cut by the clock. A learner should still
                      bootstrap here, unlike on `terminated`.
        """
        return None

    ###########################################################################
    # Persistence and copies
    ###########################################################################

    def save(self, path) -> None:
        """Write what the agent has learned (its policy) to `path`."""
        raise NotImplementedError(f"{type(self).__name__} does not implement save")

    def load(self, path) -> None:
        """Read what `save` wrote into this agent. A failed load changes nothing."""
        raise NotImplementedError(f"{type(self).__name__} does not implement load")

    def clone(self) -> "BaseAgent":
        """An independent copy: same policy, same mode, same rng state, shares
        nothing with this agent, and starts with a clean episode (empty history).
        Learners that must not copy everything (replay buffer, optimizer) override it."""
        empty_history = deque(maxlen=self.history.maxlen)
        other = copy.deepcopy(self, {id(self.history): empty_history})
        other.reset()
        return other

    ###########################################################################
    # Convenience
    ###########################################################################

    @property
    def previous_obs(self) -> EntityObservation | None:
        """The observation before the current one, or None on the first step."""
        if len(self.history) < 2:
            return None
        return self.history[-2]
    
    ###########################################################################
    # Train / Eval switch
    ###########################################################################
    
    def set_eval_mode(self):
        """Switch to evaluation behavior."""
        ...
        self.training = False

    def set_train_mode(self):
        """Switch to training behavior."""
        ...
        self.training = True