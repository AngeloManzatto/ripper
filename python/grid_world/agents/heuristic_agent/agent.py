#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:03:45 2026

@author: root
"""

"""
Heuristic agent: a hand-written policy on top of core.features.

  1. Never step into a wall, a trap or a hostile neighbor (when avoidable).
  2. Goal known -> move along the directions that reduce the distance to it.
  3. Goal unknown -> explore: head for the half of the map with the most
     unexplored cells, without immediately undoing the previous move.
  4. With probability `epsilon` take a random safe action instead, which also
     breaks the deadlocks a purely greedy rule can fall into (e.g. a wall
     between the agent and the goal).

It is the first non-random row of the benchmark: it shows whether the
features carry useful signal, and it is the bar a learned agent has to beat.
Written for the PLAYER (goal-seeking); an enemy observer would need its own rules.
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from agents.base_agent import BaseAgent
from agents.random_agent import ACTIONS
from core.features import DIRECTION_NAMES, FEATURE_NAMES, features
from engine import Action, EntityObservation

###############################################################################
# Globals
###############################################################################

# Indices follow ACTIONS / DIRECTION_NAMES: 0 up, 1 down, 2 left, 3 right
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

###############################################################################
# Heuristic Agent
###############################################################################


class HeuristicAgent(BaseAgent):

    name = "heuristic"

    def __init__(self, seed: int = 42, history_size: int = 8, epsilon: float = 0.1):
        super().__init__(seed=seed, history_size=history_size)
        self.epsilon = epsilon
        self.last = None          # index of the previous action

    def on_reset(self) -> None:
        self.last = None

    # ---- helpers ---------------------------------------------------------

    def _safe_directions(self, f: dict) -> list[int]:
        """Directions whose neighbor is not a wall, trap or hostile."""
        safe = [
            i for i, d in enumerate(DIRECTION_NAMES)
            if not (f[f"{d}_wall"] or f[f"{d}_trap"] or f[f"{d}_hostile"])
        ]
        if safe:
            return safe
        # boxed in: anything that is not a wall beats standing still
        not_wall = [i for i, d in enumerate(DIRECTION_NAMES) if not f[f"{d}_wall"]]
        return not_wall or list(range(len(ACTIONS)))

    def _ranked_directions(self, f: dict) -> list[int]:
        """Directions worth taking, best first (may be empty)."""
        if f["goal_found"]:
            drow, dcol = f["goal_drow"], f["goal_dcol"]
            scores = np.array([-drow, drow, -dcol, dcol])        # up, down, left, right
        else:
            scores = np.array([f[f"unknown_{d}"] for d in DIRECTION_NAMES])

        # tiny noise breaks ties randomly (e.g. two equally unexplored halves)
        order = np.argsort(-(scores + 1e-6 * self.rng.random(len(scores))))
        return [int(i) for i in order if scores[i] > 0]

    def select_action(self, obs: EntityObservation) -> Action:
        
        # Feature from observation as named dict
        f = dict(zip(FEATURE_NAMES, features(obs).tolist()))
        
        # Get safe actions that does not move into WALL, TRAP , HOSTILE ENTITY
        safe = self._safe_directions(f)

        # Exploratory chance
        if self.rng.random() < self.epsilon:
            choice = int(self.rng.choice(safe))
        else:
            candidates = [i for i in self._ranked_directions(f) if i in safe]

            # exploring: do not walk straight back where we just came from
            if not f["goal_found"] and self.last is not None:
                forward = [i for i in candidates if i != OPPOSITE[self.last]]
                candidates = forward or candidates

            choice = candidates[0] if candidates else int(self.rng.choice(safe))

        self.last = choice
        return ACTIONS[choice]