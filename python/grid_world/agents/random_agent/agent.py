"""
Created on Mon Oct  5 09:07:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""
Random agent: uniform over the four actions, ignores the observation.

Baseline for every experiment: any learned or evolved agent has to beat this.
"""

###############################################################################
# Libraries
###############################################################################

from agents.base_agent import BaseAgent
from engine import Action, EntityObservation

###############################################################################
# Random Agent
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]


class RandomAgent(BaseAgent):

    name = "random"

    def select_action(self, obs: EntityObservation) -> Action:
        return ACTIONS[self.rng.integers(len(ACTIONS))]