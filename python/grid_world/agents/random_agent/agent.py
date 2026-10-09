"""
Created on Mon Oct  5 09:07:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from agents.base_agent import BaseAgent

from core.actions import ACTIONS
from engine import Action, EntityObservation

###############################################################################
# Random Agent
###############################################################################

class RandomAgent(BaseAgent):

    name = "random"

    def select_action(self, obs: EntityObservation) -> Action:
        return ACTIONS[self.rng.integers(len(ACTIONS))]