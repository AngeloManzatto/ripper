"""
Created on Fri Oct  9 17:35:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import copy

from agents.base_agent import BaseAgent
from core.transition import Transition

from engine import Action, EntityObservation

###############################################################################
# Scripted Agent
###############################################################################

class ScriptedAgent(BaseAgent):

    name = "scripted"
    
    def __init__(self, action_list, seed: int = 42, history_size: int = 8):
        super().__init__(seed=seed, history_size=history_size)
        
        self.action_list   = list(action_list)
        self.original_list = list(action_list)
        self.transitions = []
        
    def select_action(self, obs: EntityObservation) -> Action:
        
        # Pop the first action from the list
        return self.action_list.pop(0)
            
    def clone(self) -> "ScriptedAgent":
        # The script holds engine Actions, which cannot be deep-copied: the copy gets a new
        # list with the same (immutable) actions. Transitions belong to the episode, not the copy.
        other = ScriptedAgent(self.original_list, history_size=self.history.maxlen)
        other.rng = copy.deepcopy(self.rng)
        other.training = self.training
        return other

    def on_reset(self) -> None:
        self.action_list = self.original_list.copy()
        self.transitions = []
        
    def on_transition(
        self,
        t: Transition
    ) -> None:
        
        self.transitions.append(t)