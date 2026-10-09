"""
Created on Fri Oct  9 17:35:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

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
            
    def on_reset(self) -> None:
        self.action_list = self.original_list.copy()
        self.transitions = []
        
    def on_transition(
        self,
        obs: EntityObservation,
        action: Action,
        next_obs: EntityObservation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        
        t = Transition(
            obs,
            action,
            next_obs, 
            terminated,
            truncated
        )
        
        self.transitions.append(t)
        