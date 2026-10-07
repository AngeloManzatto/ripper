"""
Created on Sun Oct  4 21:47:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass
from typing import Protocol

from engine import Action, Observation

###############################################################################
# Transition
###############################################################################

@dataclass(frozen=True)
class Transition:
    entity_id: int           # whose reward this is
    obs: Observation         # full observation before the action (all entities)
    action: Action           # this entity's action
    next_obs: Observation    # full observation after the step
    terminated: bool
    truncated: bool

###############################################################################
# Reward Function
###############################################################################

class RewardFn(Protocol):
    def reset(self) -> None:
        """Called at the start of every episode; clear any per-episode state."""
        ...

    def __call__(self, t: Transition) -> float:
        ...