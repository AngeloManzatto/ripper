"""
Created on Fri Oct  9 10:34:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass

from engine import Action, EntityObservation

###############################################################################
# libraries
###############################################################################

@dataclass(frozen=True)
class Transition:
    obs: EntityObservation       # this entity's view before the action
    action: Action
    next_obs: EntityObservation  # this entity's view after the step
    terminated: bool
    truncated: bool