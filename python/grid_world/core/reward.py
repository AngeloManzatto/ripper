"""
Created on Sun Oct  4 21:47:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from typing import Protocol

from core.transition import Transition

###############################################################################
# Transition
###############################################################################

###############################################################################
# Reward Function
###############################################################################

class RewardFn(Protocol):
    def reset(self) -> None:
        """Called at the start of every episode; clear any per-episode state."""
        ...

    def __call__(self, t: Transition) -> float:
        ...