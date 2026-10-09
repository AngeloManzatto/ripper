"""
Created on Sun Oct  4 21:47:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from abc import ABC, abstractmethod

from core.transition import Transition

###############################################################################
# Reward Function
###############################################################################

class RewardFn(ABC):
    
    name: str = "base"
    
    def reset(self) -> None:
        """Called at the start of every episode; clear any per-episode state."""
        ...

    @abstractmethod
    def __call__(self, t: Transition) -> float:
        raise NotImplementedError
        
###############################################################################
# Sparse Reward Function
###############################################################################       
      
class SparseGoalReward(RewardFn):
    
    name: str = "sparse_goal"
    
    def __call__(self, t: Transition) -> float:
        
        if t.next_obs.status == "GoalReached":
            return 1.0
        
        return 0.0