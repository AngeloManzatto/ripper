"""
Created on Sun Oct  4 21:47:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from abc import ABC, abstractmethod

from core.transition import Transition

from core.potentials import explore_potential, goal_potential

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
    
###############################################################################
# Shaped Goal Reward Function
###############################################################################    
    
class ShapedGoalReward(RewardFn):
    
    name: str = "shaped_goal"
    
    def __init__(self, gamma=0.95, w_goal=1.0, w_explore=0.5, w_bump=0.1):
        
        self.gamma     = gamma
        self.w_goal    = w_goal
        self.w_explore = w_explore
        self.w_bump    = w_bump
        
    def __call__(self, t: Transition) -> float:

        reward = 0.0
        
        moved = t.obs.position != t.next_obs.position
        if not moved and not t.terminated:
            reward -= self.w_bump

        if t.next_obs.status == "GoalReached":
            reward = 1.0
    
        explore_before = explore_potential(t.obs.grid)
        explore_after  = explore_potential(t.next_obs.grid)
    
        goal_before = goal_potential(t.obs.grid, t.obs.position)

        if t.next_obs.status == "GoalReached":
            goal_after = 1.0                 # standing on the goal; the grid may hide it
        else:
            goal_after = goal_potential(t.next_obs.grid, t.next_obs.position)
            
        reward += (
            self.w_goal    * (self.gamma * goal_after    - goal_before) +
            self.w_explore * (self.gamma * explore_after - explore_before)
        )
    
        return float(reward)