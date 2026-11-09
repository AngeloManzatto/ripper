"""
Created on Sun Oct  4 21:47:15 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from abc import ABC, abstractmethod
from collections import deque

from core.transition import Transition
from core.potentials import explore_potential, goal_potential

DEADLY = ("Caught", "Trapped")

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

    def __init__(self, 
                 gamma=0.95, 
                 w_goal=1.0, 
                 w_explore=0.5,
                 w_danger=0.0,
                 w_bump=0.0, 
                 w_revisit=0.0, 
                 window=8):

        if window < 1:
            raise ValueError("window must be at least 1")

        self.gamma     = gamma
        self.w_goal    = w_goal
        self.w_explore = w_explore
        self.w_danger  = w_danger
        self.w_bump    = w_bump
        self.w_revisit = w_revisit
        self.window    = window

        # The agent sees `window` positions including the one it stands on,
        # so the reward remembers only the `window - 1` before it.
        self.recent = deque(maxlen=window - 1)

    def __call__(self, t: Transition) -> float:

        reward = 0.0

        reached = t.next_obs.status == "GoalReached"

        # Bump: tried to move, stayed in place, episode not over
        moved = t.obs.position != t.next_obs.position
        if not moved and not t.terminated:
            reward -= self.w_bump

        # Goal bonus
        if reached:
            reward += 1.0
            
        if t.next_obs.status in DEADLY:
            reward -= self.w_danger

        # Potential-based shaping
        explore_before = explore_potential(t.obs.grid)
        explore_after  = explore_potential(t.next_obs.grid)

        goal_before = goal_potential(t.obs.grid, t.obs.position)
        if reached:
            goal_after = 1.0     # standing on the goal; the grid may hide it
        else:
            goal_after = goal_potential(t.next_obs.grid, t.next_obs.position)

        reward += (
            self.w_goal    * (self.gamma * goal_after    - goal_before) +
            self.w_explore * (self.gamma * explore_after - explore_before)
        )

        # Revisit: stepped onto a cell the agent can still "see" in its memory
        current = t.obs.position
        target  = t.next_obs.position
        if target != current and target in self.recent:
            reward -= self.w_revisit
        self.recent.append(current)

        return float(reward)

    def reset(self) -> None:
        self.recent.clear()