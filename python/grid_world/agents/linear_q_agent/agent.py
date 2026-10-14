"""
Created on Sat Oct 10 10:41:46 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from agents.base_agent import BaseAgent

from core.actions import ACTIONS, N_ACTIONS
from core.features import features, N_FEATURES
from core.rewards import RewardFn, SparseGoalReward
from core.transition import Transition
from engine import Action, EntityObservation

from agents.linear_q_agent.linear_q import q_values, epsilon_greedy, update

###############################################################################
# Linear Q Agent
###############################################################################

class LinearQAgent(BaseAgent):

    name = "linear_q"

    def __init__(self,
             reward_fn: RewardFn | None = None,
             n_features=N_FEATURES,
             n_actions=N_ACTIONS,
             alpha=0.1,
             gamma=0.95,
             epsilon=0.1,
             eval_epsilon=0.1,
             seed=42,
             history_size=8):
        super().__init__(seed=seed, history_size=history_size)
        
        self.n_features = n_features
        self.n_actions  = n_actions
        
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.eval_epsilon = eval_epsilon

        self.W = np.zeros((n_actions, n_features + 1), dtype="float64")
        
        self.reward_fn = reward_fn if reward_fn is not None else SparseGoalReward()
        
        self.recorder = None
        
    def select_action(self, obs: EntityObservation) -> Action:
        
        # Extract array of features from entity observation
        x = features(obs)
        
        # Calculate q value
        q = q_values(self.W, x)
        
        # Record log
        if self.recorder is not None:
            self.recorder.log_decision(x, q, self.training)
        
        epsilon = self.epsilon if self.training else self.eval_epsilon
        
        # Get action index
        action_idx = epsilon_greedy(q, epsilon=epsilon, rng=self.rng)
            
        return ACTIONS[action_idx]
    
    def on_transition(self, transition: Transition) -> None:

        # Evaluation mode: the agent is frozen
        if not self.training:
            return
        
        # What this transition is worth to this agent
        reward = self.reward_fn(transition)
    
        # The engine's Action -> row index in W
        action_idx = ACTIONS.index(transition.action)
        
        x          = features(transition.obs)
        x_next     = features(transition.next_obs)
        terminated = transition.terminated
        alpha      = self.alpha
        gamma      = self.gamma
        
        if self.recorder is None:
            
            delta = update(
                self.W, 
                x, 
                action_idx, 
                reward, 
                x_next, 
                terminated, 
                alpha, 
                gamma
            )
            
        else:
            
            q_before = q_values(self.W, x)[action_idx]
            
            delta = update(
                self.W, 
                x, 
                action_idx, 
                reward, 
                x_next, 
                terminated, 
                alpha, 
                gamma
            )
            
            q_after = q_values(self.W, x)[action_idx]
            
            self.recorder.log_update(x, delta, q_after - q_before, reward)
        
    def on_reset(self):
        
        self.reward_fn.reset()
        

        