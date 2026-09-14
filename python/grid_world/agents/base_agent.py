"""
Created on Sun Sep 13 16:34:02 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from abc import ABC, abstractmethod

###############################################################################
# Base Agent
###############################################################################

class BaseAgent(ABC):
    @abstractmethod
    def act(self, observation):
        """Given an observation, return an action index."""
        ...

    @abstractmethod
    def train(self, layout, episodes, **kwargs):
        """Train on the given layout for the given number of episodes."""
        ...
        
    @abstractmethod
    def set_eval_mode(self):
        """Switch to deterministic/evaluation behavior."""
        ...

    @abstractmethod
    def set_train_mode(self):
        """Switch back to exploratory/training behavior."""
        ...
        
    @abstractmethod
    def save(self, path):
        """SSave model."""
        ...
        
    @abstractmethod
    def load(self, path):
        """Load model."""
        ...
        
        