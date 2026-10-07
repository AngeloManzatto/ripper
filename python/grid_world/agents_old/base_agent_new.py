#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 07:34:26 2026

@author: root
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from typing import List
from abc import ABC, abstractmethod

from ripper import Action
from ripper import Observation

###############################################################################
# Base Agent
###############################################################################

class BaseAgent(ABC):
    
    name: str = "base"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def choose_action(
            self, 
            observations: List[Observation],
            last_observation: Observation,
            action_space: List[Action]
            ):
        """
        Given current observation and allowed actions,
        return an Action (possibly with data and reasoning set).
        """
        raise NotImplementedError
        
    def is_done(
            self, 
            observations: list[Observation], 
            curr_obs: Observation
            ) -> bool:
        """ 
        # Agent logic to determine if the game is finished
        """
        return False
    
    def on_transition(self, 
                      curr_obs:Observation, 
                      action:Action, 
                      next_obs:Observation) -> None:
        """
        Execute learning 
        """
        return None

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
        
    @abstractmethod
    def clone(self):
        """Load model."""
        ...