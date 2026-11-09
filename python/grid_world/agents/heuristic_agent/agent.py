"""
Created on Thu Oct  8 13:03:45 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from agents.base_agent import BaseAgent
from agents.features.frame import FEATURE_NAMES, features
from core.actions import ACTIONS, OPPOSITE, DIRECTION_NAMES
from engine import Action, EntityObservation

###############################################################################
# Directions utils based on features
###############################################################################

def safe_directions(f: dict):
    """Directions whose neighbor is not a wall, trap or hostile."""
    
    # Avoid any action idx that steps into wall, trap or hostile
    all_action_indices   = list(range(len(ACTIONS)))
    safe_hazard_indices = []
    safe_wall_indices    = []
    
    for action_idx, direction in enumerate(DIRECTION_NAMES):
        
        if not (f[f"{direction}_wall"] or 
                f[f"{direction}_trap"] or 
                f[f"{direction}_hostile"]):
            
            safe_hazard_indices.append(action_idx)
            
        if not (f[f"{direction}_wall"]):
            safe_wall_indices.append(action_idx)
            
    if safe_hazard_indices:
        return safe_hazard_indices
    
    return all_action_indices

def ranked_directions(f: dict, rng):
    """Directions worth taking, best first (may be empty)."""
    
    if f["goal_found"]:
        drow, dcol = f["goal_drow"], f["goal_dcol"]
        scores = np.array([-drow, drow, -dcol, dcol])        # up, down, left, right
    else:
        scores = np.array([f[f"unknown_{d}"] for d in DIRECTION_NAMES])
        
    # tiny noise breaks ties randomly (e.g. two equally unexplored halves)
    order = np.argsort(-(scores + 1e-6 * rng.random(len(scores))))
    
    return [int(i) for i in order if scores[i] > 0]

###############################################################################
# Heuristic Agent
###############################################################################
        
class HeuristicAgent(BaseAgent):
    
    """
    Heuristic agent: a hand-written policy on top of core.features.
    
      1. Never step into a wall, a trap or a hostile neighbor (when avoidable).
      2. Goal known -> move along the directions that reduce the distance to it.
      3. Goal unknown -> explore: head for the half of the map with the most
         unexplored cells, without immediately undoing the previous move.
      4. With probability `epsilon` take a random safe action instead, which also
         breaks the deadlocks a purely greedy rule can fall into (e.g. a wall
         between the agent and the goal).
    
    It is the first non-random row of the benchmark: it shows whether the
    features carry useful signal, and it is the bar a learned agent has to beat.
    Written for the PLAYER (goal-seeking); an enemy observer would need its own rules.
    """

    name = "heuristic"

    def __init__(self, 
                 seed: int = 42, 
                 history_size: int = 8, 
                 epsilon: float = 0.1
                 ):
        super().__init__(seed=seed, history_size=history_size)
        self.epsilon      = epsilon
        self.last = None          # index of the previous action

    def select_action(self, obs: EntityObservation) -> Action:
        
        # Feature from observation as named dict
        f = dict(zip(FEATURE_NAMES, features(obs).tolist()))
        
        # Get safe actions that does not move into WALL, TRAP , HOSTILE ENTITY
        safe = safe_directions(f)
        
        # Exploratory chance
        if self.rng.random() < self.epsilon:
            choice = int(self.rng.choice(safe))
        else:
            candidates = [i for i in ranked_directions(f, self.rng) if i in safe]

            # exploring: do not walk straight back where we just came from
            if not f["goal_found"] and self.last is not None:
                forward = [i for i in candidates if i != OPPOSITE[self.last]]
                candidates = forward or candidates

            choice = candidates[0] if candidates else int(self.rng.choice(safe))

        self.last = choice
        return ACTIONS[choice]
    
    def on_reset(self) -> None:
        self.last = None