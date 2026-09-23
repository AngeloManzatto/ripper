"""
Created on Sun Sep 20 22:30:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from ripper import Observation

###############################################################################
# Parse Entity Observation
###############################################################################

def parse_entity_observation(entity_obs):
    """
    Convert a single pyo3 EntityObservation into a plain dict.
    """
    return {
        'id': entity_obs.id,
        'position': entity_obs.position,
        'grid': np.array(entity_obs.grid, dtype=np.int32),
    }

###############################################################################
# Parse Observation
###############################################################################

def parse_observation(observation: Observation):
    
    data = {}
    
    data["player_id"] = observation.player_id
    data["enemy_ids"] = observation.enemy_ids
    data["entities"] = {}
    
    for entity_obs in observation.entities:
        
        entity_data = parse_entity_observation(entity_obs)
        
        data["entities"].append(entity_data)
    
    return data

###############################################################################
# One Hot Encode Grid
###############################################################################

def one_hot_encode_grid(grid):
    
    # One-hot encode the map tile
    grid = np.asarray(grid)

    grid_one_hot = np.zeros(
        (*grid.shape, 7),
        dtype=np.uint8
    )

    grid_one_hot[grid == -1, 0] = 1  # Unknown
    grid_one_hot[grid == 0, 1] = 1   # Free
    grid_one_hot[grid == 1, 2] = 1   # Wall
    grid_one_hot[grid == 2, 3] = 1   # Player
    grid_one_hot[grid == 3, 4] = 1   # Enemy
    grid_one_hot[grid == 4, 5] = 1   # Trap
    grid_one_hot[grid == 5, 6] = 1   # Goal

    return np.transpose(grid_one_hot, (2, 0, 1))


###############################################################################
# One Hot Encode Grid For Entity
###############################################################################

def get_one_hot_grid_for_entity(observation, entity_id):

    for entity_obs in observation.entities:
        if entity_obs.id == entity_id:
            return one_hot_encode_grid(entity_obs.grid)

    raise ValueError(f"entity_id '{entity_id}' not found in observation.entities")