"""
Created on Fri Sep 11 17:44:41 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from grid_world.training.evaluate import evaluate

###############################################################################
# MC evaluation
###############################################################################

def satisfies_minimal_criterion(
        agent, 
        layout, 
        max_tick=100, 
        episodes=20, 
        min_rate=0.2, 
        max_rate=0.7
        ):
    
    result = evaluate(agent, layout, max_tick=max_tick, episodes=episodes)
    
    win_rate = result["win_rate"]
    
    accepted = min_rate <= win_rate <= max_rate
    
    return accepted, result