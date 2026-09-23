"""
Created on Sun Sep 13 08:59:32 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from ripper import EndReason

###############################################################################
# Reward
###############################################################################

def reward_for(reason, is_player):
    
    reward = -0.1
    
    if is_player:
        
        if reason == EndReason.GoalReached:
            reward = 1.0
        elif reason in (EndReason.Caught, EndReason.Trapped, EndReason.Timeout):
            reward = -1.0
            
    else:
        
        if reason == EndReason.Caught:
            reward = 1.0
        elif reason in (EndReason.GoalReached, EndReason.Trapped, EndReason.Timeout):
            reward = -1.0
            
    return reward
        
        
