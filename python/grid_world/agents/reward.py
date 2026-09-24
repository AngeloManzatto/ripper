"""
Created on Sun Sep 13 08:59:32 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from ripper import EndReason
import numpy as np

###############################################################################
# Channel indices (matches one_hot_encode_grid in observation.py)
###############################################################################

GOAL_CHANNEL   = 6  # what the player is looking for
PLAYER_CHANNEL = 3  # what the enemy is looking for

###############################################################################
# Terminal reward
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

###############################################################################
# Novelty bonus -- reward for shrinking the fog of war
###############################################################################

def count_unknown(state):
    # channel 0 is "Unknown" in the one-hot encoding
    return int(state[0].sum())

def reward_for_novelty(state, next_state, k=0.02):
    return k * (count_unknown(state) - count_unknown(next_state))

###############################################################################
# Sighting bonus -- reward for the target entering view (once, not per-step)
###############################################################################

def reward_for_sighting(state, next_state, channel, k=0.15):
    was_visible = bool(state[channel].any())
    now_visible = bool(next_state[channel].any())
    return k if (not was_visible and now_visible) else 0.0

###############################################################################
# Distance bonus -- reward for closing in, only while the target is visible
###############################################################################

def find_visible_position(state, channel):
    coords = np.argwhere(state[channel] == 1)
    return tuple(coords[0]) if len(coords) > 0 else None

def reward_for_distance(own_pos, own_pos_next, state, next_state, channel, k=0.05):
    target_before = find_visible_position(state, channel)
    target_after  = find_visible_position(next_state, channel)

    if target_before is None or target_after is None:
        return 0.0

    dist_before = abs(own_pos[0] - target_before[0]) + abs(own_pos[1] - target_before[1])
    dist_after  = abs(own_pos_next[0] - target_after[0]) + abs(own_pos_next[1] - target_after[1])

    return k * (dist_before - dist_after)

###############################################################################
# Composed reward per role
###############################################################################

def calculate_reward(
        reason, 
        is_player, 
        position, state, 
        next_position, next_state,
        k_novel=0.02, 
        k_sight=0.15, 
        k_dist=0.05
    ):
    
    
    channel = GOAL_CHANNEL if is_player else PLAYER_CHANNEL

    reward  = reward_for(reason, is_player)
    reward += reward_for_novelty(state, next_state, k=k_novel)
    reward += reward_for_sighting(state, next_state, channel, k=k_sight)
    reward += reward_for_distance(position, next_position, state, next_state, channel, k=k_dist)

    return reward

def calculate_player_reward(reason, own_pos, own_pos_next, state, next_state):
    return calculate_reward(reason, True, own_pos, own_pos_next, state, next_state)

def calculate_enemy_reward(reason, own_pos, own_pos_next, state, next_state):
    return calculate_reward(reason, False, own_pos, own_pos_next, state, next_state)