"""
Created on Thu Sep 10 13:06:16 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections import Counter

from ripper import Action

from agents.observation import get_one_hot_grid_for_entity, get_position_for_entity, get_legal_action_mask, TRAIL_CHANNEL
from agents.trail_tracker import TrailTracker

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Evaluate Match
###############################################################################

def evaluate_match(agents, agent_ids, world, episodes, verbose=False):
    
    # Outcomes for each agent. Palyer and Enemy are decided differently
    outcomes = {agent_id: [] for agent_id in agent_ids}
    
    for agent, agent_id in zip(agents, agent_ids):
        
        agent.set_eval_mode()
    
    try:
        for ep in range(episodes):
            world.reset(reposition=True)
            observation = world.observation()
            done = False
            ticks = 0
            
            # Fresh trail tracker per agent this episode, seeded with starting position
            grid_shape = get_one_hot_grid_for_entity(observation, world.get_player_id()).shape[1:]
            trail_trackers = {
                agent_id: TrailTracker(grid_shape=grid_shape, maxlen=4)
                for agent_id in agent_ids
            }
 
            while not done:
                
                actions = {}
                
                for agent, agent_id in zip(agents, agent_ids):
                    
                    # Get current state for agent
                    state = get_one_hot_grid_for_entity(observation, agent_id)
                    state[TRAIL_CHANNEL] = trail_trackers[agent_id].get_grid()
                    
                    # Get mask for legal actions
                    position = get_position_for_entity(observation, agent_id)
                    legal_mask = get_legal_action_mask(state, position)
                    
                    # Play an action given current env state
                    action_idx = agent.act(state, legal_mask)
                                           
                    # Regist action for agent
                    actions[agent_id] = ACTIONS[action_idx]
                    
                # Act on env
                step_result = world.step(actions)
                
                # Get current env result
                done, reason = step_result.done, step_result.reason
                
                # Update observation current state
                observation = world.observation()
                
                # Record the move each (still-present) agent just made
                next_ids = {entity.id for entity in observation.entities}
                for agent_id in agent_ids:
                    if agent_id in next_ids:
                        trail_trackers[agent_id].update(get_position_for_entity(observation, agent_id))
                
                # Update step count
                ticks += 1
                
            # Store outcomes
            for agent_id in agent_ids:
                outcomes[agent_id].append(str(reason))
                
            if verbose:
                print(f"    [eval] episode {ep+1}/{episodes} | ticks={ticks} | reason={reason}")
    finally:
        for agent in agents:
            agent.set_train_mode()
        
    player_id = world.get_player_id()
    results = {}
    for agent_id in agent_ids:
        is_player = agent_id == player_id
        win_reason = "EndReason.GoalReached" if is_player else "EndReason.Caught"
        breakdown = Counter(outcomes[agent_id])

        results[agent_id] = {
            "win_rate": breakdown.get(win_reason, 0) / episodes,
            "timeout_rate": breakdown.get("EndReason.Timeout", 0) / episodes,
            "breakdown": dict(breakdown),
        }
        
    if verbose:
        print(f"    [eval] done | {results}")
            
    return results

###############################################################################
# Evaluate MC (Minimum Criterion POET)
###############################################################################

def passes_minimal_criterion(results, win_low=0.2, win_high=0.7, timeout_cap=0.3):
    """
    Decide whether a layout's current evaluation is a valid stepping stone:
    neither side dominates, and the episodes are actually resolving.

    Parameters
    ----------
    results : dict
        Shape returned by evaluate_match:
        {agent_id: {"win_rate": float, "timeout_rate": float, "breakdown": dict}, ...}
    win_low, win_high : float
        Every agent's win_rate must fall within [win_low, win_high] (inclusive).
    timeout_cap : float
        Reject if timeout_rate exceeds this (checked once -- it's the same
        value across every agent in `results`, since it comes from the same
        shared set of episode outcomes).

    Returns
    -------
    bool
    """
 
    for value in results.values():
        
        if not win_low <= value["win_rate"] <= win_high:
            return False
        
        if value["timeout_rate"] > timeout_cap:
            return False

    return True