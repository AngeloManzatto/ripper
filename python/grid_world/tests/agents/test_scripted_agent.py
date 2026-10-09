"""
Created on Fri Oct  9 17:42:54 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import ripper

from ripper import Action

from runtime.runner import run_episode
from agents.scripted_agent.agent import ScriptedAgent

###############################################################################
# Test enemy trap death transitions
###############################################################################

def test_enemy_trap_death_transitions():
    
    #################################
    # Layout
    #################################
    
    layout = (
        "#####\n"
        "#PTE#\n"
        "#...#\n"
        "#..G#\n"
        "#####"
    )
    
    #################################
    # Config
    #################################
    
    config = ripper.WorldConfig(
        max_tick=50, 
        player_perception_range=3
        )
    
    #################################
    # World
    #################################
    
    world = ripper.World(
        layout,
        config=config, 
        seed=42
    )
    
    player_agent = ScriptedAgent(action_list=[Action.Down, Action.Down, Action.Right, Action.Right,])
    enemy_agent  = ScriptedAgent(action_list=[Action.Left,])
    
    obs = world.reset().observation
    
    player_id = obs.player_id
    enemy_id  = obs.enemy_ids[0]
    
    agents = {
        player_id : player_agent,
        enemy_id  : enemy_agent
        }
    
    result = run_episode(
        world=world,
        agents=agents
    )
    
    assert result.outcome == "GoalReached"
    assert result.steps == 4
    
    assert [t.terminated for t in enemy_agent.transitions] == [True]
    assert [t.terminated for t in player_agent.transitions] == [False, False, False, True]
    assert enemy_agent.transitions[0].next_obs.status == "Trapped"
    
###############################################################################
# Run
###############################################################################

def run():
    test_enemy_trap_death_transitions()
    print("scripted agent: all checks passed")


if __name__ == "__main__":
    run()