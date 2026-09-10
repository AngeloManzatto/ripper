"""
Created on Thu Sep 10 13:06:16 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import torch

import ripper
from ripper import Action

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Evaluate
###############################################################################

def evaluate(agent, layout, max_tick=100, episodes=20):
    world = ripper.PyWorld(layout, max_tick)
    player_id = world.get_player_id()
    wins = 0

    for _ in range(episodes):
        world.reset()
        observation = world.observation()
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(observation.grid, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action_idx = q_values.argmax(dim=1).item()

            step_result = world.step({player_id: ACTIONS[action_idx]})
            done, reason = step_result.done, step_result.reason
            observation = world.observation()

        if reason == ripper.EndReason.GoalReached:
            wins += 1

    return wins / episodes