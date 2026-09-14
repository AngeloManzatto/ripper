"""
Created on Thu Sep 10 13:06:16 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import ripper
from ripper import Action
from collections import Counter

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Evaluate
###############################################################################

def evaluate(agent, layout, max_tick=100, episodes=50):
    agent.set_eval_mode()

    world = ripper.PyWorld(layout, max_tick)
    player_id = world.get_player_id()
    outcomes = []

    for _ in range(episodes):
        world.reset()
        observation = world.observation()
        done = False

        while not done:
            action_idx = agent.act(observation.grid)
            step_result = world.step({player_id: ACTIONS[action_idx]})
            done, reason = step_result.done, step_result.reason
            observation = world.observation()

        outcomes.append(str(reason))

    agent.set_train_mode()

    counts = Counter(outcomes)
    win_rate = counts.get("EndReason.GoalReached", 0) / episodes

    return {
        "win_rate": win_rate,
        "episodes": episodes,
        "outcomes": dict(counts),
    }