"""
Created on Mon Sep 21 22:12:48 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import time

from agents.observation import get_one_hot_grid_for_entity

from ripper import Action

###############################################################################
# Globals
###############################################################################

ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

###############################################################################
# Play match
###############################################################################

def play_match(agents, agent_ids, world, delay=0.3, max_steps=None):
    """
    Run a single episode with each agent acting greedily (eval mode),
    printing the world state after every tick. Restores each agent's
    prior mode (train/eval) when done, even if interrupted.
    """
    assert len(agents) == len(agent_ids)

    for agent in agents:
        agent.set_eval_mode()

    try:
        world.reset()
        observation = world.observation()
        done = False
        step = 0

        print(f"--- Step {step} ---")
        print(world)

        while not done and (max_steps is None or step < max_steps):
            actions = {}

            for agent, agent_id in zip(agents, agent_ids):
                state = get_one_hot_grid_for_entity(observation, agent_id)
                action_idx = agent.act(state)
                actions[agent_id] = ACTIONS[action_idx]

            step_result = world.step(actions)
            done, reason = step_result.done, step_result.reason

            observation = world.observation()
            step += 1

            time.sleep(delay)
            print(f"--- Step {step} ---")
            print(world)

        print(f"\nEpisode finished after {step} steps — reason: {reason}")

    finally:
        for agent in agents:
            agent.set_train_mode()