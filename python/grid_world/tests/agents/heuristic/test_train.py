#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:07:03 2026

@author: root
"""

"""
Evaluate the HeuristicAgent on the same eval layouts the learned agent uses.

No training: the heuristic is a fixed rule set, so everything here is an
evaluation. Same EVAL_SEED, same task, same n_eval as test_train.py, so the
numbers can be compared directly with the learner's frozen outcome counts.
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from engine import GenerationConfig, World, WorldConfig, generate_valid_layout

from agents.heuristic_agent.agent import HeuristicAgent      # adjust if your path differs
from experiments.baselines.tasks import TASKS, EVAL_SEED
from runtime.play import play
from runtime.evaluate import evaluate

###############################################################################
# What to run
###############################################################################

RUN_TABLE       = True    # A) epsilon x agent seeds, outcome counts
RUN_DETERMINISM = True    # B) same config twice must print the same counts
RUN_PLAY        = True    # C) watch one episode

###############################################################################
# Settings
###############################################################################

AGENT_SEEDS = (0, 1, 2)
EPSILONS    = (0.0, 0.1, 0.2)      # 0.1 is the heuristic's default
N_EVAL      = 100                  # same as the learner's evaluation

task         = TASKS["default"]
world_config = WorldConfig(**task.world_kwargs)


def make_agent(agent_seed, epsilon):
    return HeuristicAgent(seed=agent_seed, epsilon=epsilon)


def evaluate_agent(agent, n_episodes=N_EVAL):
    agent.set_eval_mode()
    return evaluate(
        agent=agent,
        n_episodes=n_episodes,
        seed=EVAL_SEED,
        generation_kwargs=task.generation_kwargs,
        world_config=world_config,
    )


def counts_row(result):
    c = result.outcome_counts
    return (c.get("GoalReached", 0), c.get("Caught", 0),
            c.get("Trapped", 0), c.get("Timeout", 0))

###############################################################################
# A) Table: epsilon x agent seeds
###############################################################################

if RUN_TABLE:

    results = {}

    print("epsilon seed  goal caught trapped timeout")
    for epsilon in EPSILONS:
        for agent_seed in AGENT_SEEDS:
            row = counts_row(evaluate_agent(make_agent(agent_seed, epsilon)))
            results[(epsilon, agent_seed)] = row
            print(f"{epsilon:7.2f} {agent_seed:4d} {row[0]:5d} {row[1]:6d} "
                  f"{row[2]:7d} {row[3]:7d}")

    print("\nepsilon  mean goal  caught  trapped  timeout")
    for epsilon in EPSILONS:
        m = np.array([results[(epsilon, s)] for s in AGENT_SEEDS]).mean(axis=0)
        print(f"{epsilon:7.2f} {m[0]:10.1f} {m[1]:7.1f} {m[2]:8.1f} {m[3]:8.1f}")

###############################################################################
# B) Determinism: same agent seed twice must give identical counts
###############################################################################

if RUN_DETERMINISM:

    for _ in range(2):
        print("determinism:", counts_row(evaluate_agent(make_agent(0, 0.1))))

###############################################################################
# C) Play one episode
###############################################################################

if RUN_PLAY:

    agent = make_agent(0, 0.1)
    agent.set_eval_mode()

    episode_seed = 10
    layout = generate_valid_layout(
        GenerationConfig(**task.generation_kwargs, seed=episode_seed)
    )
    world = World(layout, config=world_config, seed=episode_seed)
    player_id = world.reset().observation.player_id

    play(world, {player_id: agent})