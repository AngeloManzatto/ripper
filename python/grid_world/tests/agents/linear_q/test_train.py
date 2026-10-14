"""
Created on Tue Oct 13 06:26:31 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import time

from engine import GenerationConfig, World, WorldConfig, generate_valid_layout
from core.rewards import SparseGoalReward, ShapedGoalReward

from agents.linear_q_agent.agent import LinearQAgent
from agents.linear_q_agent.diagnostics import Recorder
from experiments.baselines.tasks import TASKS, EVAL_SEED
from runtime.train import train
from runtime.play import play
from runtime.evaluate import evaluate

###############################################################################
# Agent
###############################################################################

reward_fn  = ShapedGoalReward(w_bump=0.1)
alpha      = alpha=0.02
epsilon    = 0.1
agent_seed = 0

agent = LinearQAgent(reward_fn, alpha=alpha, epsilon=epsilon, eval_epsilon=0.0, seed=agent_seed)

rec = Recorder()
agent.recorder = rec

###############################################################################
# Task
###############################################################################

task = TASKS["default"]

###############################################################################
# Train
###############################################################################

n_blocks   = 20 
block_size = 25 
n_eval     = 100
world_config = WorldConfig(**task.world_kwargs)

start = time.time()
train_result = train(
    agent,
    n_blocks=n_blocks,
    block_size=block_size,
    n_eval=n_eval,
    train_seed=0,
    eval_seed=EVAL_SEED,
    generation_kwargs=task.generation_kwargs,
    world_config=world_config,
)

seconds = time.time() - start

###############################################################################
# Play
###############################################################################

agent.set_eval_mode()

episode_seed = 0

generation_config = {
    "width" : 16,
    "height" : 16,
    "num_enemies" : 1
    }

layout = generate_valid_layout(
    GenerationConfig(**generation_config, seed=episode_seed)
)


world = World(layout, config=None, seed=episode_seed)

# The player id can differ between layouts
player_id = world.reset().observation.player_id

agents = {
    player_id : agent
    }

play(world, agents)

###############################################################################
# Evaluate
###############################################################################

print("episodes  eval_goal  eval_steps")
for i in range(len(train_result.checkpoints)):
    print(f"{train_result.checkpoints[i]:8d}  "
          f"{train_result.eval_goal_rates[i]:9.3f}  "
          f"{train_result.eval_mean_steps[i]:10.1f}")
print("AUC:", round(train_result.auc, 3))
print("Online AUC:", round(train_result.online_auc, 3))

n_episodes = 100

final = evaluate(
    agent=agent,
    n_episodes=100,
    seed=EVAL_SEED,
    generation_kwargs=task.generation_kwargs,
    world_config=world_config,
)

print(final.outcome_counts)

agent.recorder = None
s = rec.summary()
for k, v in s.items():
    print(k, v)