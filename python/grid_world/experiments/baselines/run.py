"""
Created on Fri Oct 16 15:41:31 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from experiments.baselines.benchmark import run_benchmark
from experiments.baselines.tasks import TASKS
from experiments.baselines.compare import compare
from core.rewards import ShapedGoalReward

###############################################################################
# Agents Config
###############################################################################

agents_config = {
    "random"     : {},
    "heuristic"  : {},
    "q_learning" : {"reward_fn" : ShapedGoalReward(w_bump=0.5, w_revisit=0.5, w_danger=0.5)}
}

train_episodes = {"q_learning": 1000}      # episodes of training per agent seed

for task in TASKS.values():
    for agent_name in agents_config:
        agent_params = agents_config[agent_name]

        run_dir = run_benchmark(
            agent_name, task,
            agent_params=agent_params,
            train_episodes=train_episodes.get(agent_name, 0),
        )
        print(f"{task.name:<8} -> {run_dir}")

compare()