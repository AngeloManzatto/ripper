"""
Created on Mon Oct 12 07:52:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from dataclasses import dataclass

import numpy as np

from runtime.evaluate import evaluate, play_episodes
from runtime.runner import EpisodeResult

###############################################################################
# Train Result
###############################################################################

@dataclass(frozen=True)
class TrainResult:
    checkpoints: list[int]
    eval_goal_rates: list[float]
    eval_mean_steps: list[float]
    train_goal_rates: list[float]

    @property
    def auc(self) -> float:
        """Mean held-out goal rate over the trained checkpoints (checkpoint 0,
        the untrained agent, is excluded). Equal spacing makes this the
        normalized area under the learning curve, in [0, 1]."""
        return float(np.mean(self.eval_goal_rates[1:]))

    @property
    def online_auc(self) -> float:
        return float(np.mean(self.train_goal_rates))

###############################################################################
# Train agent: learn by playing episodes
###############################################################################

def train_agent(
        agent,
        n_episodes,
        seed,
        generation_kwargs=None,
        world_config=None
    ) -> list[EpisodeResult]:
    """Let `agent` learn by playing episodes seed, seed+1, ...

    The agent is put in train mode for the run and its previous mode is
    restored afterwards. Returns the per-episode results.
    """

    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    was_training = agent.training
    agent.set_train_mode()

    try:
        return play_episodes(agent, n_episodes, seed, generation_kwargs, world_config)
    finally:
        if was_training:
            agent.set_train_mode()
        else:
            agent.set_eval_mode()

###############################################################################
# Train with checkpoints
###############################################################################

def train(
        agent,
        *,
        n_blocks,
        block_size,
        n_eval,
        train_seed,
        eval_seed,
        generation_kwargs=None,
        world_config=None
    ) -> TrainResult:

    # Check the three counts are at least 1 ----
    counts = {
        "n_blocks"   : n_blocks,
        "block_size" : block_size,
        "n_eval"     : n_eval
    }

    for name, value in counts.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, {value} was provided.")

    # Training seeds must not reach the held-out seeds ----
    n_train_episodes = n_blocks * block_size
    last_train_seed = train_seed + n_train_episodes - 1
    if last_train_seed >= eval_seed:
        raise ValueError(
            f"training seeds reach {last_train_seed}, they must stay below "
            f"eval_seed {eval_seed}"
        )

    initial_training = agent.training            # restored at the end

    checkpoints = []
    eval_goal_rates = []
    eval_mean_steps = []
    train_goal_rates = []
    episodes_done = 0

    try:
        # ---- Checkpoint 0: the untrained agent (evaluate freezes it) ----
        summary = evaluate(
            agent,
            n_episodes=n_eval,
            seed=eval_seed,
            generation_kwargs=generation_kwargs,
            world_config=world_config,
        )
        checkpoints.append(0)
        eval_goal_rates.append(summary.rate("GoalReached"))
        eval_mean_steps.append(summary.mean_steps)

        # ---- Training blocks ----
        for block in range(n_blocks):

            results = train_agent(
                agent,
                block_size,
                train_seed + episodes_done,
                generation_kwargs=generation_kwargs,
                world_config=world_config,
            )
            goals = sum(1 for r in results if r.outcome == "GoalReached")
            train_goal_rates.append(goals / block_size)
            episodes_done += block_size

            summary = evaluate(
                agent,
                n_episodes=n_eval,
                seed=eval_seed,
                generation_kwargs=generation_kwargs,
                world_config=world_config,
            )
            checkpoints.append(episodes_done)
            eval_goal_rates.append(summary.rate("GoalReached"))
            eval_mean_steps.append(summary.mean_steps)

    finally:
        if initial_training:
            agent.set_train_mode()
        else:
            agent.set_eval_mode()

    return TrainResult(
        checkpoints=checkpoints,
        eval_goal_rates=eval_goal_rates,
        eval_mean_steps=eval_mean_steps,
        train_goal_rates=train_goal_rates
    )