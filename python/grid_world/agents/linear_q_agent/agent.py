"""
Created on Sat Oct 10 10:41:46 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Sequence

import numpy as np

from agents.base_agent import BaseAgent
from agents.features.frame import features, FEATURE_NAMES, N_FEATURES
from agents.features.temporal import memory_flags, MEMORY_NAMES, N_MEMORY
from agents.linear_q_agent.linear_q import q_values, epsilon_greedy, update

from core.actions import ACTIONS, N_ACTIONS
from core.rewards import RewardFn, SparseGoalReward
from core.transition import Transition

from engine import Action, EntityObservation

###############################################################################
# Input
###############################################################################

# What the Q-function sees: the frame features, then the recent-visit flags.
INPUT_NAMES = list(FEATURE_NAMES) + list(MEMORY_NAMES)
N_INPUTS    = N_FEATURES + N_MEMORY

def build_input(history: Sequence[EntityObservation]) -> np.ndarray:
    """Input vector for the state at the end of `history`.

    history: observations, oldest first, current one last (at least one).
    Returns a float32 array of length N_INPUTS, in the order of INPUT_NAMES.
    """
    frame  = features(history[-1])
    memory = np.asarray(memory_flags(history), dtype=np.float32)

    return np.concatenate([frame, memory])

###############################################################################
# Linear Q Agent
###############################################################################

class LinearQAgent(BaseAgent):

    name = "linear_q"

    def __init__(self,
                 reward_fn: RewardFn | None = None,
                 n_inputs=N_INPUTS,
                 n_actions=N_ACTIONS,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=0.1,
                 eval_epsilon=0.1,
                 seed=42,
                 history_size=8):
        super().__init__(seed=seed, history_size=history_size)

        self.n_inputs     = n_inputs
        self.n_actions    = n_actions

        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.eval_epsilon = eval_epsilon

        # One row per action, one column per input, plus a bias column
        self.W = np.zeros((n_actions, n_inputs + 1), dtype="float64")

        self.reward_fn = reward_fn if reward_fn is not None else SparseGoalReward()

        # The reward shapes with its own gamma and remembers its own window.
        # If they differ from the agent's, shaping silently stops being
        # consistent, so refuse to build such an agent.
        for name, mine in (("gamma", gamma), ("window", history_size)):
            theirs = getattr(self.reward_fn, name, None)
            if theirs is not None and theirs != mine:
                raise ValueError(
                    f"reward {name}={theirs} does not match agent {name}={mine}"
                )

        self.recorder = None

    ###########################################################################
    # Decision
    ###########################################################################

    def select_action(self, obs: EntityObservation) -> Action:

        # BaseAgent already appended `obs`: history[-1] is the current state
        x = build_input(self.history)
        q = q_values(self.W, x)

        if self.recorder is not None:
            self.recorder.log_decision(x, q, self.training)

        epsilon = self.epsilon if self.training else self.eval_epsilon

        action_idx = epsilon_greedy(q, epsilon=epsilon, rng=self.rng)

        return ACTIONS[action_idx]

    ###########################################################################
    # Learning
    ###########################################################################

    def on_transition(self, transition: Transition) -> None:

        # Evaluation mode: the agent is frozen
        if not self.training:
            return

        # What this transition is worth to this agent
        reward = self.reward_fn(transition)

        # The engine's Action -> row index in W
        action_idx = ACTIONS.index(transition.action)

        # State the agent acted in: history[-1] is transition.obs
        history = list(self.history)
        x = build_input(history)

        # State it landed in: the same window, moved one step forward
        next_history = (history + [transition.next_obs])[-self.history.maxlen:]
        x_next = build_input(next_history)

        q_before = None
        if self.recorder is not None:
            q_before = q_values(self.W, x)[action_idx]

        delta = update(
            self.W,
            x,
            action_idx,
            reward,
            x_next,
            transition.terminated,
            self.alpha,
            self.gamma,
        )

        if self.recorder is not None:
            q_after = q_values(self.W, x)[action_idx]
            self.recorder.log_update(x, delta, q_after - q_before, reward)

    def on_reset(self):

        self.reward_fn.reset()