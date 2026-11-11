"""
Created on Wed Nov 11 09:45:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from agents.ddqn_agent.replay_buffer import ReplayBuffer
from agents.neat_agent.agent import N_INPUTS, build_input

from core.actions import ACTIONS, N_ACTIONS
from core.rewards import RewardFn, SparseGoalReward
from core.transition import Transition

from engine import Action, EntityObservation

###############################################################################
# Network
###############################################################################

def build_mlp(n_inputs: int = N_INPUTS, n_actions: int = N_ACTIONS, hidden: int = 128) -> nn.Module:
    """A small fully connected Q-network: n_inputs -> hidden -> hidden -> n_actions.

    It maps a batch of input vectors (shape [batch, n_inputs], float32) to one Q-value
    per action (shape [batch, n_actions]). It lives on the CPU. The weights come from
    torch's own random generator: call torch.manual_seed before building it to repeat it.
    """
    return nn.Sequential(
        nn.Linear(n_inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )

###############################################################################
# The Double DQN target
###############################################################################

def td_targets(policy_net, target_net, rewards, next_states, terminated, gamma):
    """What the Q-value of the action taken should be: reward + gamma * value of where it landed.

    Double DQN: the POLICY network chooses the best action in the next state, the TARGET network
    says how much that action is worth. (Taking the maximum of one network over its own noisy
    estimates overestimates; splitting choice and value reduces it.)
    A game that really ended (terminated = 1) has no future: the target is the reward alone.
    A game cut by the clock is not terminated: it still bootstraps.

    rewards, terminated: float tensors [batch]; next_states: float tensor [batch, n_inputs].
    Returns a float tensor [batch] with no gradient.
    """
    with torch.no_grad():
        best = policy_net(next_states).argmax(dim=1, keepdim=True)
        value = target_net(next_states).gather(1, best).squeeze(1)

        return rewards + gamma * value * (1.0 - terminated)

###############################################################################
# DDQN Agent
###############################################################################

class DdqnAgent(BaseAgent):
    """
    A Double DQN agent. It sees what the NEAT agent sees (build_input of the NEAT agent),
    plays the action with the largest Q-value, and learns from its own transitions.

    Playing
        With probability `epsilon` while training, `eval_epsilon` in eval mode, the action is
        drawn at random from the agent's own rng. An epsilon of 0 never touches the rng.

    Learning (training mode only; in eval mode on_transition does nothing)
        on_transition stores (state, action, reward, next state, terminated) in a replay buffer.
        Once the buffer holds `min_buffer_size` transitions, every transition also does one update:
        a batch of `batch_size` stored transitions is drawn (with the agent's rng) and the policy
        network is moved towards td_targets (Huber loss, gradient norm clipped to 1).
        Every `target_update_freq` updates the target network becomes a copy of the policy network.
        Every update multiplies epsilon by `epsilon_decay`, never below `epsilon_min`
        (the default decay of 1.0 keeps epsilon fixed).
        The reward comes from `reward_fn` (default SparseGoalReward), as in LinearQAgent.

    clone(): the networks (policy and target) are copied, with the current epsilon, the mode and
    the rng state. The copy gets a FRESH optimizer and an EMPTY replay buffer, and counts updates
    from zero: it carries what was learned, not the memory of how.

    Save / load are not implemented: pickle the agent (the pool does; the buffer goes with it).
    """

    name = "ddqn"

    def __init__(self,
                 policy_net,
                 reward_fn: RewardFn | None = None,
                 epsilon=0.1,
                 eval_epsilon=0.0,
                 epsilon_min=0.05,
                 epsilon_decay=1.0,
                 gamma=0.99,
                 lr=1e-4,
                 batch_size=64,
                 min_buffer_size=1000,
                 buffer_capacity=20_000,
                 target_update_freq=500,
                 seed=42,
                 history_size=8):
        super().__init__(seed=seed, history_size=history_size)

        for label, value in (("epsilon", epsilon), ("eval_epsilon", eval_epsilon),
                             ("epsilon_min", epsilon_min), ("gamma", gamma)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not batch_size <= min_buffer_size <= buffer_capacity:
            raise ValueError(
                f"need batch_size <= min_buffer_size <= buffer_capacity, "
                f"got {batch_size}, {min_buffer_size}, {buffer_capacity}"
            )
        if target_update_freq < 1:
            raise ValueError(f"target_update_freq must be >= 1, got {target_update_freq}")

        self._check_network(policy_net)

        self.reward_fn = reward_fn if reward_fn is not None else SparseGoalReward()

        # The reward shapes with its own gamma and window: refuse a mismatch (as LinearQAgent does)
        for label, mine in (("gamma", gamma), ("window", history_size)):
            theirs = getattr(self.reward_fn, label, None)
            if theirs is not None and theirs != mine:
                raise ValueError(f"reward {label}={theirs} does not match agent {label}={mine}")

        self.policy_net = policy_net
        self.target_net = copy.deepcopy(policy_net)
        self.target_net.eval()
        self.target_net.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.epsilon = epsilon
        self.eval_epsilon = eval_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq

        self.learn_steps = 0          # updates done so far
        self.last_loss = None         # loss of the last update, None before the first

    @staticmethod
    def _check_network(policy_net):
        """A dummy input must give one Q-value per action. Raises ValueError otherwise."""
        try:
            with torch.no_grad():
                out = policy_net(torch.zeros(1, N_INPUTS, dtype=torch.float32))
        except RuntimeError as error:
            raise ValueError(f"the network cannot take {N_INPUTS} inputs: {error}") from error

        if tuple(out.shape) != (1, N_ACTIONS):
            raise ValueError(
                f"the network gives {tuple(out.shape)} for one input, expected (1, {N_ACTIONS})"
            )

    ###########################################################################
    # Playing
    ###########################################################################

    def q_values(self) -> np.ndarray:
        """The Q-value of every action in the current state (history[-1])."""
        x = torch.as_tensor(build_input(self.history), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(x).squeeze(0).numpy()

    def select_action(self, obs: EntityObservation) -> Action:

        # BaseAgent already appended `obs`: history[-1] is the current state
        epsilon = self.epsilon if self.training else self.eval_epsilon

        if epsilon > 0.0 and self.rng.random() < epsilon:
            return ACTIONS[int(self.rng.integers(N_ACTIONS))]

        # np.argmax: ties go to the first action, like the NEAT agent
        return ACTIONS[int(np.argmax(self.q_values()))]

    ###########################################################################
    # Learning
    ###########################################################################

    def on_transition(self, transition: Transition) -> None:

        # Evaluation mode: the agent is frozen
        if not self.training:
            return

        reward = float(self.reward_fn(transition))

        # State the agent acted in: history[-1] is transition.obs
        history = list(self.history)
        x = build_input(history)

        # State it landed in: the same window, moved one step forward
        next_history = (history + [transition.next_obs])[-self.history.maxlen:]
        x_next = build_input(next_history)

        # Only a real end cuts the future. A game cut by the clock (truncated) still bootstraps.
        self.replay_buffer.push(x, ACTIONS.index(transition.action), reward, x_next, transition.terminated)

        self._update()

    def _update(self):
        """One gradient step on a batch from the buffer, if the buffer is ready. Returns the loss."""
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        states, actions, rewards, next_states, terminated = (
            torch.from_numpy(a) for a in self.replay_buffer.sample(self.batch_size, self.rng)
        )

        q_taken = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = td_targets(self.policy_net, self.target_net, rewards, next_states, terminated, self.gamma)

        loss = F.smooth_l1_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learn_steps += 1
        self.last_loss = loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.last_loss

    def on_reset(self):

        self.reward_fn.reset()

    ###########################################################################
    # Copies
    ###########################################################################

    def clone(self) -> "DdqnAgent":
        other = DdqnAgent(
            copy.deepcopy(self.policy_net),
            reward_fn=copy.deepcopy(self.reward_fn),
            epsilon=self.epsilon,
            eval_epsilon=self.eval_epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            gamma=self.gamma,
            lr=self.lr,
            batch_size=self.batch_size,
            min_buffer_size=self.min_buffer_size,
            buffer_capacity=self.buffer_capacity,
            target_update_freq=self.target_update_freq,
            history_size=self.history.maxlen,
        )
        other.target_net.load_state_dict(self.target_net.state_dict())
        other.rng = copy.deepcopy(self.rng)
        other.training = self.training
        return other