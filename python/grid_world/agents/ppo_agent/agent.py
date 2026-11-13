"""
Created on Fri Nov 13 09:10:00 2026

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
from agents.neat_agent.agent import N_INPUTS, build_input

from core.actions import ACTIONS, N_ACTIONS
from core.rewards import RewardFn, SparseGoalReward
from core.transition import Transition

from engine import Action, EntityObservation

###############################################################################
# Network
###############################################################################

class ActorCritic(nn.Module):
    """One shared body and two heads: the actor (a score per action) and the critic (one state value).

    forward maps a batch of input vectors (float32, [batch, n_inputs]) to
    (logits [batch, n_actions], value [batch]). It lives on the CPU. The weights come from
    torch's own random generator: call torch.manual_seed before building it to repeat it.
    """

    def __init__(self, n_inputs: int = N_INPUTS, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


def build_actor_critic(n_inputs: int = N_INPUTS, n_actions: int = N_ACTIONS, hidden: int = 128) -> ActorCritic:
    """Same idea as build_mlp of the DDQN agent: n_inputs -> hidden -> hidden -> (actor, critic)."""
    return ActorCritic(n_inputs, n_actions, hidden)

###############################################################################
# Generalized Advantage Estimation
###############################################################################

def compute_gae(rewards, values, next_values, terminated, ended, gamma, gae_lambda):
    """How much better each action turned out than the critic expected, and the value it should have had.

    All arguments are 1-d arrays of one rollout, in the order the steps were played.
        rewards, values       : reward of the step, critic value of the state it was taken in
        next_values           : critic value of the state it landed in
        terminated            : 1 if the game really ended on this step: no future, next value is ignored
                                (a game cut by the clock is NOT terminated: it still bootstraps)
        ended                 : 1 if the episode stops after this step (terminated or truncated) or the
                                rollout stops here: the advantage chain does not cross it

    delta[t]     = reward[t] + gamma * next_value[t] * (1 - terminated[t]) - value[t]
    advantage[t] = delta[t] + gamma * gae_lambda * (1 - ended[t]) * advantage[t+1]
    return[t]    = advantage[t] + value[t]

    Returns (advantages, returns), two float64 arrays.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=np.float64)
    ended = np.asarray(ended, dtype=np.float64)

    advantages = np.zeros(len(rewards), dtype=np.float64)
    following = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
        following = delta + gamma * gae_lambda * (1.0 - ended[t]) * following
        advantages[t] = following

    return advantages, advantages + values

###############################################################################
# PPO Agent
###############################################################################

class PpoAgent(BaseAgent):
    """
    A PPO agent (clipped objective, GAE). It sees what the NEAT and DDQN agents see
    (build_input of the NEAT agent) and learns from its own transitions.

    Playing
        Training: the action is SAMPLED from the policy (softmax of the actor's scores), using the
        agent's own rng. The policy itself explores; there is no epsilon.
        Eval mode: the action with the largest score (ties: the first); with probability
        `eval_epsilon` it is drawn at random instead (as the DDQN agent does, to break two
        memoryless agents that flip-flop). An eval_epsilon of 0 never touches the rng.

    Learning (training mode only; in eval mode on_transition does nothing)
        PPO is on-policy: it learns from what the CURRENT policy just played and then throws it away.
        on_transition stores one step (state, action, log-probability of that action, critic value,
        reward, next state, terminated, episode ended). When `rollout_steps` steps are stored
        (they may come from several games), one update runs and the rollout is emptied:
            advantages and returns by compute_gae, advantages normalized to mean 0 / std 1,
            `epochs` passes over the rollout in minibatches of `batch_size` (order drawn with the
            agent's rng); loss = clipped policy loss + value_coef * value error - entropy_coef * entropy,
            gradient norm clipped to 1.
        The log-probability and the value are computed in on_transition with the same weights that
        chose the action (the weights only change at the end of an update), so they are exact.
        The reward comes from `reward_fn` (default SparseGoalReward), as in DdqnAgent.

    clone(): the network is copied, with the mode and the rng state. The copy gets a FRESH optimizer
    and an EMPTY rollout, and counts updates from zero: it carries what was learned, not the memory.

    Save / load are not implemented. Like every agent, one that has played holds engine observations
    (its history) that can be neither copied nor pickled: copy it with clone().
    """

    name = "ppo"

    def __init__(self,
                 net,
                 reward_fn: RewardFn | None = None,
                 eval_epsilon=0.0,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 lr=3e-4,
                 rollout_steps=512,
                 epochs=4,
                 batch_size=64,
                 seed=42,
                 history_size=8):
        super().__init__(seed=seed, history_size=history_size)

        for label, value in (("eval_epsilon", eval_epsilon), ("gamma", gamma), ("gae_lambda", gae_lambda)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}")
        if clip_epsilon <= 0:
            raise ValueError(f"clip_epsilon must be > 0, got {clip_epsilon}")
        if entropy_coef < 0 or value_coef < 0:
            raise ValueError(f"entropy_coef and value_coef must be >= 0, got {entropy_coef}, {value_coef}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if rollout_steps < 2:
            raise ValueError(f"rollout_steps must be >= 2, got {rollout_steps}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._check_network(net)

        self.reward_fn = reward_fn if reward_fn is not None else SparseGoalReward()

        # The reward shapes with its own gamma and window: refuse a mismatch (as DdqnAgent does)
        for label, mine in (("gamma", gamma), ("window", history_size)):
            theirs = getattr(self.reward_fn, label, None)
            if theirs is not None and theirs != mine:
                raise ValueError(f"reward {label}={theirs} does not match agent {label}={mine}")

        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.eval_epsilon = eval_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lr = lr
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.batch_size = batch_size

        self._clear_rollout()

        self.updates = 0              # rollouts learned from so far
        self.last_loss = None         # mean loss of the last update, None before the first
        self.last_entropy = None      # mean policy entropy of the last update

    @staticmethod
    def _check_network(net):
        """A dummy input must give one score per action and one value. Raises ValueError otherwise."""
        try:
            with torch.no_grad():
                logits, value = net(torch.zeros(1, N_INPUTS, dtype=torch.float32))
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"the network cannot take {N_INPUTS} inputs and give (logits, value): {error}") from error

        if tuple(logits.shape) != (1, N_ACTIONS):
            raise ValueError(f"the actor gives {tuple(logits.shape)} for one input, expected (1, {N_ACTIONS})")
        if tuple(value.shape) != (1,):
            raise ValueError(f"the critic gives {tuple(value.shape)} for one input, expected (1,)")

    def _clear_rollout(self):
        self._states, self._actions, self._log_probs, self._values = [], [], [], []
        self._rewards, self._next_states, self._terminated, self._ended = [], [], [], []

    @property
    def rollout_size(self) -> int:
        """Steps stored and not yet learned from."""
        return len(self._rewards)

    ###########################################################################
    # Playing
    ###########################################################################

    def action_probabilities(self) -> np.ndarray:
        """The policy in the current state (history[-1]): one probability per action, sums to 1."""
        x = torch.as_tensor(build_input(self.history), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy().astype(np.float64)

        return probs / probs.sum()           # float32 softmax sums to 1 only up to ~1e-7

    def select_action(self, obs: EntityObservation) -> Action:

        # BaseAgent already appended `obs`: history[-1] is the current state
        if self.training:
            probs = self.action_probabilities()
            # Inverse-CDF draw with one number from the agent's rng
            index = int(np.searchsorted(np.cumsum(probs), self.rng.random(), side="right"))
            return ACTIONS[min(index, N_ACTIONS - 1)]

        if self.eval_epsilon > 0.0 and self.rng.random() < self.eval_epsilon:
            return ACTIONS[int(self.rng.integers(N_ACTIONS))]

        # np.argmax: ties go to the first action, like the other agents
        return ACTIONS[int(np.argmax(self.action_probabilities()))]

    ###########################################################################
    # Learning
    ###########################################################################

    def on_transition(self, transition: Transition) -> None:

        # Evaluation mode: the agent is frozen
        if not self.training:
            return

        reward = float(self.reward_fn(transition))
        index = ACTIONS.index(transition.action)

        # State the agent acted in: history[-1] is transition.obs
        history = list(self.history)
        x = build_input(history)

        # State it landed in: the same window, moved one step forward
        next_history = (history + [transition.next_obs])[-self.history.maxlen:]
        x_next = build_input(next_history)

        with torch.no_grad():
            logits, value = self.net(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))
            log_prob = F.log_softmax(logits, dim=1)[0, index].item()

        self._states.append(x)
        self._actions.append(index)
        self._log_probs.append(log_prob)
        self._values.append(value.item())
        self._rewards.append(reward)
        self._next_states.append(x_next)
        self._terminated.append(float(transition.terminated))
        self._ended.append(float(transition.terminated or transition.truncated))

        if len(self._rewards) >= self.rollout_steps:
            self._update()

    def _update(self):
        """Learn from the stored rollout, then empty it. Returns the mean loss."""
        states = torch.as_tensor(np.array(self._states), dtype=torch.float32)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)
        actions = torch.as_tensor(self._actions, dtype=torch.long)
        old_log_probs = torch.as_tensor(self._log_probs, dtype=torch.float32)

        with torch.no_grad():
            _, next_values = self.net(next_states)

        ended = list(self._ended)
        ended[-1] = 1.0                      # the rollout stops here: the chain does not go on
        advantages, returns = compute_gae(self._rewards, self._values, next_values.numpy(),
                                          self._terminated, ended, self.gamma, self.gae_lambda)

        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        returns = torch.as_tensor(returns, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = len(actions)
        losses, entropies = [], []

        for _ in range(self.epochs):
            order = torch.from_numpy(self.rng.permutation(n))

            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]

                logits, values = self.net(states[idx])
                log_probs = F.log_softmax(logits, dim=1)
                new_log_probs = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

                policy_loss = -torch.min(ratio * advantages[idx], clipped * advantages[idx]).mean()
                value_loss = F.mse_loss(values, returns[idx])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

                losses.append(loss.item())
                entropies.append(entropy.item())

        self._clear_rollout()

        self.updates += 1
        self.last_loss = float(np.mean(losses))
        self.last_entropy = float(np.mean(entropies))

        return self.last_loss

    def on_reset(self):

        self.reward_fn.reset()

    ###########################################################################
    # Copies
    ###########################################################################

    def clone(self) -> "PpoAgent":
        other = PpoAgent(
            copy.deepcopy(self.net),
            reward_fn=copy.deepcopy(self.reward_fn),
            eval_epsilon=self.eval_epsilon,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            lr=self.lr,
            rollout_steps=self.rollout_steps,
            epochs=self.epochs,
            batch_size=self.batch_size,
            history_size=self.history.maxlen,
        )
        other.rng = copy.deepcopy(self.rng)
        other.training = self.training
        return other