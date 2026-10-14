"""
Created on Sat Oct 10 19:50:08 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import dataclasses
import numpy as np
import ripper
from ripper import Action

from core.actions import ACTIONS
from core.features import features, N_FEATURES
from core.rewards import RewardFn, SparseGoalReward
from core.transition import Transition
from agents.linear_q_agent.agent import LinearQAgent

LAYOUT = (
    "#####\n"
    "#P.T#\n"
    "#...#\n"
    "#..G#\n"
    "#####"
)

PATH = [Action.Down, Action.Down, Action.Right, Action.Right]   # (1,1) -> goal (3,3)


def walk(actions):
    """Real transitions from the real engine for the player's scripted path."""
    config = ripper.WorldConfig(max_tick=50, player_perception_range=3)
    world = ripper.World(LAYOUT, config=config, seed=42)
    obs = world.reset().observation
    pid = obs.player_id
    current = {e.id: e for e in obs.entities}[pid]

    out = []
    for a in actions:
        ts = world.step({pid: a})
        nxt = {e.id: e for e in ts.observation.entities}[pid]
        out.append(Transition(current, a, nxt, ts.terminated, ts.truncated))
        current = nxt
    return out


def make(reward_fn=None, **kw):
    return LinearQAgent(reward_fn or SparseGoalReward(), **kw)


class ConstantReward(RewardFn):
    name = "constant"
    def __init__(self, value):
        self.value = value
        self.resets = 0
    def reset(self):
        self.resets += 1
    def __call__(self, t):
        return self.value


def phi(obs):
    return np.append(features(obs), 1.0)


def test_learns_from_the_goal_transition():
    ts = walk(PATH)
    assert ts[-1].terminated and ts[-1].next_obs.status == "GoalReached"

    agent = make(alpha=0.5, gamma=0.9, epsilon=0.0)
    agent.on_transition(ts[-1])

    right = ACTIONS.index(Action.Right)
    assert np.allclose(agent.W[right], 0.5 * 1.0 * phi(ts[-1].obs))   # delta = 1 - 0
    others = [i for i in range(len(ACTIONS)) if i != right]
    assert np.all(agent.W[others] == 0.0)


def test_zero_reward_from_zero_weights_changes_nothing():
    agent = make(alpha=0.5, gamma=0.9)
    agent.on_transition(walk(PATH)[0])
    assert np.all(agent.W == 0.0)


def test_truncation_does_not_cut_the_bootstrap_but_termination_does():
    t0 = walk(PATH)[0]                         # Down, reward 0, not terminal
    down = ACTIONS.index(Action.Down)

    agent = make(alpha=0.5, gamma=0.9)
    agent.W[0, -1] = 2.0                       # Q(any state, action 0) = 2.0
    agent.on_transition(dataclasses.replace(t0, truncated=True))
    # target = 0 + 0.9 * 2.0 = 1.8 ; delta = 1.8 ; W[down] = 0.5 * 1.8 * phi
    assert np.allclose(agent.W[down], 0.9 * phi(t0.obs))

    agent = make(alpha=0.5, gamma=0.9)
    agent.W[0, -1] = 2.0
    agent.on_transition(dataclasses.replace(t0, terminated=True))
    assert np.all(agent.W[down] == 0.0)        # target 0, prediction 0


def test_uses_its_own_reward_function():
    t0 = dataclasses.replace(walk(PATH)[0], terminated=True)
    down = ACTIONS.index(Action.Down)

    agent = make(ConstantReward(5.0), alpha=0.5, gamma=0.9)
    agent.on_transition(t0)
    assert np.allclose(agent.W[down], 0.5 * 5.0 * phi(t0.obs))   # delta = 5 - 0


def test_select_action_is_greedy_with_epsilon_zero():
    obs = walk(PATH)[0].obs
    agent = make(epsilon=0.0)
    agent.W[2, -1] = 1.0                       # action 2 has the highest Q everywhere
    assert all(agent.select_action(obs) == ACTIONS[2] for _ in range(50))


def test_evaluation_mode_is_greedy_and_frozen():
    ts = walk(PATH)
    agent = make(epsilon=1.0, alpha=0.5)       # would be fully random if learning
    agent.W[2, -1] = 1.0
    agent.learning = False

    assert all(agent.select_action(ts[0].obs) == ACTIONS[2] for _ in range(50))

    before = agent.W.copy()
    agent.on_transition(ts[-1])                # the goal transition: would change W
    assert np.array_equal(agent.W, before)


def test_reset_keeps_weights_and_resets_the_reward():
    reward = ConstantReward(1.0)
    agent = make(reward)
    agent.W[0, 0] = 3.0
    agent.reset()
    assert reward.resets == 1
    assert agent.W[0, 0] == 3.0


def run():
    test_learns_from_the_goal_transition()
    test_zero_reward_from_zero_weights_changes_nothing()
    test_truncation_does_not_cut_the_bootstrap_but_termination_does()
    test_uses_its_own_reward_function()
    test_select_action_is_greedy_with_epsilon_zero()
    test_evaluation_mode_is_greedy_and_frozen()
    test_reset_keeps_weights_and_resets_the_reward()
    print("linear q agent: all checks passed")


if __name__ == "__main__":
    run()