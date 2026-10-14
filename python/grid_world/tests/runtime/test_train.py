"""
Created on Mon Oct 12 07:52:09 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from engine import WorldConfig
from agents.base_agent import BaseAgent
from agents.linear_q_agent.agent import LinearQAgent
from core.actions import ACTIONS
from core.rewards import RewardFn
from runtime.train import train

EVAL_SEED = 10_000
FAST = WorldConfig(max_tick=30)


class SpyAgent(BaseAgent):
    """Random-acting agent that logs the `training` flag at the start of every episode."""
    name = "spy"

    def __init__(self, seed=42):
        super().__init__(seed=seed)          # BaseAgent sets self.training = True
        self.flags = []

    def select_action(self, obs):
        return ACTIONS[self.rng.integers(len(ACTIONS))]

    def on_reset(self):
        self.flags.append(self.training)


class ConstantReward(RewardFn):
    name = "constant"
    def __call__(self, t):
        return 1.0


def run_train(agent, **kw):
    params = dict(n_blocks=2, block_size=3, n_eval=4, train_seed=0,
                  eval_seed=EVAL_SEED, world_config=FAST)
    params.update(kw)
    return train(agent, **params)


def test_schedule_alternates_frozen_evaluation_and_training():
    spy = SpyAgent(seed=0)
    run_train(spy)

    # eval (4 episodes), then (train 3 + eval 4) x 2
    expected = ([False] * 4 + [True] * 3) * 2 + [False] * 4
    assert spy.flags == expected


def test_training_mode_is_restored():
    for initial in (True, False):
        spy = SpyAgent(seed=0)
        if initial:
            spy.set_train_mode()
        else:
            spy.set_eval_mode()
        run_train(spy)
        assert spy.training is initial


def test_result_shape_and_auc():
    result = run_train(SpyAgent(seed=0))

    assert list(result.checkpoints) == [0, 3, 6]
    assert len(result.eval_goal_rates) == 3
    assert len(result.eval_mean_steps) == 3
    assert len(result.train_goal_rates) == 2
    assert all(0.0 <= r <= 1.0 for r in result.eval_goal_rates)
    assert np.isclose(result.auc, np.mean(result.eval_goal_rates[1:]))


def test_bad_arguments_raise():
    cases = [
        (dict(block_size=0), ValueError),
        (dict(n_blocks=0), ValueError),
        (dict(n_eval=0), ValueError),
        # training seeds would run into the held-out range
        (dict(train_seed=9_999, n_blocks=1, block_size=2), ValueError),
    ]
    for kw, error in cases:
        try:
            run_train(SpyAgent(seed=0), **kw)
            assert False, f"{kw} must raise {error.__name__}"
        except error:
            pass


def test_real_learner_is_deterministic_and_actually_learns():
    def run():
        agent = LinearQAgent(ConstantReward(), alpha=0.02, seed=0)
        result = run_train(agent)
        return agent, result

    a1, r1 = run()
    a2, r2 = run()

    assert list(r1.eval_goal_rates) == list(r2.eval_goal_rates)
    assert np.array_equal(a1.W, a2.W)
    assert not np.all(a1.W == 0.0)          # training changed the weights


def run():
    test_schedule_alternates_frozen_evaluation_and_training()
    test_training_mode_is_restored()
    test_result_shape_and_auc()
    test_bad_arguments_raise()
    test_real_learner_is_deterministic_and_actually_learns()
    print("train: all checks passed")


if __name__ == "__main__":
    run()