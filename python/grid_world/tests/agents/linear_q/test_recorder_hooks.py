"""
Created on Tue Oct 13 22:40:55 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from engine import WorldConfig
from agents.linear_q_agent.agent import LinearQAgent
from agents.linear_q_agent.diagnostics import Recorder, expected_q_change
from core.rewards import RewardFn
from runtime.evaluate import evaluate
from runtime.train import train, _frozen_eval

EVAL_SEED = 10_000
FAST = WorldConfig(max_tick=30)


class ConstantReward(RewardFn):
    name = "constant"
    def __call__(self, t):
        return 1.0


def make_agent():
    return LinearQAgent(ConstantReward(), alpha=0.02, seed=0)


def test_recorder_is_off_by_default():
    assert make_agent().recorder is None


def test_training_logs_one_update_per_decision():
    agent = make_agent()
    agent.recorder = Recorder()

    evaluate(agent, n_episodes=3, seed=0, generation_kwargs=None, world_config=FAST)

    assert len(agent.recorder.decisions) > 0
    assert len(agent.recorder.updates) == len(agent.recorder.decisions)
    assert all(row["training"] for row in agent.recorder.decisions)


def test_frozen_evaluation_logs_decisions_but_no_updates():
    agent = make_agent()
    agent.recorder = Recorder()
    before = agent.W.copy()

    _frozen_eval(agent, 3, EVAL_SEED, None, FAST)

    assert len(agent.recorder.decisions) > 0
    assert len(agent.recorder.updates) == 0
    assert all(not row["training"] for row in agent.recorder.decisions)
    assert np.array_equal(before, agent.W)


def test_logged_update_rows_obey_the_identity():
    agent = make_agent()
    agent.recorder = Recorder()

    evaluate(agent, n_episodes=3, seed=0, generation_kwargs=None, world_config=FAST)

    assert len(agent.recorder.updates) > 0
    for row in agent.recorder.updates:
        expected = expected_q_change(row["x"], row["delta"], agent.alpha)
        assert np.isclose(row["q_change"], expected, rtol=1e-3, atol=1e-7)


def test_recorder_does_not_change_behaviour():
    def run_training(with_recorder):
        agent = make_agent()
        if with_recorder:
            agent.recorder = Recorder()
        train(agent, n_blocks=2, block_size=3, n_eval=3, train_seed=0,
              eval_seed=EVAL_SEED, world_config=FAST)
        return agent

    plain = run_training(False)
    recorded = run_training(True)

    assert np.array_equal(plain.W, recorded.W)
    assert len(recorded.recorder.decisions) > 0


def run():
    test_recorder_is_off_by_default()
    test_training_logs_one_update_per_decision()
    test_frozen_evaluation_logs_decisions_but_no_updates()
    test_logged_update_rows_obey_the_identity()
    test_recorder_does_not_change_behaviour()
    print("recorder hooks: all checks passed")


if __name__ == "__main__":
    run()