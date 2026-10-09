"""
Created on Fri Oct  9 19:51:09 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from types import SimpleNamespace

from core.transition import Transition
from core.reward import RewardFn, SparseGoalReward


def make_transition(next_status, terminated=False, truncated=False, kind="player"):
    """A Transition whose only meaningful content is the entity's next status."""
    obs = SimpleNamespace(status="Alive", kind=kind)
    next_obs = SimpleNamespace(status=next_status, kind=kind)
    return Transition(obs, None, next_obs, terminated, truncated)


def test_contract():
    # RewardFn is abstract: it cannot be instantiated directly.
    try:
        RewardFn()
        assert False, "RewardFn must be abstract"
    except TypeError:
        pass

    reward = SparseGoalReward()
    assert isinstance(reward, RewardFn)
    assert reward.name == "sparse_goal"

def test_goal_reached_pays_one():
    reward = SparseGoalReward()
    assert reward(make_transition("GoalReached", terminated=True)) == 1.0


def test_other_terminal_statuses_pay_zero():
    reward = SparseGoalReward()
    assert reward(make_transition("Trapped", terminated=True)) == 0.0
    assert reward(make_transition("Caught", terminated=True)) == 0.0


def test_ordinary_step_pays_zero():
    reward = SparseGoalReward()
    assert reward(make_transition("Alive")) == 0.0


def test_timeout_pays_zero():
    reward = SparseGoalReward()
    assert reward(make_transition("Alive", truncated=True)) == 0.0


def test_bystander_when_episode_ends_pays_zero():
    # An enemy that did not catch the player: still Alive, but the episode ended.
    reward = SparseGoalReward()
    assert reward(make_transition("Alive", terminated=True, kind="enemy")) == 0.0


def test_same_rule_for_enemy_role():
    # An enemy that caught the player has GoalReached: same rule, same payout.
    reward = SparseGoalReward()
    assert reward(make_transition("GoalReached", terminated=True, kind="enemy")) == 1.0
    assert reward(make_transition("Trapped", terminated=True, kind="enemy")) == 0.0


def test_return_type_is_float():
    reward = SparseGoalReward()
    for status in ("GoalReached", "Alive", "Trapped", "Caught"):
        value = reward(make_transition(status))
        assert type(value) is float, f"{status}: got {type(value)}"


def test_reset_is_a_harmless_noop():
    reward = SparseGoalReward()
    assert reward.reset() is None
    assert reward(make_transition("GoalReached", terminated=True)) == 1.0


def test_pure_function_of_the_transition():
    reward = SparseGoalReward()
    t = make_transition("GoalReached", terminated=True)
    assert reward(t) == reward(t) == 1.0


def run():
    test_contract()
    test_goal_reached_pays_one()
    test_other_terminal_statuses_pay_zero()
    test_ordinary_step_pays_zero()
    test_timeout_pays_zero()
    test_bystander_when_episode_ends_pays_zero()
    test_same_rule_for_enemy_role()
    test_return_type_is_float()
    test_reset_is_a_harmless_noop()
    test_pure_function_of_the_transition()
    print("rewards: all checks passed")


if __name__ == "__main__":
    run()