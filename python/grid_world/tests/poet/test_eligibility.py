"""
Created on Wed Nov  4 16:13:48 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Tests of poet/eligibility.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import copy

import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import eligibility


from poet.eligibility import is_eligible
from poet.entry import Entry


@contextmanager
def patched(module, **replacements):
    """Replace attributes of `module` and ALWAYS put the originals back, even when a test fails."""
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


class FakeAgent:
    def __init__(self, name, state=0):
        self.name, self.state = name, state

    def clone(self):
        return FakeAgent(self.name, self.state)


class FakeLearner:
    def __init__(self, name):
        self.name, self.trained = name, 0

    def agent(self):
        return FakeAgent(self.name)

    def train(self, layout, opponent, rng):
        self.trained += 1

    def clone(self):
        return copy.deepcopy(self)


def make_entry(layout="LAYOUT"):
    return Entry(layout, {"player": FakeLearner("p"), "enemy_0": FakeLearner("e")})


@contextmanager
def stand_ins(outcomes="OUTCOMES", answer=(True, {"player": 0.5})):
    """play_games and passes_minimal_criterion replaced by recorders."""
    calls = {"play": [], "criterion": []}

    def fake_play(layout, agents, **kwargs):
        calls["play"].append((layout, agents, kwargs))
        return outcomes

    def fake_criterion(received, **limits):
        calls["criterion"].append((received, limits))
        return answer

    with patched(eligibility, play_games=fake_play, passes_minimal_criterion=fake_criterion):
        yield calls


###############################################################################
# is_eligible
###############################################################################

def test_it_plays_the_layout_of_the_entry_with_its_agents_and_the_given_seeds():
    entry = make_entry("MY LAYOUT")
    with stand_ins() as calls:
        is_eligible(entry, 60, 230_000)
    assert len(calls["play"]) == 1
    layout, agents, kwargs = calls["play"][0]
    assert layout == "MY LAYOUT"
    assert set(agents) == {"player", "enemy_0"} and agents["player"].name == "p"
    assert kwargs == {"n_games": 60, "first_seed": 230_000}


def test_it_plays_clones_of_the_agents_and_not_the_agents_of_the_entry():
    played = []

    class SpyLearner(FakeLearner):
        def __init__(self, name):
            super().__init__(name)
            self.agent_object = FakeAgent(name, state=5)

        def agent(self):
            return self.agent_object

    spy = Entry("LAYOUT", {"player": SpyLearner("p"), "enemy_0": SpyLearner("e")})
    with patched(eligibility, play_games=lambda layout, agents, **kw: played.append(agents) or "OUTCOMES",
                 passes_minimal_criterion=lambda outcomes, **limits: (True, {})):
        is_eligible(spy, 10, 0)
    for role, agent in played[0].items():
        assert agent is not spy.learners[role].agent_object                     # a copy ...
        assert (agent.name, agent.state) == (spy.learners[role].agent_object.name, 5)     # ... of the same agent


def test_measuring_twice_gives_the_same_answer_and_leaves_the_agents_alone():
    class Drifting(FakeAgent):
        """Plays differently every time it plays: its state moves, like a random generator does."""
        def clone(self):
            return Drifting(self.name, self.state)

    class DriftLearner(FakeLearner):
        def __init__(self, name):
            super().__init__(name)
            self.agent_object = Drifting(name)

        def agent(self):
            return self.agent_object

    def drifting_games(layout, agents, n_games, first_seed):
        agents["player"].state += 1                                            # playing moves the agent
        wins = 3 + agents["player"].state                                      # and the result depends on where it is
        return {"player": ["GoalReached"] * wins + ["Caught"] * (10 - wins),
                "enemy_0": ["GoalReached"] * 3 + ["Alive"] * 7}

    entry = Entry("LAYOUT", {"player": DriftLearner("p"), "enemy_0": DriftLearner("e")})
    with patched(eligibility, play_games=drifting_games):
        first = is_eligible(entry, 10, 0)
        second = is_eligible(entry, 10, 0)
    assert first == second
    assert entry.learners["player"].agent_object.state == 0                      # the entry's agent never moved


def test_it_gives_the_outcomes_to_the_criterion_and_returns_its_answer():
    answer = (False, {"player": 0.9, "enemy_0": 0.1, "timeout": 0.0})
    with stand_ins(outcomes="THE OUTCOMES", answer=answer) as calls:
        result = is_eligible(make_entry(), 10, 0)
    assert calls["criterion"] == [("THE OUTCOMES", {})]
    assert result is answer


def test_the_limits_are_passed_to_the_criterion_and_only_those():
    with stand_ins() as calls:
        is_eligible(make_entry(), 10, 0, low=0.1, high=0.9, timeout_cap=0.5)
        is_eligible(make_entry(), 10, 0)
    assert calls["criterion"][0][1] == {"low": 0.1, "high": 0.9, "timeout_cap": 0.5}
    assert calls["criterion"][1][1] == {}                # defaults stay the criterion's own


def test_an_unknown_limit_is_an_error():
    def balanced(layout, agents, n_games, first_seed):
        return {"player": ["GoalReached"] * 4 + ["Caught"] * 6, "enemy_0": ["GoalReached"] * 3 + ["Alive"] * 7}

    with patched(eligibility, play_games=balanced):          # the real criterion decides what is a limit
        try:
            is_eligible(make_entry(), 10, 0, lower=0.1)
        except TypeError:
            pass
        else:
            raise AssertionError("TypeError was not raised")


def test_it_does_not_train_or_change_the_entry():
    entry = make_entry()
    with stand_ins():
        is_eligible(entry, 10, 0)
    assert all(learner.trained == 0 for learner in entry.learners.values())
    assert entry.layout == "LAYOUT" and set(entry.learners) == {"player", "enemy_0"}


def test_a_different_first_seed_plays_different_games():
    with stand_ins() as calls:
        is_eligible(make_entry(), 10, 5)
        is_eligible(make_entry(), 10, 99)
    assert [call[2]["first_seed"] for call in calls["play"]] == [5, 99]


def test_with_the_real_criterion_a_balanced_setting_is_eligible():
    def balanced(layout, agents, n_games, first_seed):
        return {"player": ["GoalReached"] * 4 + ["Caught"] * 6,
                "enemy_0": ["GoalReached"] * 3 + ["Alive"] * 7}

    with patched(eligibility, play_games=balanced):
        eligible, report = is_eligible(make_entry(), 10, 0)
    assert eligible is True and report == {"player": 0.4, "enemy_0": 0.3, "timeout": 0.0}


def test_with_the_real_criterion_an_easy_setting_is_not_eligible_and_limits_can_change_that():
    def easy(layout, agents, n_games, first_seed):
        return {"player": ["GoalReached"] * 9 + ["Caught"], "enemy_0": ["GoalReached"] * 3 + ["Alive"] * 7}

    with patched(eligibility, play_games=easy):
        assert is_eligible(make_entry(), 10, 0)[0] is False
        assert is_eligible(make_entry(), 10, 0, high=0.95)[0] is True


###############################################################################
# The real engine (not run in my sandbox: it needs your engine)
###############################################################################

def test_real_is_eligible_returns_an_answer_and_its_report():
    try:
        from engine import GenerationConfig, generate_valid_layout
        from agents.heuristic_agent.agent import HeuristicAgent
    except ImportError:
        raise unittest.SkipTest("the real engine is not available")

    class Fixed:
        def __init__(self, agent):
            self._agent = agent

        def agent(self):
            return self._agent

        def train(self, layout, opponent, rng):
            pass

        def clone(self):
            return Fixed(copy.deepcopy(self._agent))

    layout = generate_valid_layout(GenerationConfig(width=16, height=16, wall_density=0.1, num_enemies=1,
                                                    num_traps=0, min_player_goal_distance=10))
    entry = Entry(layout, {"player": Fixed(HeuristicAgent()), "enemy_0": Fixed(HeuristicAgent())})
    eligible, report = is_eligible(entry, n_games=20, first_seed=230_000)
    assert eligible in (True, False)
    assert set(report) == {"player", "enemy_0", "timeout"}


###############################################################################
# Runner
###############################################################################

def run_all():
    # Only the tests defined in THIS file: a console that stays open still holds the tests of earlier runs
    tests = [(name, test) for name, test in globals().items()
             if name.startswith("test_") and callable(test)]
    passed, skipped, failed = 0, 0, []

    for name, test in tests:
        try:
            test()
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"skip  {name}  ({reason})")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")

    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed" + (f": {failed}" if failed else ""))

    try:
        get_ipython()                    # inside Spyder / IPython: just report
    except NameError:
        if failed:
            sys.exit(1)                  # from a plain terminal: a failing exit code


if __name__ == "__main__":
    run_all()