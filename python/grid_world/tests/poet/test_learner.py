"""
Created on Tue Nov  3 14:39:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Tests of poet/learner.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.

check_contract(learner) is written so that ANY learner (the NEAT one later) can be run through it.
"""

import copy
import importlib
import random
import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import learner as learner_module

importlib.reload(learner_module)   # pick up edits without restarting the console
from poet.learner import Learner

@contextmanager
def raises(exception, match=None):
    """The block must raise `exception` (with `match` inside the message when given)."""
    try:
        yield
    except exception as error:
        if match is not None and match not in str(error):
            raise AssertionError(f"{exception.__name__} message {str(error)!r} does not contain {match!r}")
    else:
        raise AssertionError(f"{exception.__name__} was not raised")


###############################################################################
# A fake learner (no NEAT, no engine) and the contract any learner must keep
###############################################################################

class FakeAgent:
    def __init__(self, skill):
        self.skill = skill


class FakeLearner:
    """Its 'skill' goes up by a random amount every time it trains."""
    def __init__(self, skill=0.0):
        self.skill = skill
        self.trained_on = []

    def agent(self):
        return FakeAgent(self.skill)

    def train(self, layout, opponent, rng):
        self.skill += rng.random()
        self.trained_on.append((layout, opponent))

    def clone(self):
        return copy.deepcopy(self)


def check_contract(learner, layout, opponent):
    """Raise AssertionError when `learner` does not keep what POET relies on. The learner is
    modified (it is trained a few times)."""
    assert isinstance(learner, Learner), "agent, train and clone are all needed"

    # agent(): playing with it, or asking twice, changes nothing
    first = learner.agent()
    assert first is not None
    assert learner.agent() is not None

    # train(): returns None and takes a layout, an opponent and an rng
    assert learner.train(layout, opponent, random.Random(0)) is None

    # clone(): a different object, with the same agent for now
    twin = learner.clone()
    assert twin is not learner
    assert twin.agent() is not learner.agent()                 # never the same agent object
    assert vars(twin.agent()) == vars(learner.agent()), "the clone starts as good as the original"

    # independence, both ways
    before = vars(learner.agent()).copy()
    for _ in range(3):
        twin.train(layout, opponent, random.Random(1))
    assert vars(learner.agent()) == before, "training the clone changed the original"

    before_twin = vars(twin.agent()).copy()
    for _ in range(3):
        learner.train(layout, opponent, random.Random(2))
    assert vars(twin.agent()) == before_twin, "training the original changed the clone"


###############################################################################
# Learner
###############################################################################

def test_a_class_with_the_three_methods_is_a_learner_without_inheriting_anything():
    assert issubclass(FakeLearner, object) and Learner not in FakeLearner.__mro__
    assert isinstance(FakeLearner(), Learner)


def test_a_class_missing_a_method_is_not_a_learner():
    class NoClone:
        def agent(self): ...
        def train(self, layout, opponent, rng): ...

    class NoTrain:
        def agent(self): ...
        def clone(self): ...

    class NoAgent:
        def train(self, layout, opponent, rng): ...
        def clone(self): ...

    for incomplete in (NoClone(), NoTrain(), NoAgent(), object()):
        assert not isinstance(incomplete, Learner), type(incomplete).__name__


def test_the_protocol_names_the_three_methods_and_nothing_about_a_budget():
    methods = {name for name in vars(Learner) if not name.startswith("_")}
    assert methods == {"agent", "train", "clone"}


def test_the_signatures_are_the_ones_poet_calls():
    import inspect
    names = lambda method: list(inspect.signature(method).parameters)
    assert names(Learner.agent) == ["self"]
    assert names(Learner.train) == ["self", "layout", "opponent", "rng"]     # no budget: the learner owns it
    assert names(Learner.clone) == ["self"]


def test_the_learner_module_knows_nothing_about_neat():
    import inspect
    source = inspect.getsource(learner_module).lower()
    assert "neat" not in source                                 # no import, no vocabulary


###############################################################################
# The contract
###############################################################################

def test_a_good_learner_passes_the_contract():
    check_contract(FakeLearner(), layout="LAYOUT", opponent=FakeAgent(1.0))


def test_the_contract_catches_a_clone_that_shares_state_with_the_original():
    class Shallow(FakeLearner):
        def clone(self):
            return self                        # not a copy at all

    with raises(AssertionError):
        check_contract(Shallow(), "LAYOUT", FakeAgent(1.0))


def test_the_contract_catches_a_clone_that_shares_its_agent_state():
    shared = FakeAgent(0.0)

    class SharedAgent(FakeLearner):
        def agent(self):
            return shared

        def train(self, layout, opponent, rng):
            shared.skill += rng.random()

        def clone(self):
            return SharedAgent()

    with raises(AssertionError):
        check_contract(SharedAgent(), "LAYOUT", FakeAgent(1.0))


def test_the_contract_catches_a_train_that_returns_something():
    class Returns(FakeLearner):
        def train(self, layout, opponent, rng):
            super().train(layout, opponent, rng)
            return self.skill

    with raises(AssertionError):
        check_contract(Returns(), "LAYOUT", FakeAgent(1.0))


def test_the_contract_catches_an_object_that_is_not_a_learner():
    with raises(AssertionError, match="all needed"):
        check_contract(object(), "LAYOUT", FakeAgent(1.0))


def test_the_contract_catches_a_clone_that_is_worse_than_the_original():
    class Forgets(FakeLearner):
        def clone(self):
            return Forgets(0.0)

    learner = Forgets()
    with raises(AssertionError, match="starts as good"):
        check_contract(learner, "LAYOUT", FakeAgent(1.0))


###############################################################################
# Runner
###############################################################################

def run_all():
    # Only the tests defined in THIS file: a console that stays open still holds the tests of earlier runs
    tests = [(name, test) for name, test in globals().items()
             if name.startswith("test_") and callable(test) ]
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