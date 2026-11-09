"""
Created on Mon Nov  2 08:42:20 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################


"""Tests of poet/games.py, one group per function.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import sys
import traceback
import unittest
from contextlib import contextmanager

from poet.games import split_roles

###############################################################################
# Small helpers
###############################################################################

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


class Agent:
    """Anything can take a role: split_roles only sorts."""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Agent({self.name})"


###############################################################################
# split_roles
###############################################################################

def test_split_roles_only_player():
    player = Agent("p")
    assert split_roles({"player": player}) == (player, None)


def test_split_roles_player_and_one_enemy():
    player, enemy = Agent("p"), Agent("e0")
    got_player, got_enemies = split_roles({"player": player, "enemy_0": enemy})
    assert got_player is player and got_enemies == [enemy] and got_enemies[0] is enemy


def test_split_roles_enemies_come_in_order_of_their_number_not_of_the_dict():
    player, e0, e1, e2 = Agent("p"), Agent("e0"), Agent("e1"), Agent("e2")
    _, enemies = split_roles({"enemy_2": e2, "player": player, "enemy_0": e0, "enemy_1": e1})
    assert enemies == [e0, e1, e2]


def test_split_roles_needs_a_player():
    with raises(ValueError, match="player"):
        split_roles({"enemy_0": Agent("e0")})
    with raises(ValueError, match="player"):
        split_roles({})


def test_split_roles_unknown_roles_are_rejected():
    player = Agent("p")
    for bad in ("boss", "Player", "enemy", "enemy_x", "enemy_01", "enemy_-1", "enemy_ 0", 0, None):
        with raises(ValueError, match="unknown role"):
            split_roles({"player": player, bad: Agent("x")})


def test_split_roles_an_enemy_number_beyond_the_number_of_agents_is_unknown():
    with raises(ValueError, match="unknown role"):
        split_roles({"player": Agent("p"), "enemy_0": Agent("e0"), "enemy_5": Agent("e5")})


def test_split_roles_gaps_in_the_enemy_numbers_are_rejected():
    with raises(ValueError, match=r"enemy_0"):
        split_roles({"player": Agent("p"), "enemy_1": Agent("e1")})
    with raises(ValueError, match=r"enemy_1"):
        split_roles({"player": Agent("p"), "enemy_0": Agent("e0"), "enemy_2": Agent("e2")})


def test_split_roles_does_not_modify_its_argument():
    player, enemy = Agent("p"), Agent("e0")
    roles = {"enemy_0": enemy, "player": player}
    split_roles(roles)
    assert list(roles) == ["enemy_0", "player"] and roles["player"] is player


###############################################################################
# Runner
###############################################################################

def run_all():
    tests = [(name, test) for name, test in globals().items() if name.startswith("test_") and callable(test)]
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