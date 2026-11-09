"""
Created on Fri Nov  6 08:51:07 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Tests of poet/transfer.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import copy

import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import transfer

from poet.entry import Entry
from poet.transfer import role_rate, transfer_into, transfer_round


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


###############################################################################
# Stand-ins: an agent has a skill (its goal rate), a learner holds one agent
###############################################################################

class Agent:
    def __init__(self, name, skill):
        self.name, self.skill = name, skill

    def clone(self):
        return Agent(self.name, self.skill)


class Learner:
    """agent() gives the SAME object every time, like a learner that holds its agent."""
    def __init__(self, name, skill):
        self._agent = Agent(name, skill)

    def agent(self):
        return self._agent

    def train(self, layout, opponent, rng):
        pass

    def clone(self):
        return copy.deepcopy(self)


def make_entry(layout, player, enemy, name=""):
    return Entry(layout, {"player": Learner(f"{name}p", player), "enemy_0": Learner(f"{name}e", enemy)})


@contextmanager
def games():
    """Replace play_games: every role wins skill * n_games of the games. Yields the calls made."""
    calls = []

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        calls.append(dict(layout=layout, agents=dict(agents_by_role), n_games=n_games, first_seed=first_seed))
        return {role: ["GoalReached"] * round(agent.skill * n_games) + ["Timeout"] * (n_games - round(agent.skill * n_games))
                for role, agent in agents_by_role.items()}

    with patched(transfer, play_games=fake_play):
        yield calls


###############################################################################
# role_rate
###############################################################################

def test_role_rate_is_the_goal_rate_of_the_agent_in_its_role():
    entry = make_entry("MINE", 0.1, 0.9)
    with games():
        assert role_rate(entry, "player", Agent("x", 0.3), 10, 500) == 0.3
        assert role_rate(entry, "enemy_0", Agent("y", 0.7), 10, 500) == 0.7


def test_role_rate_plays_the_entry_layout_against_the_entry_own_opponent_on_the_given_games():
    entry = make_entry("MINE", 0.1, 0.9)
    with games() as calls:
        role_rate(entry, "player", Agent("x", 0.3), 10, 500)
        role_rate(entry, "enemy_0", Agent("y", 0.7), 10, 500)

    player_call, enemy_call = calls
    assert player_call["layout"] == "MINE" and (player_call["n_games"], player_call["first_seed"]) == (10, 500)
    assert {r: a.name for r, a in player_call["agents"].items()} == {"player": "x", "enemy_0": "e"}
    assert {r: a.name for r, a in enemy_call["agents"].items()} == {"player": "p", "enemy_0": "y"}


def test_role_rate_plays_clones_and_never_the_agents_themselves():
    entry = make_entry("MINE", 0.1, 0.9)
    agent = Agent("x", 0.3)
    with games() as calls:
        role_rate(entry, "player", agent, 10, 500)

    played = calls[0]["agents"]
    assert played["player"] is not agent
    assert played["enemy_0"] is not entry.learners["enemy_0"].agent()


###############################################################################
# transfer_into
###############################################################################

def test_the_best_candidate_is_adopted_in_each_role_on_its_own():
    entry = make_entry("MINE", 0.1, 0.2, "own")
    candidates = [make_entry("A", 0.5, 0.1, "a"), make_entry("B", 0.3, 0.6, "b")]

    with games():
        report = transfer_into(entry, candidates, 10, 500)

    assert entry.learners["player"].agent().name == "ap"          # 0.5 beats 0.3 and 0.1
    assert entry.learners["enemy_0"].agent().name == "be"         # 0.6 beats 0.1 and 0.2
    assert report["player"] == {"own": 0.1, "best": 0.5, "from": 0, "adopted": True}
    assert report["enemy_0"] == {"own": 0.2, "best": 0.6, "from": 1, "adopted": True}


def test_a_role_without_a_better_candidate_keeps_its_learner():
    entry = make_entry("MINE", 0.4, 0.2, "own")
    candidates = [make_entry("A", 0.5, 0.1, "a")]
    own_enemy = entry.learners["enemy_0"]

    with games():
        report = transfer_into(entry, candidates, 10, 500)

    assert entry.learners["player"].agent().name == "ap"
    assert entry.learners["enemy_0"] is own_enemy
    assert report["enemy_0"] == {"own": 0.2, "best": 0.1, "from": 0, "adopted": False}


def test_an_equal_candidate_is_not_adopted():
    entry = make_entry("MINE", 0.4, 0.4, "own")
    own = dict(entry.learners)

    with games():
        report = transfer_into(entry, [make_entry("A", 0.4, 0.4, "a")], 10, 500)

    assert entry.learners == own
    assert not report["player"]["adopted"] and not report["enemy_0"]["adopted"]


def test_the_margin_asks_for_a_clear_gain_and_the_limit_itself_is_not_enough():
    def adopted(margin):
        entry = make_entry("MINE", 0.4, 0.4, "own")
        with games():
            return transfer_into(entry, [make_entry("A", 0.5, 0.4, "a")], 10, 500, margin)["player"]["adopted"]

    assert adopted(0.0) is True
    assert adopted(0.09) is True
    assert adopted(0.1) is False          # 0.5 is not MORE than 0.4 + 0.1
    assert adopted(0.2) is False


def test_among_equal_candidates_the_first_one_wins():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    candidates = [make_entry("A", 0.5, 0.5, "a"), make_entry("B", 0.5, 0.5, "b"), make_entry("C", 0.2, 0.2, "c")]

    with games():
        report = transfer_into(entry, candidates, 10, 500)

    assert report["player"]["from"] == 0 and report["enemy_0"]["from"] == 0
    assert entry.learners["player"].agent().name == "ap"


def test_the_adopted_learner_is_a_clone_and_the_candidates_are_left_alone():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    candidate = make_entry("A", 0.5, 0.5, "a")
    before = {role: (learner, learner.agent().skill) for role, learner in candidate.learners.items()}

    with games():
        transfer_into(entry, [candidate], 10, 500)

    for role in candidate.learners:
        assert entry.learners[role] is not candidate.learners[role]          # a clone, not the same learner
        assert entry.learners[role].agent() is not candidate.learners[role].agent()
        assert candidate.learners[role] is before[role][0]                  # the candidate still has its own
        assert candidate.learners[role].agent().skill == before[role][1]


def test_training_the_adopted_learner_does_not_change_the_candidate():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    candidate = make_entry("A", 0.5, 0.5, "a")

    with games():
        transfer_into(entry, [candidate], 10, 500)

    entry.learners["player"].agent().skill = 0.99
    assert candidate.learners["player"].agent().skill == 0.5


def test_everyone_is_measured_on_the_same_games_of_the_entry_layout():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    candidates = [make_entry("A", 0.5, 0.5, "a"), make_entry("B", 0.3, 0.3, "b")]

    with games() as calls:
        transfer_into(entry, candidates, 10, 777)

    assert len(calls) == 6                                            # (own + 2 candidates) x 2 roles
    assert all(c["layout"] == "MINE" for c in calls)                  # never a candidate's layout
    assert all((c["n_games"], c["first_seed"]) == (10, 777) for c in calls)


def test_the_enemy_is_measured_against_the_player_the_entry_has_after_its_own_decision():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    candidates = [make_entry("A", 0.5, 0.5, "a")]

    with games() as calls:
        transfer_into(entry, candidates, 10, 500)

    player_calls, enemy_calls = calls[:2], calls[2:]
    # in the player's measurements the enemy is still the entry's own ...
    assert all(c["agents"]["enemy_0"].name == "owne" for c in player_calls)
    # ... and in the enemy's measurements the player is the ADOPTED one ("ap"), not the old "ownp"
    assert all(c["agents"]["player"].name == "ap" for c in enemy_calls)


def test_the_enemy_keeps_being_measured_against_the_old_player_when_the_player_did_not_change():
    entry = make_entry("MINE", 0.6, 0.1, "own")
    candidates = [make_entry("A", 0.5, 0.5, "a")]

    with games() as calls:
        transfer_into(entry, candidates, 10, 500)

    assert all(c["agents"]["player"].name == "ownp" for c in calls[2:])


def test_without_candidates_nothing_is_played_and_nothing_changes():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    own = dict(entry.learners)

    with games() as calls:
        report = transfer_into(entry, [], 10, 500)

    assert calls == [] and entry.learners == own
    for role in own:
        assert report[role] == {"own": None, "best": None, "from": None, "adopted": False}


def test_a_negative_margin_is_refused():
    with raises(ValueError, match="margin"):
        transfer_into(make_entry("MINE", 0.1, 0.1), [], 10, 500, margin=-0.1)


def test_the_entry_own_agents_are_not_moved_by_the_measuring():
    entry = make_entry("MINE", 0.1, 0.1, "own")
    agents = {role: learner.agent() for role, learner in entry.learners.items()}
    states = {role: vars(agent).copy() for role, agent in agents.items()}

    with games():
        transfer_into(entry, [make_entry("A", 0.0, 0.0, "a")], 10, 500)

    for role, agent in agents.items():
        assert entry.learners[role].agent() is agent and vars(agent) == states[role]


###############################################################################
# transfer_round
###############################################################################

def test_every_entry_looks_at_all_the_others_and_never_at_itself():
    pool = [make_entry("L0", 0.1, 0.1, "x"), make_entry("L1", 0.2, 0.2, "y"), make_entry("L2", 0.3, 0.3, "z")]

    with games() as calls:
        transfer_round(pool, 10, 500)

    # the first entry, player role: its own agent, then the agents of the two others, on ITS layout
    first = [c for c in calls if c["layout"] == "L0"][:3]
    assert [c["agents"]["player"].name for c in first] == ["xp", "yp", "zp"]


def test_the_reports_come_in_the_order_of_the_pool():
    pool = [make_entry("L0", 0.1, 0.1, "x"), make_entry("L1", 0.9, 0.9, "y")]

    with games():
        reports = transfer_round(pool, 10, 500)

    assert len(reports) == 2
    assert reports[0]["player"]["adopted"] is True and reports[0]["player"]["from"] == 0
    assert reports[1]["player"]["adopted"] is False


def test_a_swap_does_not_make_the_entries_swap_back_and_forth():
    pool = [make_entry("L0", 0.1, 0.1, "x"), make_entry("L1", 0.9, 0.9, "y")]

    with games():
        transfer_round(pool, 10, 500)
        again = transfer_round(pool, 10, 500)

    assert all(not r[role]["adopted"] for r in again for role in r)
    assert [e.learners["player"].agent().name for e in pool] == ["yp", "yp"]


def test_a_pool_of_one_entry_plays_nothing():
    pool = [make_entry("L0", 0.1, 0.1)]

    with games() as calls:
        reports = transfer_round(pool, 10, 500)

    assert calls == []
    assert reports == [{role: {"own": None, "best": None, "from": None, "adopted": False} for role in ("player", "enemy_0")}]


def test_the_layouts_stay_with_their_entries():
    pool = [make_entry("L0", 0.1, 0.1), make_entry("L1", 0.9, 0.9)]

    with games():
        transfer_round(pool, 10, 500)

    assert [entry.layout for entry in pool] == ["L0", "L1"]


###############################################################################
# The real engine (skipped when it cannot be imported)
###############################################################################

def test_real_heuristic_learners_transfer_and_the_answer_is_repeatable():
    try:
        from engine import GenerationConfig, generate_valid_layout
        from agents.heuristic_agent.agent import HeuristicAgent
    except ImportError as error:
        raise unittest.SkipTest(f"real engine not available here: {error}")

    class Fixed:
        def __init__(self, agent):
            self._agent = agent

        def agent(self):
            return self._agent

        def train(self, layout, opponent, rng):
            pass

        def clone(self):
            return Fixed(self._agent.clone())

    layout = generate_valid_layout(GenerationConfig(width=12, height=12, wall_density=0.1, num_enemies=1,
                                                    num_traps=0, min_player_goal_distance=6))

    def make_pool():
        return [Entry(layout, {"player": Fixed(HeuristicAgent(seed=s)), "enemy_0": Fixed(HeuristicAgent(seed=s + 10))})
                for s in range(3)]

    first = transfer_round(make_pool(), 10, 500_000)
    second = transfer_round(make_pool(), 10, 500_000)

    assert first == second
    assert all(0 <= report[role]["best"] <= 1 for report in first for role in report)


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