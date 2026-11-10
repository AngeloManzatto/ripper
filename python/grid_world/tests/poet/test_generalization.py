"""
Created on Tue Nov 10 15:28:11 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Tests of poet/generalization.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.
"""

import sys
import traceback
import unittest
from contextlib import contextmanager

from poet import generalization

from poet.generalization import (held_out_layouts, player_outcomes_on_layouts, pressure_on_layouts,
                                 skill_on_layouts, summarize)


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
# Stand-ins: an agent has a skill, a layout has a difficulty
###############################################################################

class Agent:
    def __init__(self, name, skill):
        self.name, self.skill = name, skill
        self.clones = 0

    def clone(self):
        self.clones += 1
        return Agent(self.name, self.skill)


DIFFICULTY = {"easy": 1.0, "hard": 0.5}


@contextmanager
def games():
    """Replace play_games: the agent of the role wins skill * difficulty * n_games games.
    Yields the calls that were made."""
    calls = []

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        calls.append(dict(layout=layout, agents=dict(agents_by_role), n_games=n_games, first_seed=first_seed))
        outcomes = {}
        for role, agent in agents_by_role.items():
            wins = round(agent.skill * DIFFICULTY[layout] * n_games)
            outcomes[role] = ["GoalReached"] * wins + ["Timeout"] * (n_games - wins)
        return outcomes

    with patched(generalization, play_games=fake_play):
        yield calls


###############################################################################
# held_out_layouts
###############################################################################

def test_held_out_layouts_makes_one_layout_per_seed_in_order():
    seen = []

    def make(seed):
        seen.append(seed)
        return f"layout-{seed}"

    assert held_out_layouts(make, 3, 500) == ["layout-500", "layout-501", "layout-502"]
    assert seen == [500, 501, 502]


def test_held_out_layouts_is_repeatable():
    make = lambda seed: seed * 2
    assert held_out_layouts(make, 4, 10) == held_out_layouts(make, 4, 10)


def test_held_out_layouts_needs_at_least_one():
    with raises(ValueError, match="n_layouts"):
        held_out_layouts(lambda s: s, 0, 1)


###############################################################################
# skill_on_layouts
###############################################################################

def test_one_rate_per_layout_in_the_order_given():
    with games():
        rates = skill_on_layouts("player", Agent("a", 0.8), ["easy", "hard", "hard"], None, 10, 500)
        assert rates == [0.8, 0.4, 0.4]
        assert skill_on_layouts("player", Agent("a", 0.8), ["hard", "easy"], None, 10, 500) == [0.4, 0.8]


def test_the_player_can_play_against_the_engine_alone():
    with games() as calls:
        skill_on_layouts("player", Agent("a", 0.8), ["easy"], None, 10, 500)
    assert list(calls[0]["agents"]) == ["player"]


def test_an_opponent_takes_the_other_role_in_both_cases():
    with games() as calls:
        skill_on_layouts("player", Agent("a", 0.8), ["easy"], Agent("o", 0.3), 10, 500)
        skill_on_layouts("enemy_0", Agent("a", 0.8), ["easy"], Agent("o", 0.3), 10, 500)
    assert {r: a.name for r, a in calls[0]["agents"].items()} == {"player": "a", "enemy_0": "o"}
    assert {r: a.name for r, a in calls[1]["agents"].items()} == {"player": "o", "enemy_0": "a"}


def test_the_rate_is_the_one_of_the_role_measured_not_the_opponents():
    with games():
        assert skill_on_layouts("player", Agent("a", 0.8), ["easy"], Agent("o", 0.3), 10, 500) == [0.8]
        assert skill_on_layouts("enemy_0", Agent("a", 0.8), ["easy"], Agent("o", 0.3), 10, 500) == [0.8]
        assert skill_on_layouts("enemy_0", Agent("a", 0.2), ["easy"], Agent("o", 0.9), 10, 500) == [0.2]


def test_every_layout_gets_a_fresh_clone_of_the_agent_and_of_the_opponent():
    agent, opponent = Agent("a", 0.8), Agent("o", 0.3)
    with games() as calls:
        skill_on_layouts("player", agent, ["easy", "hard", "easy"], opponent, 10, 500)

    played = [c["agents"] for c in calls]
    assert all(p["player"] is not agent and p["enemy_0"] is not opponent for p in played)
    assert len({id(p["player"]) for p in played}) == 3 and len({id(p["enemy_0"]) for p in played}) == 3
    assert (agent.clones, opponent.clones) == (3, 3)


def test_every_layout_is_played_on_the_same_games():
    with games() as calls:
        skill_on_layouts("player", Agent("a", 0.8), ["easy", "hard", "easy"], None, 20, 700_000)
    assert [(c["layout"], c["n_games"], c["first_seed"]) for c in calls] == \
        [("easy", 20, 700_000), ("hard", 20, 700_000), ("easy", 20, 700_000)]


def test_two_agents_are_measured_on_the_same_games():
    with games() as calls:
        skill_on_layouts("player", Agent("a", 0.8), ["easy", "hard"], None, 20, 700_000)
        skill_on_layouts("player", Agent("b", 0.3), ["easy", "hard"], None, 20, 700_000)
    seeds = [(c["layout"], c["n_games"], c["first_seed"]) for c in calls]
    assert seeds[:2] == seeds[2:]


def test_the_agent_and_the_opponent_are_not_moved():
    agent, opponent = Agent("a", 0.8), Agent("o", 0.3)
    with games():
        skill_on_layouts("player", agent, ["easy", "hard"], opponent, 10, 500)
    assert (agent.name, agent.skill, opponent.name, opponent.skill) == ("a", 0.8, "o", 0.3)


def test_an_enemy_without_a_player_is_refused_before_anything_is_played():
    with games() as calls:
        with raises(ValueError, match="needs a player"):
            skill_on_layouts("enemy_0", Agent("a", 0.8), ["easy"], None, 10, 500)
    assert calls == []


def test_an_unknown_role_and_no_layouts_are_refused():
    with games() as calls:
        with raises(ValueError, match="role"):
            skill_on_layouts("enemy_1", Agent("a", 0.8), ["easy"], None, 10, 500)
        with raises(ValueError, match="no layouts"):
            skill_on_layouts("player", Agent("a", 0.8), [], None, 10, 500)
    assert calls == []


###############################################################################
# pressure_on_layouts
###############################################################################

class Enemy:
    """An enemy with a strength: the player of skill s wins s * (1 - strength) * difficulty."""
    def __init__(self, name, strength):
        self.name, self.strength = name, strength
        self.clones = 0

    def clone(self):
        self.clones += 1
        return Enemy(self.name, self.strength)


@contextmanager
def duels():
    """Replace play_games: the player's goal rate depends on the player AND on the enemy facing it."""
    calls = []

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        calls.append(dict(layout=layout, agents=dict(agents_by_role), n_games=n_games, first_seed=first_seed))
        player, enemy = agents_by_role["player"], agents_by_role["enemy_0"]
        wins = round(player.skill * (1 - enemy.strength) * DIFFICULTY[layout] * n_games)
        return {"player": ["GoalReached"] * wins + ["Timeout"] * (n_games - wins),
                "enemy_0": ["Alive"] * n_games}

    with patched(generalization, play_games=fake_play):
        yield calls


def close(values, expected):
    assert len(values) == len(expected) and all(abs(v - e) < 1e-9 for v, e in zip(values, expected)), \
        f"{values} != {expected}"


def test_pressure_is_the_share_of_games_the_player_fails_in():
    with duels():
        close(pressure_on_layouts(Enemy("e", 0.5), ["easy", "hard"], Agent("p", 0.8), 10, 500), [0.6, 0.8])


def test_a_stronger_enemy_gives_more_pressure_and_a_harmless_one_the_least():
    with duels():
        weak = pressure_on_layouts(Enemy("weak", 0.0), ["easy"], Agent("p", 0.8), 10, 500)
        strong = pressure_on_layouts(Enemy("strong", 0.75), ["easy"], Agent("p", 0.8), 10, 500)
    close(weak, [0.2])
    close(strong, [0.8])


def test_the_enemy_plays_the_enemy_role_and_the_fixed_player_the_player_role():
    with duels() as calls:
        pressure_on_layouts(Enemy("e", 0.5), ["easy"], Agent("p", 0.8), 10, 500)
    assert calls[0]["agents"]["player"].name == "p" and calls[0]["agents"]["enemy_0"].name == "e"


def test_pressure_needs_a_player_and_plays_nothing_without_one():
    with duels() as calls:
        with raises(ValueError, match="needs a player"):
            pressure_on_layouts(Enemy("e", 0.5), ["easy"], None, 10, 500)
    assert calls == []


def test_pressure_uses_fresh_clones_and_the_same_games_for_every_enemy():
    player, one, other = Agent("p", 0.8), Enemy("one", 0.2), Enemy("other", 0.6)
    with duels() as calls:
        pressure_on_layouts(one, ["easy", "hard"], player, 20, 700_000)
        pressure_on_layouts(other, ["easy", "hard"], player, 20, 700_000)

    assert [(c["layout"], c["n_games"], c["first_seed"]) for c in calls[:2]] == \
           [("easy", 20, 700_000), ("hard", 20, 700_000)]
    assert [(c["layout"], c["n_games"], c["first_seed"]) for c in calls[2:]] == \
           [("easy", 20, 700_000), ("hard", 20, 700_000)]
    assert all(c["agents"]["player"] is not player for c in calls)
    assert (player.clones, one.clones, other.clones) == (4, 2, 2)
    assert (one.strength, other.strength, player.skill) == (0.2, 0.6, 0.8)


###############################################################################
# player_outcomes_on_layouts
###############################################################################

SCRIPT = {   # what the player's 10 games end in, per layout
    "a": ["Caught"] * 5 + ["Timeout"] * 3 + ["GoalReached"] * 2,
    "b": ["Trapped"] * 1 + ["Timeout"] * 9,
}


@contextmanager
def scripted():
    """Replace play_games: the player's outcomes are the ones written in SCRIPT for the layout."""
    calls = []

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        calls.append(dict(layout=layout, agents=dict(agents_by_role), n_games=n_games, first_seed=first_seed))
        return {"player": list(SCRIPT[layout]), "enemy_0": ["Alive"] * len(SCRIPT[layout])}

    with patched(generalization, play_games=fake_play):
        yield calls


def test_outcomes_are_kept_apart_and_in_the_order_of_the_layouts():
    with scripted():
        rows = player_outcomes_on_layouts(Enemy("e", 0.5), ["a", "b"], Agent("p", 0.8), 10, 500)
    assert rows == [{"goal": 0.2, "caught": 0.5, "trapped": 0.0, "timeout": 0.3},
                    {"goal": 0.0, "caught": 0.0, "trapped": 0.1, "timeout": 0.9}]


def test_the_shares_of_a_layout_add_up_to_one():
    with scripted():
        rows = player_outcomes_on_layouts(Enemy("e", 0.5), ["a", "b"], Agent("p", 0.8), 10, 500)
    assert all(abs(sum(row.values()) - 1) < 1e-9 for row in rows)


def test_the_player_and_the_enemy_take_their_roles_with_fresh_clones_on_the_same_games():
    player, enemy = Agent("p", 0.8), Enemy("e", 0.5)
    with scripted() as calls:
        player_outcomes_on_layouts(enemy, ["a", "b"], player, 20, 700_000)

    assert [(c["layout"], c["n_games"], c["first_seed"]) for c in calls] == [("a", 20, 700_000), ("b", 20, 700_000)]
    assert all(c["agents"]["player"].name == "p" and c["agents"]["enemy_0"].name == "e" for c in calls)
    assert all(c["agents"]["player"] is not player and c["agents"]["enemy_0"] is not enemy for c in calls)
    assert (player.clones, enemy.clones) == (2, 2)


def test_player_outcomes_need_a_player_and_layouts_and_play_nothing_otherwise():
    with scripted() as calls:
        with raises(ValueError, match="needs a player"):
            player_outcomes_on_layouts(Enemy("e", 0.5), ["a"], None, 10, 500)
        with raises(ValueError, match="no layouts"):
            player_outcomes_on_layouts(Enemy("e", 0.5), [], Agent("p", 0.8), 10, 500)
    assert calls == []


def test_an_outcome_that_cannot_end_the_players_game_is_an_error_not_a_missing_share():
    SCRIPT["typo"] = ["GoalReached"] * 9 + ["GoalReched"]
    SCRIPT["alive"] = ["GoalReached"] * 9 + ["Alive"]
    try:
        with scripted():
            with raises(ValueError, match="GoalReched"):
                player_outcomes_on_layouts(Enemy("e", 0.5), ["typo"], Agent("p", 0.8), 10, 500)
            with raises(ValueError, match="Alive"):
                player_outcomes_on_layouts(Enemy("e", 0.5), ["alive"], Agent("p", 0.8), 10, 500)
    finally:
        del SCRIPT["typo"], SCRIPT["alive"]


def test_the_goal_share_is_one_minus_the_pressure_of_the_same_games():
    with scripted():
        shares = player_outcomes_on_layouts(Enemy("e", 0.5), ["a", "b"], Agent("p", 0.8), 10, 500)
    assert [1 - row["goal"] for row in shares] == [0.8, 1.0]


###############################################################################
# summarize
###############################################################################

def test_summarize_gives_mean_worst_and_best():
    assert summarize([0.2, 0.5, 0.8]) == {"mean": 0.5, "worst": 0.2, "best": 0.8}


def test_summarize_of_one_rate_and_of_none():
    assert summarize([0.4]) == {"mean": 0.4, "worst": 0.4, "best": 0.4}
    with raises(ValueError, match="no rates"):
        summarize([])


###############################################################################
# The real engine (skipped when it cannot be imported)
###############################################################################

def test_real_a_heuristic_player_on_new_layouts_is_repeatable():
    try:
        from engine import GenerationConfig, generate_valid_layout
        from agents.heuristic_agent.agent import HeuristicAgent
    except ImportError as error:
        raise unittest.SkipTest(f"real engine not available here: {error}")

    config = dict(width=12, height=12, wall_density=0.1, num_enemies=1, num_traps=0, min_player_goal_distance=6)
    layouts = held_out_layouts(lambda seed: generate_valid_layout(GenerationConfig(**config, seed=seed)), 3, 10_000)
    assert len(layouts) == 3

    first = skill_on_layouts("player", HeuristicAgent(), layouts, None, 10, 700_000)
    second = skill_on_layouts("player", HeuristicAgent(), layouts, None, 10, 700_000)
    assert first == second and all(0 <= rate <= 1 for rate in first)

    enemy = skill_on_layouts("enemy_0", HeuristicAgent(), layouts, HeuristicAgent(), 10, 700_000)
    assert len(enemy) == 3

    pressure = pressure_on_layouts(HeuristicAgent(), layouts, HeuristicAgent(), 10, 700_000)
    assert pressure == pressure_on_layouts(HeuristicAgent(), layouts, HeuristicAgent(), 10, 700_000)
    assert len(pressure) == 3 and all(0 <= value <= 1 for value in pressure)


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