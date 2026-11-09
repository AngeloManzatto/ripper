"""
Tests for agents/neat_agent/fitness.py.

calculate_fitness(outcomes)
    The share of the games in which the agent reached its goal, whatever its role: it only counts
    "GoalReached" among the outcomes that run_episode reported for the agent.

    - The result is goals / games, in [0.0, 1.0], and does not depend on the order of the games.
    - Every other outcome ("Trapped", "Caught", "Timeout", "Alive") counts as no goal.
    - An empty list or an unknown outcome raises ValueError (a typo must not score as a loss).
      Nothing is modified.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

from agents.neat_agent.fitness import GOAL, OUTCOMES, calculate_fitness

###############################################################################
# helpers
###############################################################################

def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False

###############################################################################
# calculate_fitness
###############################################################################

def test_the_rate_is_goals_over_games():
    outcomes = ["GoalReached", "Caught", "Timeout", "GoalReached", "Trapped", "Timeout", "Alive", "Caught"]
    assert calculate_fitness(outcomes) == 2 / 8
    assert calculate_fitness(outcomes + ["GoalReached"]) == 3 / 9


def test_only_goal_reached_counts():
    assert GOAL == "GoalReached"
    for outcome in OUTCOMES:
        assert calculate_fitness([outcome]) == (1.0 if outcome == "GoalReached" else 0.0), outcome


def test_all_goals_is_one_and_none_is_zero():
    assert calculate_fitness(["GoalReached"] * 5) == 1.0
    assert calculate_fitness(["Timeout", "Caught", "Alive", "Trapped"]) == 0.0


def test_the_order_of_the_games_does_not_matter():
    outcomes = ["Timeout", "GoalReached", "Caught", "GoalReached"]
    assert calculate_fitness(outcomes) == calculate_fitness(list(reversed(outcomes))) == 0.5


def test_empty_list_and_unknown_outcome_raise():
    assert raises(ValueError, calculate_fitness, [])
    assert raises(ValueError, calculate_fitness, ["GoalReached", "GoalReched"])
    assert raises(ValueError, calculate_fitness, ["PlayerEscaped"])


def test_the_input_is_not_modified():
    outcomes = ["GoalReached", "Caught", "Timeout"]
    before = list(outcomes)
    calculate_fitness(outcomes)
    assert outcomes == before

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"neat fitness: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()