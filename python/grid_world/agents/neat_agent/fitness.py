"""
Created on Mon Oct 26 09:00:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# One signal for every agent: did it reach its goal?
###############################################################################

"""
Every agent gets the same signal, "GoalReached", from the engine when it achieves what its role
wants: the player when it reaches the exit, an enemy when it catches the player. run_episode
reports it in EpisodeResult.agent_outcomes, one outcome per agent. The fitness only looks at that
signal, so it is the same for every agent and knows nothing about roles:

    calculate_fitness(outcomes) = share of the games whose outcome is "GoalReached"

This is the goal part of the fitness, a number in [0, 1]. Other signals (novelty) are mixed with
it at the level of the population, see novelty.mix_fitness.
"""

GOAL = "GoalReached"

# Everything run_episode can report for an agent (see EpisodeResult)
OUTCOMES = ("GoalReached", "Trapped", "Caught", "Timeout", "Alive")


def calculate_fitness(outcomes: list) -> float:
    """Share of the games in which the agent reached its goal, in [0.0, 1.0].

    outcomes: one outcome per game, the one run_episode reported for this agent
              (result.agent_outcomes[agent_id]).
    Raises ValueError for an empty list or an outcome that is not in OUTCOMES, so a typo never
    scores as a loss. Nothing is modified.
    """
    if len(outcomes) == 0:
        raise ValueError("calculate_fitness needs at least one game")
    for outcome in outcomes:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}, expected one of {OUTCOMES}")

    return sum(outcome == GOAL for outcome in outcomes) / len(outcomes)