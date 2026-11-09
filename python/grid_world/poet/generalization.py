"""
Created on Sun Nov  8 12:18:46 2026

@author: Angelo Antonio Manzatto

Does what an agent learned work on layouts it never saw?

The skill of an agent in a role is measured against a fixed opponent that does not learn, so two
agents that grew up with different partners can be compared on equal terms:
    a player      against the engine's own enemy (or any fixed agent you give)
    an enemy      against a fixed player (it needs one: without a player there is no game)
Every agent plays the same games on the same layouts, so a difference between two agents is not luck
of the starting positions.

"""

###############################################################################
# libraries
###############################################################################

from poet.entry import OPPONENT_OF
from poet.games import goal_rate, play_games

###############################################################################
# Held out layouts
###############################################################################

def held_out_layouts(make_layout, n_layouts, first_seed):
    """n_layouts new layouts, make_layout(seed) for the seeds first_seed ... first_seed + n_layouts - 1.

    make_layout: a function seed -> layout (for example one that calls generate_valid_layout with
    a GenerationConfig and that seed). The same arguments give the same layouts. Use seeds that
    nothing was ever trained on. Raises ValueError for n_layouts < 1.
    """
    if n_layouts < 1:
        raise ValueError(f"n_layouts must be >= 1, got {n_layouts}")

    return [make_layout(seed) for seed in range(first_seed, first_seed + n_layouts)]

###############################################################################
# Skill on layouts
###############################################################################

def skill_on_layouts(role, agent, layouts, opponent, n_games, first_seed):
    """The goal rate of `agent` playing `role` on each layout, in the order of `layouts`.

    opponent : the fixed agent that plays the other role, or None to leave it to the engine (only for
               the player: an enemy cannot play without a player). It does not learn.
    n_games, first_seed : the games played on EVERY layout (the same seeds for every agent you
               measure, so the comparison is paired).

    The agent and the opponent play as CLONES, a fresh pair for every layout: nothing the agent
    does on one layout carries over to the next, and the agent you pass is not moved.
    Raises ValueError for an unknown role, no layouts, or an enemy without an opponent.
    """
    if role not in OPPONENT_OF:
        raise ValueError(f"role must be one of {tuple(OPPONENT_OF)}, got {role!r}")
    if len(layouts) == 0:
        raise ValueError("there are no layouts to play on")
    if role != "player" and opponent is None:
        raise ValueError(f"{role!r} needs a player to play against: pass an opponent")

    rates = []
    for layout in layouts:
        agents = {role: agent.clone()}
        if opponent is not None:
            agents[OPPONENT_OF[role]] = opponent.clone()

        outcomes = play_games(layout, agents, n_games=n_games, first_seed=first_seed)
        rates.append(goal_rate(outcomes[role]))

    return rates

###############################################################################
# Pressure on layouts
###############################################################################

def pressure_on_layouts(enemy, layouts, player, n_games, first_seed):
    """How hard `enemy` makes life for a fixed player on each layout, in the order of `layouts`.
 
    The pressure is the share of games in which the fixed player does NOT reach its goal while
    facing this enemy (caught, trapped or timed out): a number in [0, 1], higher means a stronger
    enemy. It is the right yardstick for an enemy because the enemy's own catch rate is squeezed
    by the player (a strong player leaves it very little to catch), while the pressure still moves
    when the enemy only blocks, delays or herds the player.
 
    player : the fixed agent that plays the player, the same for every enemy you compare. It does
             not learn. It is required: without a player there is no game.
    n_games, first_seed : the games played on EVERY layout (the same for every enemy you
             measure, so the comparison is paired).
 
    Both play as CLONES, a fresh pair for every layout (see skill_on_layouts).
    Raises ValueError for no player, no layouts or fewer than one game.
    """
    if player is None:
        raise ValueError("pressure needs a player to push against: pass a player")
 
    goal_rates = skill_on_layouts("player", player, layouts, enemy, n_games, first_seed)
 
    return [1 - rate for rate in goal_rates]
 

###############################################################################
# Summarize
###############################################################################

def summarize(rates):
    """{"mean", "worst", "best"} of a list of rates (one per layout). Raises ValueError when empty."""
    if len(rates) == 0:
        raise ValueError("there are no rates to summarize")

    return {"mean": sum(rates) / len(rates), "worst": min(rates), "best": max(rates)}