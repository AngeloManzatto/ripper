"""
Created on Mon Nov  2 10:12:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from poet.games import goal_rate
 
###############################################################################
# Passes minimal criterion
###############################################################################

def passes_minimal_criterion(outcomes, low=0.2, high=0.7, timeout_cap=0.3):
    """Accept the setting when both agents have a fair chance and the games end.
 
    outcomes: role -> list of outcomes, one per game, as play_games returns them. It must hold
              "player" and "enemy_0", both with the same number of games (at least one).
    Accepted when
        - the goal rate of the player is between low and high (limits included),
        - the goal rate of the enemy is between low and high (limits included),
        - the share of games that ran out of time ("Timeout") is at most timeout_cap.
    Returns (accepted, report): report = {"player": rate, "enemy_0": rate, "timeout": share}, to
    see WHICH condition failed. The timeout share is measured on the player's games, a timeout ends
    the game for everybody.
    Raises ValueError for a missing role, lists of different length or empty, or limits that are
    not 0 <= low <= high <= 1 and 0 <= timeout_cap <= 1.
    """
    if not 0 <= low <= high <= 1:
        raise ValueError(f"needs 0 <= low <= high <= 1, got low={low}, high={high}")
    if not 0 <= timeout_cap <= 1:
        raise ValueError(f"timeout_cap must be in [0, 1], got {timeout_cap}")
 
    for role in ("player", "enemy_0"):
        if role not in outcomes:
            raise ValueError(f"the minimal criterion needs the outcomes of {role!r}")
    if len(outcomes["player"]) != len(outcomes["enemy_0"]):
        raise ValueError("player and enemy_0 must have the outcomes of the same games, got "
                         f"{len(outcomes['player'])} and {len(outcomes['enemy_0'])}")
 
    report = {
        "player": goal_rate(outcomes["player"]),         # also rejects empty lists and typos
        "enemy_0": goal_rate(outcomes["enemy_0"]),
    }
    games = outcomes["player"]
    report["timeout"] = sum(outcome == "Timeout" for outcome in games) / len(games)
 
    accepted = (
        low <= report["player"] <= high
        and low <= report["enemy_0"] <= high
        and report["timeout"] <= timeout_cap
    )
 
    return accepted, report