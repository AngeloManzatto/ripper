"""
Created on Tue Nov  3 15:21:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from poet.criterion import passes_minimal_criterion
from poet.entry import entry_agents
from poet.games import play_games

###############################################################################
# Is eligible
###############################################################################

def is_eligible(entry, n_games, first_seed, **limits):
    """Play n_games games on the layout of the entry, seeds first_seed ... first_seed + n_games - 1,
    with the agents the entry plays with now, and apply the minimal criterion.

    limits: low, high and timeout_cap of passes_minimal_criterion (its defaults when left out).
    Returns (eligible, report), the answer and the goal rates and timeout share behind it.
    Nothing is trained and the entry is not modified. The seeds are chosen by the caller, and must
    not be the ones the learners train on.
    """
    
    agents = {role: agent.clone() for role, agent in entry_agents(entry).items()}
    
    outcomes = play_games(entry.layout, agents, n_games=n_games, first_seed=first_seed)

    return passes_minimal_criterion(outcomes, **limits)