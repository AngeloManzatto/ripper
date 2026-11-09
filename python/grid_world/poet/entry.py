"""
Created on Tue Nov  3 15:08:01 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""An entry of the POET pool: one layout and the learners that live on it.

The layout is the environment, the learners (player and enemy) are the ones being trained on it.
POET only sees learners through agent(), train() and clone() (see poet.learner).
"""

from dataclasses import dataclass

ROLES = ("player", "enemy_0")

# Whom each role is trained against
OPPONENT_OF = {"player": "enemy_0", "enemy_0": "player"}

###############################################################################
# Entry
###############################################################################

@dataclass
class Entry:
    """layout: the layout of this entry.
    learners: role -> Learner, exactly the roles "player" and "enemy_0" (one enemy for now)."""
    layout: object
    learners: dict

    def __post_init__(self):
        if set(self.learners) != set(ROLES):
            raise ValueError(f"an entry needs learners for exactly {ROLES}, got {tuple(self.learners)}")

def entry_agents(entry):
    """The agents to play with now, by role: {"player": ..., "enemy_0": ...}. Ready for play_games."""
    return {role: learner.agent() for role, learner in entry.learners.items()}

def train_entry(entry, rng):
    """One round of training on the entry's layout. The player trains against the enemy's current
    agent, then the enemy trains against the player's agent AS IT IS AFTER its own training.
    Every learner decides how much work that is. Changes the learners in place, returns None.
    """
    for role in ROLES:
        opponent = entry.learners[OPPONENT_OF[role]].agent()
        entry.learners[role].train(entry.layout, opponent, rng)