"""
Created on Tue Nov  3 14:28:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""What POET needs from whatever learns an agent.

POET never looks inside a learner. It only asks for the agent to play with, asks the learner to
improve on a layout, and asks for an independent copy. Any algorithm that offers these three
methods can be used: the learner owns everything else (its population, its training budget, how
it picks its best agent).
"""

from typing import Protocol, runtime_checkable

###############################################################################
# Learner
###############################################################################

@runtime_checkable
class Learner(Protocol):
    """A learner of one role (the player or the enemy). It needs no base class: having the three
    methods is enough."""

    def agent(self):
        """The agent to play with right now (the learner's best). Any BaseAgent. Playing with it
        must not change the learner."""
        ...

    def train(self, layout, opponent, rng):
        """Improve on this layout against `opponent` (an agent), for the amount of work the
        learner itself decides. Randomness comes from `rng` only. Changes the learner in place
        and returns None."""
        ...

    def clone(self):
        """An independent copy: training the copy never changes the original and the other way
        round. A child entry starts from the clone of its parent's learners."""
        ...