"""
Created on Tue Nov  3 19:31:47 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""A child of an entry: the layout of the parent, mutated, with the parent's learners.

The child is only accepted when the parent's agents, playing the NEW layout, pass the minimal
criterion there (the same check as is_eligible). A child that is too easy or too hard for them
is not worth training.
"""

from poet.eligibility import is_eligible
from poet.entry import Entry
from poet.environment import mutate_layout

###############################################################################
# Make child
###############################################################################

def make_child(parent, mutation_config, mutation_seed, n_games, first_seed, **limits):
    """Try to make one child of `parent`. Returns (child, report):
        child  : a new Entry on the mutated layout, whose learners are CLONES of the parent's, or
                 None when it was not accepted
        report : the goal rates and timeout share behind the decision, or None when the engine
                 found no valid mutation (nothing was played then)

    mutation_config, mutation_seed: see environment.mutate_layout (the same seed, the same child
                                    layout).
    n_games, first_seed, limits: see is_eligible, measured on the mutated layout.
    The parent is not modified and nothing is trained. Learners are cloned only for an accepted
    child.
    """
    try:
        layout = mutate_layout(parent.layout, mutation_config, mutation_seed)
    except RuntimeError:                                  # "no valid mutation found"
        return None, None

    # The parent's own agents on the new layout. No clones yet: the check only plays.
    candidate = Entry(layout, parent.learners)
    accepted, report = is_eligible(candidate, n_games, first_seed, **limits)

    if not accepted:
        return None, report

    child = Entry(layout, {role: learner.clone() for role, learner in parent.learners.items()})
    return child, report