"""
Created on Tue Nov  3 20:56:12 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import math

###############################################################################
# Select novel
###############################################################################

def select_novel(candidates, pool, n_keep, distance):
    """Pick up to `n_keep` candidates, the most different from what the pool already holds.

    candidates : entries that passed the criterion and want to join.
    pool       : the entries already kept.
    n_keep     : the most that may be picked.
    distance   : distance(candidate, other) -> a number >= 0 that is 0 only for entries that are the
                 same for the pool's purposes. It should be symmetric. A negative or NaN distance
                 raises ValueError.

    A candidate's novelty is the distance to the NEAREST entry among the pool and the candidates
    already picked, so two similar siblings do not both get in. A candidate with novelty 0 never gets
    in, even when there is room: an exact copy adds nothing. With an empty pool every candidate is
    equally new at first, and ties go to the first in the list.

    Returns the picked entries (the same objects), most novel first. The arguments are not modified.
    """
    if n_keep < 0:
        raise ValueError(f"n_keep must be >= 0, got {n_keep}")

    def measure(candidate, other):
        value = distance(candidate, other)
        if math.isnan(value) or value < 0:
            raise ValueError(f"distance must be a number >= 0, got {value!r}")
        return value

    # Distance of every candidate to its nearest entry so far; an empty pool means "nothing to compare with"
    novelty = [min((measure(c, p) for p in pool), default=math.inf) for c in candidates]

    remaining = list(range(len(candidates)))
    picked = []

    while remaining and len(picked) < n_keep:
        best = max(remaining, key=novelty.__getitem__)       # max returns the first one on ties

        if novelty[best] == 0:
            break                                            # everything left is a copy of something

        picked.append(candidates[best])
        remaining.remove(best)

        for i in remaining:                                  # the pick is now part of what they must differ from
            novelty[i] = min(novelty[i], measure(candidates[i], candidates[best]))

    return picked

###############################################################################
# Admit
###############################################################################

def admit(pool, children, max_size):
    """The pool with the children added at the end. When it is over `max_size`, the OLDEST entries
    (the first ones) leave. Returns a new list; the arguments are not modified."""
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")
    return (list(pool) + list(children))[-max_size:]