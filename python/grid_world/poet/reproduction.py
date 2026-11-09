"""
Created on Wed Nov  4 15:59:35 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

"""Reproduction: the new entries a pool produces in one round.

Only the eligible entries are parents. Each tries to make children; the accepted children compete,
and only the most novel ones are returned. Joining the pool is a separate step (pool.admit), and so
is training.
"""

from poet.child import make_child
from poet.eligibility import is_eligible
from poet.environment import layout_distance
from poet.pool import select_novel

###############################################################################
# Entry distance
###############################################################################

def entry_distance(a, b):
    """The distance between two entries is the distance between their layouts."""
    return layout_distance(a.layout, b.layout)

###############################################################################
# Reproduce
###############################################################################

def reproduce(pool, mutation_config, mutation_seed, n_children, max_new, n_games, first_seed, **limits):
    """The children that should join `pool` after this round. Returns (children, report).

    pool            : the entries (poet.entry.Entry).
    mutation_config : see environment.mutate_layout.
    mutation_seed   : seed of the first attempt. Attempt i uses mutation_seed + i, parent after
                      parent, so the same arguments give the same children and no two attempts
                      use the same seed. The caller moves it forward between rounds.
    n_children      : mutation attempts per eligible parent (at least 1).
    max_new         : the most children returned.
    n_games, first_seed, limits : see is_eligible. They decide who is a parent AND whether a child
                      is accepted, on the same games.

    Every entry of the pool is a possible parent when eligible, and every entry of the pool is
    something a child must differ from. The children come from select_novel, most novel first.
    report: {"eligible": parents, "attempts": mutations tried, "accepted": children that passed the
    criterion, "picked": children returned}. With no eligible parent nothing is tried and the report
    shows it, so the caller can warn. The pool is not modified and nothing is trained.
    """
    if n_children < 1:
        raise ValueError(f"n_children must be >= 1, got {n_children}")

    parents = [entry for entry in pool if is_eligible(entry, n_games, first_seed, **limits)[0]]

    candidates, attempts = [], 0
    for parent in parents:
        for _ in range(n_children):
            child, _ = make_child(parent, mutation_config, mutation_seed + attempts, n_games, first_seed, **limits)
            attempts += 1
            if child is not None:
                candidates.append(child)

    children = select_novel(candidates, pool, max_new, entry_distance)

    report = {"eligible": len(parents), "attempts": attempts,
              "accepted": len(candidates), "picked": len(children)}
    return children, report