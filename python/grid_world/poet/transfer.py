"""
Created on Fri Nov  6 08:49:11 2026

@author: Angelo Antonio Manzatto

Transfer: an entry adopts what another entry has learned, when that does better on its own layout.

What one entry learns can help in another layout: a player that escaped on one map may also escape
on a neighbouring one. Every role is looked at on its own. The entry measures its own agent and
the agents of the other entries in the same games, on ITS layout and against ITS opponent, and
takes the best one when it is better than its own.

What moves is the whole learner (a clone of it), not only the agent: poet cannot put an agent
into a learner, and the learner is what keeps learning afterwards. The candidates stay as they are.

"""

###############################################################################
# libraries
###############################################################################

from poet.entry import OPPONENT_OF, ROLES
from poet.games import goal_rate, play_games

###############################################################################
# Role rate
###############################################################################

def role_rate(entry, role, agent, n_games, first_seed):
    """The goal rate of `agent` playing `role` on the layout of `entry` against the entry's own
    agent of the other role, in the games first_seed ... first_seed + n_games - 1.

    Both play as CLONES, so every agent measured meets the opponent in the same state and nothing
    of the entry moves (an agent keeps drawing from its own random generator while it plays).
    """
    opponent_role = OPPONENT_OF[role]
    agents = {
        role: agent.clone(),
        opponent_role: entry.learners[opponent_role].agent().clone(),
    }
    outcomes = play_games(entry.layout, agents, n_games=n_games, first_seed=first_seed)

    return goal_rate(outcomes[role])

###############################################################################
# Transfer into
###############################################################################

def transfer_into(entry, candidates, n_games, first_seed, margin=0.0):
    """Let `entry` adopt, role by role, the learner of the candidate that plays it best.

    candidates : the other entries. They are not modified.
    n_games, first_seed : the games every agent is measured on (the same for everyone, so the
                 comparison is fair). They must not be games the learners train on.
    margin     : how much better than the entry's own agent a candidate must be, in goal rate. A
                 candidate is adopted only when rate > own + margin, so equal agents never
                 swap (margin 0 is the strict rule). With few games a small difference is noise:
                 raise it to ask for a clearer gain.

    The player is decided first. The enemy is then measured against the player the entry has AFTER
    that, like in train_entry. Among candidates with the same rate the first one wins.

    Changes entry.learners in place (an adopted learner is a clone of the candidate's).
    Returns a report per role: {"own": rate of the entry's agent, "best": the best candidate's rate,
    "from": the index in `candidates` of that candidate, "adopted": bool}. Without candidates
    nothing is played and own, best and from are None.
    Raises ValueError for a negative margin.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    report = {}

    for role in ROLES:
        if len(candidates) == 0:
            report[role] = {"own": None, "best": None, "from": None, "adopted": False}
            continue

        own = role_rate(entry, role, entry.learners[role].agent(), n_games, first_seed)

        best, best_index = None, None
        for index, candidate in enumerate(candidates):
            rate = role_rate(entry, role, candidate.learners[role].agent(), n_games, first_seed)
            if best is None or rate > best:           # strictly: the first of equals stays
                best, best_index = rate, index

        adopted = best > own + margin
        if adopted:
            entry.learners[role] = candidates[best_index].learners[role].clone()

        report[role] = {"own": own, "best": best, "from": best_index, "adopted": adopted}

    return report


def transfer_round(pool, n_games, first_seed, margin=0.0):
    """One round of transfer for the whole pool: every entry, in the order of the pool, looks at all
    the OTHER entries (see transfer_into). The pool as it is at that moment counts, so an entry
    that adopted something earlier in the round can be a candidate for the next one.

    Returns one report per entry, in the order of the pool: the report of transfer_into.
    Changes the entries in place.
    """
    return [
        transfer_into(entry, [other for other in pool if other is not entry], n_games, first_seed, margin)
        for entry in pool
    ]