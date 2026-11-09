"""
Created on Fri Nov  6 21:18:39 2026

@author: Angelo Antonio Manzatto

The POET loop: reproduce, let the new entries join the pool, train everything, transfer, repeat.

One generation is, in this order:
    1. reproduce : the eligible entries make children, the most novel accepted ones are chosen
    2. admit     : the children join the pool, the oldest entries leave when it is over max_pool
    3. train     : every entry of the pool trains on its own layout (the new ones too)
    4. transfer  : every n_transfer generations, each entry may adopt what another entry learned
                   (only when the pool has more than one entry)
The loop does not print or save anything itself: on_generation is called after every generation.

"""

###############################################################################
# libraries
###############################################################################

import random

from poet.entry import train_entry
from poet.pool import admit
from poet.reproduction import reproduce
from poet.transfer import transfer_round

###############################################################################
# Adoptions
###############################################################################

def adoptions(pool, reports):
    """The adoptions of one transfer round as (entry, role, from), with entry and from as positions
    in the pool. `reports` are the reports of transfer_round: the candidates of entry i are all the
    other entries, in the order of the pool, and "from" counts in that list."""
    found = []
    for i, entry_report in enumerate(reports):
        sources = [k for k in range(len(pool)) if k != i]
        for role, row in entry_report.items():
            if row["adopted"]:
                found.append((i, role, sources[row["from"]]))
    return found

###############################################################################
# Run POET
###############################################################################

def run_poet(pool, mutation_config, n_generations, max_pool, n_children, max_new, n_games, first_seed,
             mutation_seed=0, train_seed=0, on_generation=None,
             n_transfer=None, transfer_seed=500_000, n_transfer_games=100, transfer_margin=0.0, **limits):
    """Run n_generations generations. Returns (pool, history).

    pool            : the starting entries (not empty). The list is not modified; the entries are
                      trained in place, as train_entry does.
    mutation_config : see environment.mutate_layout.
    max_pool        : the pool never has more entries than this; the oldest leave first.
    n_children      : mutation attempts per eligible parent, per generation.
    max_new         : the most children admitted per generation.
    n_games, first_seed, limits : the criterion for parents and children (see is_eligible). The games
                      are always the same ones, and must not be the ones the learners train on.
    mutation_seed   : seed of the first mutation attempt. It moves forward by the number of attempts
                      made, so no two attempts of the run share a seed.
    train_seed      : generation g trains with random.Random(train_seed + g), one generator shared by
                      the entries of that generation, in the order of the pool.
    on_generation   : optional function(generation, pool, report) called at the end of every
                      generation, after training and transfer. Use it to print, save, plot or stop
                      early by raising.
    n_transfer      : every this many generations (after the n_transfer-th, the 2 * n_transfer-th, ...)
                      the pool does a round of transfer (see transfer_round), once it has more than one
                      entry. None = never.
    transfer_seed, n_transfer_games, transfer_margin : the games every agent is measured on and the
                      gain an agent must show to be adopted (see transfer_into). Generation g plays
                      the seeds transfer_seed + g * n_transfer_games ..., so no round measures on the
                      games of another. They must not be games the learners train on.

    history[g] is the report of generation g:
        {"generation", "eligible", "attempts", "accepted", "picked"   (see reproduce),
         "dropped" (entries that left the pool), "pool" (its size after the generation),
         "transfers" (one (entry, role, from) for every adoption of this generation, entry and from
                      as positions in the pool after it; [] when there was no transfer round)}
    """
    if len(pool) == 0:
        raise ValueError("the pool needs at least one entry")
    if n_generations < 0:
        raise ValueError(f"n_generations must be >= 0, got {n_generations}")
    if max_pool < len(pool):
        raise ValueError(f"max_pool ({max_pool}) is smaller than the starting pool ({len(pool)})")
    if n_transfer is not None and n_transfer < 1:
        raise ValueError(f"n_transfer must be None or >= 1, got {n_transfer}")
    if n_transfer_games < 1:
        raise ValueError(f"n_transfer_games must be >= 1, got {n_transfer_games}")
    if transfer_margin < 0:
        raise ValueError(f"transfer_margin must be >= 0, got {transfer_margin}")

    pool = list(pool)
    history = []

    for generation in range(n_generations):
        children, report = reproduce(pool, mutation_config, mutation_seed, n_children, max_new,
                                     n_games, first_seed, **limits)
        mutation_seed += report["attempts"]

        size_before = len(pool)
        pool = admit(pool, children, max_pool)
        dropped = size_before + len(children) - len(pool)

        rng = random.Random(train_seed + generation)
        for entry in pool:
            train_entry(entry, rng)

        transfers = []
        if n_transfer is not None and (generation + 1) % n_transfer == 0 and len(pool) > 1:
            reports = transfer_round(pool, n_transfer_games, transfer_seed + generation * n_transfer_games,
                                     transfer_margin)
            transfers = adoptions(pool, reports)

        report = {"generation": generation, **report, "dropped": dropped, "pool": len(pool), "transfers": transfers}
        history.append(report)

        if on_generation is not None:
            on_generation(generation, pool, report)

    return pool, history