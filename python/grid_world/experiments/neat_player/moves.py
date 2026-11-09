"""
Created on Sat Oct 24 17:50:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Move statistics of one game
###############################################################################

def move_stats(positions: list) -> dict:
    """
    Numbers that describe HOW a player moved in one game, to find the games worth watching.

    positions: the cell the player stood on when it chose each action, in order, each one a
               hashable pair such as (row, col). One entry per action taken.

    Returns a dict:
        steps    : number of actions taken (len(positions))
        distinct : number of different cells visited
        stay     : times the player is on the same cell as one decision before, so its last
                   move did nothing (walked into a wall or the border)
        flip     : times the player is back on the cell it stood on two decisions before,
                   after standing on a different one in between (A, B, A: stepping back and
                   forth)

    An empty list gives all zeros. Nothing is modified.
    """
    stay = 0
    flip = 0

    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            stay += 1
        elif i >= 2 and positions[i] == positions[i - 2]:
            flip += 1

    return {
        "steps": len(positions),
        "distinct": len(set(positions)),
        "stay": stay,
        "flip": flip,
    }


###############################################################################
# Table of many games
###############################################################################

def behaviour_table(rows: list) -> str:
    """
    Text table of several games, plus one summary line per outcome.

    rows: one dict per game with the keys seed, outcome, steps, distinct, stay, flip
          (the outcome name and the numbers of move_stats).

    Each game gets a line. Below, for every outcome that occurred (in order of first
    appearance): how many games, and, over all steps of those games, the share of steps in
    which the player stayed on its cell ("stay") and the share in which it stepped back to the
    cell it was on two decisions before ("flip"). Raises ValueError for an empty list.
    """
    if not rows:
        raise ValueError("behaviour_table needs at least one game")

    head = f"{'seed':>8}  {'outcome':<12} {'steps':>6} {'distinct':>9} {'stay':>6} {'flip':>6}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['seed']:>8}  {r['outcome']:<12} {r['steps']:>6} {r['distinct']:>9} "
                     f"{r['stay']:>6} {r['flip']:>6}")

    lines.append("")
    outcomes = list(dict.fromkeys(r["outcome"] for r in rows))
    for outcome in outcomes:
        group = [r for r in rows if r["outcome"] == outcome]
        steps = sum(r["steps"] for r in group)
        stay = sum(r["stay"] for r in group) / steps if steps else 0.0
        flip = sum(r["flip"] for r in group) / steps if steps else 0.0
        lines.append(f"{outcome:<12} {len(group):>3} games   stay {stay:6.1%} of steps   flip {flip:6.1%} of steps")

    return "\n".join(lines)