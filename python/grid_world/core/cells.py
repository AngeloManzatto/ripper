"""
Created on Thu Oct  8 21:25:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

###############################################################################
# Cell codes
###############################################################################

UNKNOWN, FREE, WALL, PLAYER, ENEMY, TRAP, GOAL = -1, 0, 1, 2, 3, 4, 5

# One printable character per code (rendering / debugging)
SYMBOLS = {
    UNKNOWN: "?",
    FREE:    ".",
    WALL:    "#",
    PLAYER:  "P",
    ENEMY:   "E",
    TRAP:    "T",
    GOAL:    "G",
}

N_CHANNELS = len(SYMBOLS)   # one-hot channels, one per code

###############################################################################
# Entity kinds (EntityObservation.kind)
###############################################################################

KIND_PLAYER, KIND_ENEMY = "player", "enemy"

def hostile_code(kind: str) -> int:
    """Cell code of the entities this observer should treat as hostile."""
    if kind == KIND_PLAYER:
        return ENEMY
    if kind == KIND_ENEMY:
        return PLAYER
    raise ValueError(f"Unknown entity kind: {kind!r}")