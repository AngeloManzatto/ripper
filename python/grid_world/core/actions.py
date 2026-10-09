"""
Created on Thu Oct  8 21:31:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from engine import Action

###############################################################################
# Actions
###############################################################################

# Ripper engine actions for grid world
ACTIONS = [Action.Up, Action.Down, Action.Left, Action.Right]

# Ripper directions by name
DIRECTION_NAMES = ["up", "down", "left", "right"]

# (d_row, d_col) produced by each action; checked against the engine in
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# OPPOSITE[i] is the index of the action that undoes action i
OPPOSITE = [1, 0, 3, 2]