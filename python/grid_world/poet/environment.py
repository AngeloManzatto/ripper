"""
Created on Mon Nov  2 08:10:05 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import re

from engine import World, MutationConfig, mutate_valid_layout

###############################################################################
# Make world
###############################################################################

def make_world(layout, seed):
    """A world of `layout` with `seed`, reset once, and the ids of its entities.
 
    Returns (world, player_id, enemy_ids). The reset is done WITHOUT repositioning: it is only here
    to read the ids. The ids change from layout to layout. The games themselves reset the world
    again, and they decide there whether the entities are repositioned (see run_episode).
    """
    world = World(layout, None, seed)
    step = world.reset()
    return world, step.observation.player_id, step.observation.enemy_ids
 
###############################################################################
# Mutate layout
###############################################################################

def mutate_layout(layout, mutation_config, seed):
   """A mutated copy of `layout`; the same layout, config and seed always give the same result.

   mutation_config: the settings of the engine's MutationConfig as a dict, WITHOUT the seed (the
                    seed is the argument; a "seed" inside the dict raises TypeError).
   Raises RuntimeError when the engine finds no valid mutation (it returns None).
   """
   new_layout = mutate_valid_layout(layout, MutationConfig(**mutation_config, seed=seed))
   if new_layout is None:
       raise RuntimeError("no valid mutation found")
   return new_layout

###############################################################################
# Layout picture
###############################################################################

def layout_picture(layout):
    """The layout drawn as plain text, exactly as the engine places it (no repositioning).
 
    render_world colours every wall with an escape code ("\\x1b[37m#\\x1b[0m" is ten characters for
    one cell), so the codes are removed: each cell is one character and the pictures of two layouts
    of the same size line up cell by cell.
    """
    world = World(layout, None, 0)
    world.reset()
    return re.sub(r"\x1b\[[0-9;]*m", "", world.render_world())
 
###############################################################################
# Layout distance
###############################################################################

def layout_distance(layout_a, layout_b):
    """How many cells differ between two layouts (walls, traps, entities). 0 = the same picture.
 
    Symmetric. Raises ValueError for layouts of different sizes: they are not comparable.
    """
    picture_a, picture_b = layout_picture(layout_a), layout_picture(layout_b)
 
    if len(picture_a) != len(picture_b):
        raise ValueError("layouts of different sizes cannot be compared")
 
    return sum(a != b for a, b in zip(picture_a, picture_b))