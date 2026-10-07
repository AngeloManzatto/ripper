"""
Created on Sun Sep 27 07:48:13 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import textwrap
import ripper

###############################################################################
# World
###############################################################################

class World:
    """Python-facing wrapper around the Rust World."""

    def __init__(self, layout: str, config: ripper.WorldConfig | None = None,
                 seed: int | None = None):
        """World initialization"""

        # Parse layout to avoid special invisible characters (\n)
        layout = textwrap.dedent(layout).strip()

        self._world = ripper.World(layout, config, seed)

    def reset(self, reposition: bool = False) -> ripper.TimeStep:
        """Reset the episode. Returns whatever the engine's reset() returns today, unchanged."""
        return self._world.reset(reposition)

    def step(self, actions: dict[int, ripper.Action]) -> ripper.TimeStep:
        """Advance one tick. Returns whatever the engine's step() returns today, unchanged."""
        return self._world.step(actions)
    
    def render_world(self) -> str:
        return self._world.render_world()

    def render_entity_view(self, entity_id: int) -> str:
        return self._world.render_entity_view(entity_id)
    
    width  = property(lambda self: self._world.width)
    height = property(lambda self: self._world.height)
    tick   = property(lambda self: self._world.tick)
    done   = property(lambda self: self._world.done)