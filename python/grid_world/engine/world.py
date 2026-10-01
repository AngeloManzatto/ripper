"""
Created on Sun Sep 27 07:48:13 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from ripper import PyWorld, WorldConfig
import textwrap

###############################################################################
# World
###############################################################################

class World:
    """Python-facing wrapper around the Rust World."""

    def __init__(self, layout:str,  config: WorldConfig=None):
        """World initialization"""
        
        # Parse layout to avoid special invisible  characters (|n)
        layout = textwrap.dedent(layout).strip()
        
        self._world = PyWorld(layout, config or WorldConfig())

    def reset(self):
        """Reset the episode. Returns whatever the engine's reset() returns today, unchanged."""
        return self._world.reset()

    def step(self, actions: dict):
        """Advance one tick. Returns whatever the engine's step() returns today, unchanged."""
        return self._world.step(actions)