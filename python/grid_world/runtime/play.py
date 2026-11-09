"""
Created on Tue Oct  6 08:05:54 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import time

from agents.base_agent import BaseAgent
from runtime.runner import EpisodeResult, run_episode
from engine import Action, World

try:
    from IPython.display import clear_output
except ImportError:  # plain python: frames are printed one after another
    clear_output = None

###############################################################################
# Helpers
###############################################################################

_ACTION_NAMES = [
    (Action.Up, "Up"),
    (Action.Down, "Down"),
    (Action.Left, "Left"),
    (Action.Right, "Right"),
]


def _action_name(action) -> str:
    for known, name in _ACTION_NAMES:
        if action == known:
            return name
    return str(action)

###############################################################################
# Play
###############################################################################

def play(
    world: World,
    agents: dict[int, BaseAgent],
    delay: float = 0.15,
    clear: bool = True,
    reposition: bool = False,
    role_of: dict[int, str] | None = None,
) -> EpisodeResult:
    """
    Play one episode, drawing each frame.

    delay:   seconds between frames.
    clear:   redraw in place (needs an IPython console). If you see garbage
             characters, or want to scroll back through the frames, use False.
    role_of: optional entity id -> role name ("player", "enemy_0", ...). When given, the
             frames and the final line say the role instead of the id.
    """

    def name(entity_id) -> str:
        return role_of[entity_id] if role_of is not None and entity_id in role_of else str(entity_id)

    def show(world: World, step: int, actions: dict) -> None:
        if clear and clear_output is not None:
            clear_output(wait=True)

        moves = ", ".join(f"{name(eid)}:{_action_name(a)}" for eid, a in actions.items())
        print(f"step {step}  tick {world.tick}  {moves}")
        print(world.render_world(), flush=True)

        if delay > 0:
            time.sleep(delay)

    result = run_episode(world, agents, reposition=reposition, on_step=show)

    outcomes = ", ".join(f"{name(eid)}: {outcome}" for eid, outcome in result.agent_outcomes.items())
    print(f"\n{outcomes} after {result.steps} steps")
    return result