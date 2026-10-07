"""
Created on Mon Oct  5 09:27:26 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass
from typing import Callable

from agents.base_agent import BaseAgent
from engine import World

###############################################################################
# Result
###############################################################################


@dataclass(frozen=True)
class EpisodeResult:
    """
    outcome: "GoalReached" | "Trapped" | "Caught" | "Timeout"
    steps:   number of world steps played
    """

    outcome: str
    steps: int
    terminated: bool
    truncated: bool

###############################################################################
# Run episode
###############################################################################


def run_episode(
    world: World,
    agents: dict[int, BaseAgent],
    reposition: bool = False,
    on_step: Callable[[World, int, dict], None] | None = None,
) -> EpisodeResult:
    """
    Play one episode.

    agents maps entity_id -> agent. The player MUST have an agent; entities
    without one (e.g. enemies) are moved by the engine itself.

    on_step(world, step, actions) is an optional observer (rendering,
    logging, recording). It is called once after the reset (step 0, no
    actions) and once after every world step, and must not change the world.
    """

    ts = world.reset(reposition)

    # Bind to locals once: every attribute access on ts / observation copies
    # data across the Rust boundary.
    obs = ts.observation
    entities = {e.id: e for e in obs.entities}
    player_id = obs.player_id

    if player_id not in agents:
        raise ValueError(f"No agent given for the player (entity {player_id})")

    unknown = set(agents) - set(entities)
    if unknown:
        raise ValueError(f"Agents given for entities not in the world: {sorted(unknown)}")

    for agent in agents.values():
        agent.reset()

    steps = 0
    if on_step is not None:
        on_step(world, steps, {})

    while not ts.done():

        # Only entities still alive in the world get to act
        actions = {
            eid: agent.act(entities[eid])
            for eid, agent in agents.items()
            if eid in entities
        }

        ts = world.step(actions)
        next_obs = ts.observation
        next_entities = {e.id: e for e in next_obs.entities}

        # An entity despawned by this step (e.g. enemy on a trap) has no
        # next observation, so it gets no on_transition for its last step.
        for eid, action in actions.items():
            if eid in next_entities:
                agents[eid].on_transition(
                    entities[eid], action, next_entities[eid],
                    ts.terminated, ts.truncated,
                )

        entities = next_entities
        steps += 1

        if on_step is not None:
            on_step(world, steps, actions)

    # Only the player receives GoalReached / Trapped / Caught.
    # A timeout changes no status, so it is read from the flags.
    outcome = entities[player_id].status if ts.terminated else "Timeout"

    return EpisodeResult(
        outcome=outcome,
        steps=steps,
        terminated=ts.terminated,
        truncated=ts.truncated,
    )