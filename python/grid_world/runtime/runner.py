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
from core.transition import Transition
from engine import World

###############################################################################
# Result
###############################################################################

@dataclass(frozen=True)
class EpisodeResult:
    """
    agent_outcomes: entity id -> outcome, one entry per agent given to run_episode (entities
                    moved by the engine have none). The outcome is the final status of the
                    agent's entity when it is not "Alive":
                        "GoalReached" : the player reached the exit, or an enemy caught the player
                        "Trapped"     : the entity stepped on a trap
                        "Caught"      : the player was caught
                    and otherwise:
                        "Timeout"     : the clock ended the game while the agent was alive
                        "Alive"       : the game ended (terminated) while the agent was alive
    steps:          number of world steps played
    terminated:     the game ended by a game outcome of the player
    truncated:      the game ended by the clock
    """
    agent_outcomes: dict[int, str]
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
        
    # Reset agent states
    for agent in agents.values():
        agent.reset()

    # Last status seen of every agent's entity. The engine despawns a dead enemy right after the
    # step it dies in, so its final status exists only in the observation of that step: keep it,
    # because `entities` (the latest observation) no longer has the entity afterwards.
    statuses = {agent_id: entities[agent_id].status for agent_id in agents}

    steps = 0
    
    # Execute post processing if any
    if on_step is not None:
        on_step(world, steps, {})

    while not ts.done():

        # Only entities still alive in the world get to act
        actions = {}
        
        for entity_id, agent in agents.items():
            
            entity = entities.get(entity_id)       # None when the entity is gone from the world
            if entity is not None and entity.status == "Alive":
                actions[entity_id] = agent.act(entity)

        ts = world.step(actions)
        next_obs = ts.observation
        
        next_entities = {e.id: e for e in next_obs.entities}

        for agent_id in statuses:
            if agent_id in next_entities:
                statuses[agent_id] = next_entities[agent_id].status

        truncated = ts.truncated
        
        for entity_id, action in actions.items():

            if entity_id not in next_entities:
                raise RuntimeError(
                    f"Entity {entity_id} acted but is missing from the step's final observation"
                )

            entity_obs = entities[entity_id]
            next_entity_obs = next_entities[entity_id]

            terminated = ts.terminated or next_entity_obs.status != "Alive"

            transition = Transition(entity_obs, action, next_entity_obs, terminated, truncated)

            agents[entity_id].on_transition(transition)

        entities = next_entities
        steps += 1

        if on_step is not None:
            on_step(world, steps, actions)
            
    # Agent outcomes: a status that is not Alive is final, whenever it happened
    agent_outcomes = {}

    for agent_id, status in statuses.items():

        if status != "Alive":
            agent_outcomes[agent_id] = status
        elif ts.truncated:
            agent_outcomes[agent_id] = "Timeout"
        else:
            agent_outcomes[agent_id] = "Alive"

    result = EpisodeResult(
        agent_outcomes=agent_outcomes,
        steps=steps,
        terminated=ts.terminated,
        truncated=ts.truncated,
        )

    return result