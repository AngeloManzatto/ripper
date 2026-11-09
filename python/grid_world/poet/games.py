"""
Created on Mon Nov  2 08:40:55 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from collections import defaultdict

from poet.environment import make_world

from runtime.runner import run_episode
from runtime.play import play

###############################################################################
# Globals
###############################################################################

GOAL = "GoalReached"
 
# Everything run_episode can report for an agent
OUTCOMES = ("GoalReached", "Trapped", "Caught", "Timeout", "Alive")

###############################################################################
# Split Roles
###############################################################################

def split_roles(agents_by_role):
    """(player_agent, enemy_agents) from a dict role -> agent.
 
    enemy_agents is the list [enemy_0, enemy_1, ...] in order of their number, or None when there
    is no enemy agent. Raises ValueError without a "player", with a role that is not "player" or
    "enemy_<i>", or with gaps in the enemy numbers (enemy_1 without enemy_0).
    """
    if "player" not in agents_by_role:
        raise ValueError("a game needs a 'player' agent")
 
    n_enemies = 0
    for role in agents_by_role:
        if role != "player" and role not in {f"enemy_{i}" for i in range(len(agents_by_role))}:
            raise ValueError(f"unknown role {role!r}, expected 'player', 'enemy_0', 'enemy_1', ...")
        if role.startswith("enemy_"):
            n_enemies += 1
 
    if n_enemies == 0:
        return agents_by_role["player"], None
 
    missing = [f"enemy_{i}" for i in range(n_enemies) if f"enemy_{i}" not in agents_by_role]
    if missing:
        raise ValueError(f"enemy roles must be numbered from 0 without gaps, missing {missing}")
 
    return agents_by_role["player"], [agents_by_role[f"enemy_{i}"] for i in range(n_enemies)]

###############################################################################
# Build game
###############################################################################

def build_game(layout, seed, agents_by_role):
    """The world of `layout` with `seed` and the agents that play it, ready for run_episode or play.
 
    agents_by_role: dict role -> agent (see split_roles). Roles are checked BEFORE the world is built.
    Returns (world, agents, role_of):
        agents  : entity id -> agent, for the entities that have an agent. An enemy without an agent
                  is moved by the engine itself.
        role_of : entity id -> role, for EVERY entity of the world ("player", "enemy_0", ...).
    The ids change from layout to layout, the roles do not.
    Raises ValueError when the number of enemy agents is not the number of enemies of the layout
    (either giving agents to some enemies only, or to enemies that do not exist).
    """
    player_agent, enemy_agents = split_roles(agents_by_role)
 
    world, player_id, enemy_ids = make_world(layout, seed=seed)
 
    role_of = {player_id: "player"}
    for i, enemy_id in enumerate(enemy_ids):
        role_of[enemy_id] = f"enemy_{i}"
 
    agents = {player_id: player_agent}
 
    if enemy_agents is not None:
        if len(enemy_agents) != len(enemy_ids):
            raise ValueError(f"{len(enemy_agents)} enemy agents for {len(enemy_ids)} enemies")
 
        for enemy_id, enemy_agent in zip(enemy_ids, enemy_agents):
            agents[enemy_id] = enemy_agent
 
    return world, agents, role_of

###############################################################################
# Run game
###############################################################################

def run_game(layout, seed, agents_by_role):
    """One game. Returns (result, role_of): result is the runner's EpisodeResult (its agent_outcomes
    are keyed by entity id) and role_of maps each entity id of THIS game to its role.
 
    The entities are repositioned at the start (reposition=True): on a fixed layout the seed decides
    where they start, so different seeds give different games. Same layout, seed and agents give the
    same game.
    """
    world, agents, role_of = build_game(layout, seed, agents_by_role)
 
    return run_episode(world, agents, reposition=True), role_of

###############################################################################
# Watch game
###############################################################################

def watch_game(layout, seed, agents_by_role, delay=0.2):
    """Draw one game frame by frame with play(). Same game as run_game with the same seed.
 
    delay: seconds between frames. Prints the roles of the entity ids first.
    Returns the agents of the game by ROLE ({"player": ..., "enemy_0": ...}), so their records of
    the game can be looked at afterwards.
    """
    world, agents, role_of = build_game(layout, seed, agents_by_role)
 
    print("roles in this game:", role_of)
    play(world, agents, delay=delay, role_of=role_of, reposition=True)
 
    return {role_of[entity_id]: agent for entity_id, agent in agents.items()}

###############################################################################
# Play games
###############################################################################

def play_games(layout, agents_by_role, n_games=100, first_seed=0):
    """Play the seeds first_seed ... first_seed + n_games - 1 on this layout with these agents.
 
    Returns the outcomes keyed by ROLE, only for the roles that have an agent:
        {"player": ["GoalReached", "Timeout", ...], "enemy_0": [...]}
    each list holds the outcome of the role in every game, in seed order.
    Raises ValueError when n_games < 1.
    """
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")
 
    outcomes = defaultdict(list)
 
    for seed in range(first_seed, first_seed + n_games):
        result, role_of = run_game(layout, seed, agents_by_role)
 
        for entity_id, outcome in result.agent_outcomes.items():
            outcomes[role_of[entity_id]].append(outcome)
 
    return dict(outcomes)

###############################################################################
# Goal Rate
###############################################################################

def goal_rate(outcomes):
    """Share of the games in which the agent reached its goal, a number in [0, 1].
 
    outcomes: one outcome per game, as play_games returns them for a role.
    Raises ValueError for an empty list or for an outcome the engine does not report, so a typo
    is an error and never counts as a loss.
    """
    if len(outcomes) == 0:
        raise ValueError("goal_rate needs at least one game")
 
    for outcome in outcomes:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}, expected one of {OUTCOMES}")
 
    return sum(outcome == GOAL for outcome in outcomes) / len(outcomes)