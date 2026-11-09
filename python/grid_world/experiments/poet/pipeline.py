"""
Created on Sun Oct 25 11:09:07 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import re
import copy
import time
import pickle
from collections import Counter, defaultdict
import numpy as np

from engine import GenerationConfig, World, generate_valid_layout
from engine import MutationConfig, mutate_valid_layout  

from runtime.runner import run_episode
from runtime.play import play
from core.actions import N_ACTIONS

from agents.neat_agent.agent import NeatAgent, N_INPUTS
from agents.neat_agent.fitness import calculate_fitness
from agents.neat_agent.novelty import novelty_scores, mix_fitness

from agents.heuristic_agent.agent import HeuristicAgent

from neat.config import default_config
from neat.genome import init_genome
from neat.evolution import init_population, evolve_one_generation

###############################################################################
# Globals
###############################################################################

NOVELTY_WEIGHT = 0.2
K_NEIGHBOURS   = 10

POPULATION_SIZE = 50
N_GAMES         = 20
FIRST_SEED      = 0

###############################################################################
# Settings
###############################################################################

generation_config = {
    "width"  : 16,
    "height" : 16,
    "wall_density" : 0.1,
    "num_enemies" : 1,
    "num_traps" : 0,
    "min_player_goal_distance" : 10
}

mutation_config = {
    "n_wall_mutations" : 1,
    "max_wall_density" : 0.3,
    "max_enemies" : 1,
    "enemy_add_weight"     : 0.0,
    "enemy_remove_weight"  : 0.0,
    "enemy_nothing_weight" : 1.0,
    "max_traps" : 3,
    "trap_add_weight"     : 0.0,
    "trap_remove_weight"  : 0.0,
    "trap_nothing_weight" : 1.0,
    "n_repositions" : 1,
    "min_player_goal_distance" : 10,
}

###############################################################################
# Base layout
###############################################################################

layout = generate_valid_layout(GenerationConfig(**generation_config))

###############################################################################
# Make world
###############################################################################

def make_world(layout, seed):
    world = World(layout, None, seed)
    step = world.reset()
    return world, step.observation.player_id, step.observation.enemy_ids

world, player_id, enemy_ids = make_world(layout, seed=1)
print(world.render_world())

###############################################################################
# Mutate world
###############################################################################

def mutate_layout(layout, mutation_config, seed):
    new_layout = mutate_valid_layout(layout, MutationConfig(**mutation_config, seed=seed))
    if new_layout is None:         
        raise RuntimeError("no valid mutation found")
    return new_layout
    
new_layout  =  mutate_layout(layout, mutation_config, seed=1)
world, player_id, enemy_ids = make_world(new_layout, 1)
print(world.render_world())    

###############################################################################
# Two fresh agents, one game
###############################################################################

neat_config = default_config()
rng = np.random.default_rng(0)

player_agent = NeatAgent(init_genome(N_INPUTS, N_ACTIONS, rng, neat_config))
enemy_agent  = NeatAgent(init_genome(N_INPUTS, N_ACTIONS, rng, neat_config))

# One agent per entity id. Every id in this dict is controlled by its agent.
agents = {player_id: player_agent, enemy_ids[0]: enemy_agent}

result = run_episode(world, agents)
print(result.agent_outcomes)

###############################################################################
# Play many games and count the outcomes
###############################################################################

N_GAMES = 100

outcomes = defaultdict(Counter)

for seed in range(N_GAMES):
    world, player_id, enemy_ids = make_world(layout, seed=seed)
    agents = {player_id: player_agent, enemy_ids[0]: enemy_agent}
    result = run_episode(world, agents)

    for agent_id, outcome in result.agent_outcomes.items():
        outcomes[agent_id][outcome] += 1

print({agent_id: dict(counts) for agent_id, counts in outcomes.items()})

###############################################################################
# Group agents by its roles
###############################################################################

def split_roles(agents_by_role):
    """(player_agent, enemy_agents) from a dict role -> agent.
 
    enemy_agents is the list [enemy_0, enemy_1, ...] in order, or None when there is no enemy
    agent. Raises ValueError without a "player", with a role that is not "player" or
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

# split_roles: what the roles dict becomes
print("only player :", split_roles({"player": player_agent}))
print("with enemy  :", split_roles({"player": player_agent, "enemy_0": enemy_agent}))

###############################################################################
# Build game
###############################################################################

def build_game(layout, seed, agents_by_role):
    """The world of layout `seed` and the agents that play it, ready for run_episode or play.
 
    Returns (world, agents, role_of): agents maps the entity ids of THIS game to the agents, and
    role_of maps the same ids to their roles ("player", "enemy_0", ...). The ids change from
    layout to layout, the roles do not."""
 
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

agents_by_role = {"player": player_agent, "enemy_0": enemy_agent}
world, agents, role_of = build_game(layout, 0, agents_by_role)

###############################################################################
# Run game for agents
###############################################################################

def run_game(layout, seed, agents_by_role):
    """One game. Returns (result, role_of): role_of maps each entity id of THIS game to its role."""
 
    world, agents, role_of = build_game(layout, seed, agents_by_role)
 
    return run_episode(world, agents, reposition=True), role_of

# run_game: one game, and which entity id each role had in it
result, role_of = run_game(layout, 0, {"player": player_agent, "enemy_0": enemy_agent})
print("run_game    :", role_of, result.agent_outcomes, "| steps", result.steps)

###############################################################################
# Run game for agents
###############################################################################

def watch_game(layout, seed, agents_by_role, delay=0.2):
    """Draw one game frame by frame with play(). Same game as run_game with the same seed."""
 
    world, agents, role_of = build_game(layout, seed, agents_by_role)
 
    print("roles in this game:", role_of)
    play(world, agents, delay=delay, role_of=role_of, reposition=True)
 
    return {role_of[i]: agent for i, agent in agents.items()}

# run_game: one game, and which entity id each role had in it
result = watch_game(layout, 0, {"player": player_agent, "enemy_0": enemy_agent})

###############################################################################
# Evaluating agents
###############################################################################
 
def evaluate_agents(layout, agents_by_role, n_games=100, first_seed=0):
    """Play the same layouts first_seed ... first_seed + n_games - 1 with these agents.
 
    Returns (fitness, visits), both keyed by the roles that have an agent:
        fitness : calculate_fitness of the role's outcomes, e.g. {"player": 0.02, "enemy_0": 0.0}
        visits  : for each role, one frozenset of visited cells per game, in order
    Any BaseAgent works, evolving or fixed: the cells come from the record every agent keeps.
    """
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")
 
    outcomes = defaultdict(list)
    visits = defaultdict(list)
 
    for seed in range(first_seed, first_seed + n_games):
        episode, role_of = run_game(layout, seed, agents_by_role)
 
        for agent_id, outcome in episode.agent_outcomes.items():
            role = role_of[agent_id]
            outcomes[role].append(outcome)
            visits[role].append(frozenset(agents_by_role[role].positions))
 
    fitness = {role: calculate_fitness(o) for role, o in outcomes.items()}
    return fitness, dict(visits)

# evaluate_agents: the two lines you used before, now keyed by role (fitness and visited cells)
fitness, visits = evaluate_agents(layout, {"player": player_agent, "enemy_0": enemy_agent})
print("fresh player vs fresh enemy :", fitness)
fitness, visits = evaluate_agents(layout, {"player": player_agent})
print("fresh player vs engine enemy:", fitness)
print("cells visited per game      :", {role: round(sum(map(len, v)) / len(v), 1) for role, v in visits.items()})

###############################################################################
# Scoring a population
###############################################################################
 
def score_population(genomes, 
                     role, 
                     other_agents, 
                     layout, n_games, first_seed,
                     novelty_weight, k_neighbours):
    """Score every genome of one evolving population, playing `role` against the `fixed` agents.
 
    role         : the role the genomes play, "player" or "enemy_<i>".
    other_agents : dict role -> agent for the other roles. They play but do not evolve.
                   {} against the engine's random enemy. It must hold a "player" when the
                   genomes are enemies.
    Returns (rates, novelty, fitness, cells), one entry per genome, in the order of `genomes`:
        rates   : calculate_fitness of the genome's own outcomes (what we report)
        novelty : how different its behaviour is from the other genomes' (see novelty.py)
        fitness : (1 - novelty_weight) * rate + novelty_weight * novelty / biggest (what evolution uses)
        cells   : average number of cells visited per game
 
    Every genome faces the same opponent in the same way: each gets its own clone of the fixed
    agents, so a fixed agent's random choices do not depend on which genomes were played before.
    """
    if role in other_agents:
        raise ValueError(f"role {role!r} is evolving, it cannot also be fixed")
 
    rates, behaviours = [], []
 
    for genome in genomes:
        agents_by_role = {r: agent.clone() for r, agent in other_agents.items()}
        agents_by_role[role] = NeatAgent(genome)
 
        fitness, visits = evaluate_agents(layout, agents_by_role, n_games, first_seed)
 
        rates.append(fitness[role])
        behaviours.append(visits[role])
 
    novelty = novelty_scores(behaviours, k_neighbours)
    fitness = mix_fitness(rates, novelty, novelty_weight)
    cells = [sum(map(len, visits)) / len(visits) for visits in behaviours]
 
    return rates, novelty, fitness, cells
 
# Initialize population for player against default enemy
rng = np.random.default_rng(0)
genomes = [init_genome(N_INPUTS, N_ACTIONS, rng, neat_config) for _ in range(5)]

# Score population
rates, novelty, fitness, cells = score_population(
    genomes,  
   "player", {},
   layout, 20, 0, 
   NOVELTY_WEIGHT, K_NEIGHBOURS
   )

print("player population:", [round(x, 2) for x in rates], [round(x, 2) for x in novelty], [round(x, 1) for x in cells])

###############################################################################
# Scoring population against a defined enemy
###############################################################################
 
# Stage 2: enemy population against the fixed heuristic player
ROLE  = "enemy_0"
FIXED = {"player": HeuristicAgent()}

# Stage 1 would be:  ROLE = "player"  and  FIXED = {}

neat_config["population_size"] = POPULATION_SIZE
state = init_population(N_INPUTS, N_ACTIONS, np.random.default_rng(0), neat_config)
rng = np.random.default_rng(1)

def summary(rates, novelty, cells):
    return (round(sum(rates) / len(rates), 3), max(rates),
            round(sum(novelty) / len(novelty), 3), round(sum(cells) / len(cells), 1))

rates, novelty, fitness, cells = score_population(
    state["genomes"], ROLE, FIXED, layout, N_GAMES, FIRST_SEED, NOVELTY_WEIGHT, K_NEIGHBOURS)
before = summary(rates, novelty, cells)

state, stats = evolve_one_generation(state, fitness, rng, neat_config)
print("generation", state["generation"], "| genomes:", len(state["genomes"]))
print("stats:", stats)

# The new population plays the same layouts, so the comparison is paired
rates, novelty, fitness, cells = score_population(
    state["genomes"], ROLE, FIXED, layout, N_GAMES, FIRST_SEED, NOVELTY_WEIGHT, K_NEIGHBOURS)
after = summary(rates, novelty, cells)

print("                 mean goal | best goal | novelty | cells")
print("generation 0   ", before)
print("generation 1   ", after)

###############################################################################
# Many generations of one population
###############################################################################

def run_generations(role, 
                    other_agents, 
                    n_generations, layout, neat_config,
                    population_size, n_games, train_seed, valid_seed, n_valid_games,
                    novelty_weight, k_neighbours, report_every=5, seed=0, state=None, rng=None):
    """Evolve one population playing `role` against `other_agents` and print a table.

    Every generation plays on fresh layouts (train_seed + generation * n_games ...), so a genome
    cannot just remember the maps. Every `report_every` generations the best genome of the
    population plays n_valid_games on the fixed layouts starting at valid_seed (never trained on),
    and the fitness of every role in those games is printed: "role" is the evolving one, the
    others are the other_agents.

    To resume, pass the `state` a previous call returned. The generation counter of the state
    decides which layouts come next, and the population size must be the same. Pass the SAME `rng`
    object to every call (it is advanced in place): then n generations followed by m more give
    exactly the result of n + m generations in one call. Without `rng` a new one is made from
    `seed`.

    A call scores n_generations + 1 times and evolves n_generations times. A resumed call scores
    its starting generation again (same layouts, same numbers): drop the first row of its history
    when joining histories.

    Returns (state, history): the NEAT state and one dict per generation scored with the
    generation number, the mean and best goal rate, the mean novelty and cells, and the number of
    species (plus "duel" in the reported generations).
    """
    cfg = dict(neat_config)
    cfg["population_size"] = population_size

    if state is None:
        state = init_population(N_INPUTS, N_ACTIONS, np.random.default_rng(seed), cfg)
    elif len(state["genomes"]) != population_size:
        raise ValueError(f"the state has {len(state['genomes'])} genomes, population_size is {population_size}")

    if rng is None:
        rng = np.random.default_rng(seed + 1)

    history = []
    start = time.time()

    print(" gen  goal mean  goal best  novelty  cells  species  seconds | best genome on validation layouts")

    for step in range(n_generations + 1):

        generation = state["generation"]
        first_seed = train_seed + generation * n_games
        rates, novelty, fitness, cells = score_population(
            state["genomes"], role, other_agents, layout, n_games, first_seed,
            novelty_weight, k_neighbours)

        row = {
            "generation": generation,
            "goal_mean": sum(rates) / len(rates),
            "goal_best": max(rates),
            "novelty": sum(novelty) / len(novelty),
            "cells": sum(cells) / len(cells),
            "species": len(state["species_set"]["species"]),
        }
        history.append(row)

        if generation % report_every == 0:
            best = max(range(len(fitness)), key=fitness.__getitem__)

            agents_by_role = {r: agent.clone() for r, agent in other_agents.items()}
            agents_by_role[role] = NeatAgent(state["genomes"][best])
            duel, _ = evaluate_agents(layout, agents_by_role, n_valid_games, valid_seed)

            row["duel"] = duel
            duel_text = "  ".join(f"{r}={v:.2f}" for r, v in sorted(duel.items()))
            print(f"{generation:>4}  {row['goal_mean']:>9.3f}  {row['goal_best']:>9.2f}  "
                  f"{row['novelty']:>7.3f}  {row['cells']:>5.1f}  {row['species']:>7}  "
                  f"{time.time() - start:>7.0f} | {duel_text}")

        if step < n_generations:
            state, stats = evolve_one_generation(state, fitness, rng, cfg)

    return state, history

# Stage 1: player population against the engine enemy (about 1 hour: ~30 s per generation)
player_population, player_history = run_generations(
    "player", 
    {}, 
    120, layout, neat_config,
    population_size=150, n_games=20, train_seed=200_000, valid_seed=210_000, n_valid_games=100,
    novelty_weight=NOVELTY_WEIGHT, k_neighbours=K_NEIGHBOURS, report_every=10)

with open("poet_player_vs_engine.pkl", "wb") as file:
    pickle.dump((player_population, player_history), file)
    
    

# Stage 2: enemy population against the heuristic player (about 30 minutes: ~15 s per generation)
enemy_population, enemy_history = run_generations(
    "enemy_0",
    {"player": NeatAgent(player_population["best"]["genome"])}, 
    450, layout, neat_config,
    population_size=150, n_games=20, train_seed=200_000, valid_seed=210_000, n_valid_games=100,
    novelty_weight=NOVELTY_WEIGHT, k_neighbours=K_NEIGHBOURS, report_every=10
)

with open("poet_enemy_vs_heuristic.pkl", "wb") as file:
    pickle.dump((enemy_population, enemy_history), file)
    
    
# Load from file
with open('poet_player_vs_engine.pkl', 'rb') as f:
    player_population, player_history = pickle.load(f)
    
# Load from file
with open('poet_enemy_vs_heuristic.pkl', 'rb') as f:
    enemy_population, enemy_history = pickle.load(f)
    
    
###############################################################################
# Choosing a genome to look at
###############################################################################
 
def best_genome(
        genomes, 
        role, other_agents, 
        layout, n_games, first_seed,
        novelty_weight, k_neighbours
    ):
    """The genome with the highest goal rate on layouts first_seed ... first_seed + n_games - 1.
 
    Use layouts the population was never trained on. Ties go to the higher fitness (the mix with
    novelty), then to the first in the list. Returns (genome, rate)."""
 
    rates, novelty, fitness, cells = score_population(
        genomes, role, other_agents, layout, n_games, first_seed,
        novelty_weight, k_neighbours)
 
    best = max(range(len(genomes)), key=lambda i: (rates[i], fitness[i]))
 
    return genomes[best], rates[best]

player_champion = player_population["best"]["genome"]

enemy_champion, rate = best_genome(
    enemy_population["genomes"], "enemy_0", {"player": NeatAgent(player_champion)},
    layout, 50, 210_000, NOVELTY_WEIGHT, K_NEIGHBOURS)

player = NeatAgent(player_champion)
enemy  = NeatAgent(enemy_champion)

print("enemy vs its own player    :", evaluate_agents(layout, {"player": player, "enemy_0": enemy}, 200, 220_000)[0])
print("enemy vs heuristic player  :", evaluate_agents(layout, {"player": HeuristicAgent(), "enemy_0": enemy}, 200, 220_000)[0])
print("player champion vs engine enemy:", evaluate_agents(layout, {"player": player }, 200, 220_000)[0])


# run_game: one game, and which entity id each role had in it
result = watch_game(generation_config, 200_001, {"player": NeatAgent(player_champion), "enemy_0": enemy})


###############################################################################
# Play Games
###############################################################################

def play_games(environment, agents_by_role, n_games=100, first_seed=0):
    """Play the seeds first_seed ... first_seed + n_games - 1 with these agents.
 
    environment: a generation config (dict: a new random layout for every seed) or a fixed layout
                 (the seed then decides the starting positions, see uses_reposition).
    Returns (outcomes, visits), both keyed by the roles that have an agent:
        outcomes : the outcome of the role in each game, in order
        visits   : the frozenset of cells the role visited in each game, in order
    Any BaseAgent works, evolving or fixed: the cells come from the record every agent keeps.
    """
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")
 
    outcomes = defaultdict(list)
    visits = defaultdict(list)
 
    for seed in range(first_seed, first_seed + n_games):
        episode, role_of = run_game(environment, seed, agents_by_role)
 
        for agent_id, outcome in episode.agent_outcomes.items():
            role = role_of[agent_id]
            outcomes[role].append(outcome)
            visits[role].append(frozenset(agents_by_role[role].positions))
 
    return dict(outcomes), dict(visits)



###############################################################################
# Minimal criterion
###############################################################################
 
def passes_minimal_criterion(outcomes, low=0.2, high=0.7, timeout_cap=0.3):
    """Is this environment neither too hard nor too easy for these two agents?
 
    outcomes: role -> list of outcomes, from play_games. It must hold "player" and "enemy_0".
    Accepted when BOTH roles reach their goal in a share of the games between `low` and `high`
    (limits included) and the share of games that run out of time is at most `timeout_cap`.
    Returns (accepted, report): report holds the goal rate of every role and the timeout rate, to
    see which condition failed.
    """
    for role in ("player", "enemy_0"):
        if role not in outcomes:
            raise ValueError(f"the minimal criterion needs the outcomes of {role!r}")
 
    report = {role: calculate_fitness(o) for role, o in outcomes.items() if role in ("player", "enemy_0")}
 
    games = outcomes["player"]
    report["timeout"] = sum(o == "Timeout" for o in games) / len(games)
 
    accepted = (
        all(low <= report[role] <= high for role in ("player", "enemy_0"))
        and report["timeout"] <= timeout_cap
    )
    return accepted, report

outcomes = defaultdict(list)
for s in range(100):
    result, role_of = run_game(layout, s, {"player": player, "enemy_0": enemy})
    for agent_id, outcome in result.agent_outcomes.items():
        outcomes[role_of[agent_id]].append(outcome)

print({role: Counter(o) for role, o in outcomes.items()})
print(passes_minimal_criterion(outcomes))

###############################################################################
# POET entry: one environment and the two populations that live in it
###############################################################################

def init_poet(layout, neat_config, population_size, rng):
    """A fresh pair: the layout, one NEAT population per role and a champion per role.
 
    Nothing is trained yet. The populations are the NEAT states that run_generations returns and
    resumes, so an entry can be optimised in rounds without losing anything. The champions start
    as the first genome of each population and are replaced by pick_champion after every round.
    """
    cfg = dict(neat_config)
    cfg["population_size"] = population_size
 
    player  = init_population(N_INPUTS, N_ACTIONS, rng, cfg)
    enemy_0 = init_population(N_INPUTS, N_ACTIONS, rng, cfg)
 
    return {
        "layout": layout,
        "player": player,
        "enemy_0": enemy_0,
        "champions": {"player": player["genomes"][0], "enemy_0": enemy_0["genomes"][0]},
    }

entry = init_poet(layout, neat_config, population_size=10, rng=np.random.default_rng(0))

def champion(entry, role):
    """The genome that plays `role` for the others: the one pick_champion measured as the best.
    Raises ValueError for an entry that has no champions (see with_champions)."""
    if "champions" not in entry:
        raise ValueError("this entry has no 'champions'; an entry from an older run needs with_champions(entry)")
    return entry["champions"][role]

def with_champions(entry):
    """An entry saved before champions existed, with them added from each population's all-time
    best fitness (state["best"], or the first genome when it never evolved). Returns a new dict."""
    champions = {}
    for role in ("player", "enemy_0"):
        state = entry[role]
        champions[role] = state["best"]["genome"] if state["best"] is not None else state["genomes"][0]
    return {**entry, "champions": champions}

print(entry.keys(), len(entry["player"]["genomes"]), len(entry["enemy_0"]["genomes"]))
print("fresh champion is the first genome:", champion(entry, "player") is entry["player"]["genomes"][0])

pair = {"player": NeatAgent(champion(entry, "player")), "enemy_0": NeatAgent(champion(entry, "enemy_0"))}
print(evaluate_agents(entry["layout"], pair, n_games=20)[0])

###############################################################################
# Box 1: optimise the pair
###############################################################################

def pick_champion(entry, role, first_seed, novelty_weight, k_neighbours,
                  n_finalists=5, n_score_games=20, n_final_games=100):
    """The best genome of `role`'s population, measured on games it was not chosen on.
 
    Stage 1: the population is scored on `n_score_games` games from `first_seed` against the other
    role's champion, and the `n_finalists` genomes with the best fitness (the mix with novelty,
    what evolution uses) become finalists.
    Stage 2: the finalists and the current champion play `n_final_games` fresh games (the seeds
    after stage 1) against the same opponent; the highest goal rate wins, and a tie keeps the
    current champion because it is listed first.
 
    Returns (genome, rate): the champion and its goal rate in stage 2. The entry is not modified.
    """
    opponent = "enemy_0" if role == "player" else "player"
    other_agents = {opponent: NeatAgent(champion(entry, opponent))}
    genomes = entry[role]["genomes"]
 
    rates, novelty, fitness, cells = score_population(
        genomes, role, other_agents, entry["layout"], n_score_games, first_seed,
        novelty_weight, k_neighbours)
 
    ranked = sorted(range(len(genomes)), key=lambda i: -fitness[i])[:n_finalists]
    finalists = [champion(entry, role)] + [genomes[i] for i in ranked]
 
    final_seed = first_seed + n_score_games
    results = []
    for genome in finalists:
        agents_by_role = {r: agent.clone() for r, agent in other_agents.items()}
        agents_by_role[role] = NeatAgent(genome)
        rate = evaluate_agents(entry["layout"], agents_by_role, n_final_games, final_seed)[0][role]
        results.append(rate)
 
    best = max(range(len(finalists)), key=lambda i: results[i])       # first maximum: ties keep the champion
    return finalists[best], results[best]

def optimize_entry(entry, n_generations, neat_config, n_games,
                   train_seed, valid_seed, n_valid_games,
                   novelty_weight, k_neighbours, rng, report_every=10,
                   champion_seed=240_000, n_finalists=5, n_score_games=20, n_final_games=100):
    """One round of optimisation of a pair: the player evolves against the enemy's champion, then
    a new player champion is chosen (pick_champion); then the enemy evolves against that NEW
    champion and its own new champion is chosen. Each role gets `n_generations`.
 
    The population size is the one the entry already has (set by init_poet); it is not an argument.
    The environment (entry["layout"]) does not change. Returns (new_entry, history): a new dict
    with the two evolved populations and the new champions (the argument is not modified) and the
    history of each role, {"player": [...], "enemy_0": [...]}, as run_generations reports it.
 
    champion_seed, n_finalists, n_score_games, n_final_games: see pick_champion. The games used to
    choose a champion start at champion_seed and move on with the generation, so no round chooses
    on the games of another.
 
    Rounds can be chained: pass the returned entry back in, with the SAME `rng` object (it is
    advanced in place), and each population continues where it stopped. Two rounds of n
    generations are NOT the same as one round of 2n, because the roles alternate.
    """
    new_entry = dict(entry)
    history = {}
 
    for role, opponent in (("player", "enemy_0"), ("enemy_0", "player")):
        other_agents = {opponent: NeatAgent(champion(new_entry, opponent))}
 
        new_entry[role], history[role] = run_generations(
            role, other_agents, n_generations, entry["layout"], neat_config,
            len(new_entry[role]["genomes"]), n_games, train_seed, valid_seed, n_valid_games,
            novelty_weight, k_neighbours, report_every=report_every,
            state=new_entry[role], rng=rng)
 
        first_seed = champion_seed + new_entry[role]["generation"] * (n_score_games + n_final_games)
        genome, rate = pick_champion(
            new_entry, role, first_seed, novelty_weight, k_neighbours,
            n_finalists, n_score_games, n_final_games)
 
        changed = genome is not new_entry["champions"][role]
        new_entry["champions"] = {**new_entry["champions"], role: genome}
        print(f"  champion {role}: goal rate {rate:.2f} on {n_final_games} fresh games "
              f"({'new' if changed else 'kept'})")
 
    return new_entry, history

entry = init_poet(layout, neat_config, population_size=20, rng=np.random.default_rng(0))
rng = np.random.default_rng(1)

entry, history = optimize_entry(
    entry, 
    n_generations=5, neat_config=neat_config, n_games=10,
    train_seed=200_000, valid_seed=210_000, n_valid_games=50,
    novelty_weight=NOVELTY_WEIGHT, k_neighbours=K_NEIGHBOURS, 
    rng=rng, report_every=5)

###############################################################################
# Box 2: is this pair allowed to have children?
###############################################################################
 
def is_eligible(entry, n_games=100, first_seed=0, low=0.2, high=0.7, timeout_cap=0.3):
    """Can this pair be a parent? Yes when its two champions, playing each other on the pair's own
    layout, pass the minimal criterion: both roles reach their goal in [low, high] of the games and
    at most `timeout_cap` of the games time out (see passes_minimal_criterion).
 
    Use seeds the populations were never trained on (first_seed), so a champion that was only lucky
    on its training games does not count. Returns (eligible, report): the report holds the goal rate
    of each role and the timeout rate.
    """
    agents_by_role = {
        "player": NeatAgent(champion(entry, "player")),
        "enemy_0": NeatAgent(champion(entry, "enemy_0")),
    }
    outcomes, _ = play_games(entry["layout"], agents_by_role, n_games, first_seed)
 
    return passes_minimal_criterion(outcomes, low, high, timeout_cap)

print(is_eligible(entry, n_games=100, first_seed=230_000))

###############################################################################
# Box 3: one child attempt
###############################################################################
 
def make_child(parent, mutation_config, seed, n_games=100, first_seed=0,
               low=0.2, high=0.7, timeout_cap=0.3):
    """One attempt at a child of `parent`.
 
    1. The layout is mutated (mutate_layout, `seed` decides how).
    2. The child starts as an exact COPY of the parent's two populations: nothing is shared with
       the parent, so evolving the child never changes the parent.
    3. The parent's champions play the new layout (play_games, `n_games` from `first_seed`) and
       the minimal criterion decides whether the child is worth keeping.
 
    Returns (child, report): `child` is the new entry, or None when the criterion rejects the
    layout; `report` has the goal rate of each role and the timeout rate of those games.
    """
    layout = mutate_layout(parent["layout"], mutation_config, seed)
 
    child = {
        "layout": layout,
        "player": copy.deepcopy(parent["player"]),
        "enemy_0": copy.deepcopy(parent["enemy_0"]),
        "champions": copy.deepcopy(parent["champions"]),
    }
 
    agents_by_role = {
        "player": NeatAgent(champion(child, "player")),
        "enemy_0": NeatAgent(champion(child, "enemy_0")),
    }
    outcomes, _ = play_games(layout, agents_by_role, n_games, first_seed)
 
    accepted, report = passes_minimal_criterion(outcomes, low, high, timeout_cap)
 
    return (child if accepted else None), report

###############################################################################
# Growing a pair from scratch until it can be a parent
###############################################################################

def grow_entry(entry, max_rounds, n_generations, neat_config, n_games,
               train_seed, valid_seed, n_valid_games, novelty_weight, k_neighbours, rng,
               eligible_seed=230_000, n_eligible_games=100, save_path=None, report_every=10):
    """Optimise `entry` round after round (optimize_entry) until it is eligible (is_eligible).
 
    After every round the pair is checked on the same seeds, eligible_seed ..., which training
    never uses, and the entry is saved to `save_path` (pickle) when given, so a long run can be
    stopped and picked up again from the saved entry (the rng is not saved, pass a new one).
    Stops at the first eligible round, or after `max_rounds`.
 
    Returns (entry, log): the entry after the last round and one dict per round with the round
    number, whether it was eligible and the report of that check.
    """
    log = []
 
    for round_number in range(1, max_rounds + 1):
        entry, _ = optimize_entry(
            entry, n_generations, neat_config, n_games, train_seed, valid_seed, n_valid_games,
            novelty_weight, k_neighbours, rng, report_every)
 
        eligible, report = is_eligible(entry, n_eligible_games, eligible_seed)
        log.append({"round": round_number, "eligible": eligible, "report": report})
 
        print(f"round {round_number:>3} | eligible: {eligible} | "
              + "  ".join(f"{k}={v:.2f}" for k, v in report.items()))
 
        if save_path is not None:
            with open(save_path, "wb") as file:
                pickle.dump(entry, file)
 
        if eligible:
            break
 
    return entry, log

entry = init_poet(layout, neat_config, population_size=50, rng=np.random.default_rng(0))
rng = np.random.default_rng(1)

entry, log = grow_entry(
    entry, max_rounds=20, n_generations=10, neat_config=neat_config, n_games=20,
    train_seed=200_000, valid_seed=210_000, n_valid_games=50,
    novelty_weight=NOVELTY_WEIGHT, k_neighbours=K_NEIGHBOURS, rng=rng,
    save_path="poet_entry_0.pkl", report_every=10)

with open("poet_entry_0.pkl", "rb") as f:
    parent = pickle.load(f)

print("parent:", is_eligible(parent, 100, 230_000)[1])

for seed in range(10):
    child, report = make_child(parent, mutation_config, seed=seed, n_games=100, first_seed=230_000)
    print(seed, child is not None, {k: round(v, 2) for k, v in report.items()})
    
###############################################################################
# Box 5: keep the most novel children
###############################################################################
 
def layout_picture(layout):
    """The layout drawn as plain text, exactly as the engine places it (no repositioning).
 
    render_world colours every wall with an escape code ("\x1b[37m#\x1b[0m" is ten characters
    for one cell), so the codes are removed: each cell is one character and two pictures of the
    same size line up cell by cell."""
    world = World(layout, None, 0)
    world.reset()
    return re.sub(r"\x1b\[[0-9;]*m", "", world.render_world())
 
 
def layout_distance(picture_a, picture_b):
    """How many cells differ between two layout pictures (walls, traps, entities). 0 = same.
    Raises ValueError when the pictures have different sizes: they are not comparable."""
    if len(picture_a) != len(picture_b):
        raise ValueError("layouts of different sizes cannot be compared")
    return sum(a != b for a, b in zip(picture_a, picture_b))
 
 
def select_novel(candidates, pool, n_keep):
    """Pick up to `n_keep` children, the most different from what the pool already holds.
 
    candidates : child entries that passed the minimal criterion.
    pool       : the entries already kept. Their layouts are what a child must differ from.
    A child's novelty is the distance to the NEAREST layout in the pool or among the children
    already picked, so two similar siblings do not both get in. Children that are an exact copy of
    something (distance 0) never get in. Returns the picked children, most novel first.
    """
    seen = [layout_picture(entry["layout"]) for entry in pool]
    pictures = [layout_picture(entry["layout"]) for entry in candidates]
 
    remaining = list(range(len(candidates)))
    picked = []
 
    while remaining and len(picked) < n_keep:
        novelty = {
            i: min((layout_distance(pictures[i], s) for s in seen), default=float("inf"))
            for i in remaining
        }
        best = max(remaining, key=novelty.__getitem__)       # ties go to the first candidate
 
        if novelty[best] == 0:
            break                                            # nothing left that is new
 
        picked.append(candidates[best])
        seen.append(pictures[best])
        remaining.remove(best)
 
    return picked
 
 
def admit(pool, children, max_size):
    """The pool with the children added at the end. When it is over `max_size`, the OLDEST
    entries (the first ones) leave. Returns a new list; the arguments are not modified."""
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")
    return (list(pool) + list(children))[-max_size:]

pool = [parent]
children = []
for seed in range(10):
    child, report = make_child(parent, mutation_config, seed=seed, n_games=100, first_seed=230_000)
    if child is not None:
        children.append(child)

pictures = [layout_picture(c["layout"]) for c in children]
parent_picture = layout_picture(parent["layout"])
print("distance to the parent:", [layout_distance(p, parent_picture) for p in pictures])

picked = select_novel(children, pool, n_keep=3)
print("picked layouts (indexes):", [children.index(c) for c in picked])

###############################################################################
# The loop: boxes 1 to 5 together
###############################################################################
 
def run_poet(pool, n_iterations, optimize_settings, mutation_config, rng,
             n_mutate=1, n_children=5, max_new=2, max_size=4,
             n_games=100, first_seed=230_000, save_path=None):
    """The POET loop for a pool of entries (oldest first). Each iteration:
 
    1. Every `n_mutate` iterations, REPRODUCE, from the pool as it stands:
         - every entry is checked (is_eligible) and the eligible ones are the parents,
         - every parent makes `n_children` attempts (make_child; a new mutation seed from `rng`
           for each); the ones that pass the minimal criterion are the candidates,
         - the `max_new` most novel candidates are picked (select_novel) and admitted (admit):
           when the pool is over `max_size`, the oldest entries leave.
    2. OPTIMISE every entry of the pool, the new ones included (optimize_entry).
 
    Reproducing first means a pair that is in the window at the start of the iteration gets to
    have children before its own optimisation can move it out of the window.
 
    optimize_settings: the arguments of optimize_entry that do not change, as a dict:
        n_generations, neat_config, n_games, train_seed, valid_seed, n_valid_games,
        novelty_weight, k_neighbours, report_every.
    n_games, first_seed: the games of every eligibility and child check (see is_eligible).
 
    Returns (pool, log): the final pool and one dict per iteration with the eligibility report of
    each entry that was checked (why it was or was not a parent), the number of eligible parents,
    candidates and children admitted, and the wall count of every entry in the pool at the end.
    With `save_path` the pool is saved (pickle) after every iteration. The pool passed in is not
    modified.
    """
    pool = list(pool)
    log = []
 
    for iteration in range(1, n_iterations + 1):
 
        reports, parents, candidates, admitted = [], [], [], 0
 
        if iteration % n_mutate == 0:
            for entry in pool:
                eligible, report = is_eligible(entry, n_games, first_seed)
                reports.append(report)
                if eligible:
                    parents.append(entry)
 
            for parent in parents:
                for _ in range(n_children):
                    seed = int(rng.integers(2**31))
                    child, _ = make_child(parent, mutation_config, seed, n_games, first_seed)
                    if child is not None:
                        candidates.append(child)
 
            chosen = select_novel(candidates, pool, max_new)
            admitted = len(chosen)
            pool = admit(pool, chosen, max_size)
 
        pool = [optimize_entry(entry, rng=rng, **optimize_settings)[0] for entry in pool]
 
        log.append({
            "iteration": iteration,
            "reports": reports,
            "eligible": len(parents),
            "candidates": len(candidates),
            "admitted": admitted,
            "walls": [layout_picture(entry["layout"]).count("#") for entry in pool],
        })
 
        checked = "  ".join(
            f"[{r['player']:.2f}/{r['enemy_0']:.2f}/{r['timeout']:.2f}]" for r in reports)
        print(f"iteration {iteration:>3} | pool {len(pool)} | eligible parents {len(parents)} | "
              f"candidates {len(candidates)} | admitted {admitted} | walls {log[-1]['walls']}")
        if reports:
            print(f"              checked before reproducing (player/enemy/timeout): {checked}")
            if not parents:
                print("              WARNING: no entry is eligible, nothing reproduced this iteration")
 
        if save_path is not None:
            with open(save_path, "wb") as file:
                pickle.dump(pool, file)
 
    return pool, log

with open("poet_entry_0.pkl", "rb") as f:
    pool = [pickle.load(f)]

optimize_settings = dict(
    n_generations=10, neat_config=neat_config, n_games=20,
    train_seed=200_000, valid_seed=210_000, n_valid_games=50,
    novelty_weight=NOVELTY_WEIGHT, k_neighbours=K_NEIGHBOURS, report_every=10)

mutation_config = {**mutation_config, "n_wall_mutations": 12, "max_wall_density": 0.4}

pool, log = run_poet(
    pool, n_iterations=6, optimize_settings=optimize_settings,
    mutation_config=mutation_config, rng=np.random.default_rng(4),
    n_mutate=1, n_children=5, max_new=2, max_size=4,
    save_path="poet_pool_v2.pkl")