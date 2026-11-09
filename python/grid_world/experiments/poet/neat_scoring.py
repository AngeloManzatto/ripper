"""
Created on Thu Nov  5 06:52:04 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################


"""Scoring a population of genomes: how well does each one play, and how differently?

This is the evaluation inside the NEAT learner. Every genome plays the same games against the same
opponent, and gets a goal rate, a novelty (how differently it moves than the others) and the fitness
that mixes both, which is what evolution uses.
"""

from collections import defaultdict

from agents.neat_agent.agent import NeatAgent
from agents.neat_agent.novelty import mix_fitness, novelty_scores
from poet.games import goal_rate, run_game


def play_with_visits(layout, agents_by_role, n_games, first_seed):
    """Play the seeds first_seed ... first_seed + n_games - 1 on this layout.

    Returns (outcomes, visits), both keyed by the roles that have an agent:
        outcomes : the outcome of the role in each game, in seed order
        visits   : the frozenset of cells the role's agent stood on in each game, in seed order
    The cells come from the record every BaseAgent keeps (agent.positions), so any agent works.
    Raises ValueError when n_games < 1.
    """
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")

    outcomes = defaultdict(list)
    visits = defaultdict(list)

    for seed in range(first_seed, first_seed + n_games):
        result, role_of = run_game(layout, seed, agents_by_role)

        for entity_id, outcome in result.agent_outcomes.items():
            role = role_of[entity_id]
            outcomes[role].append(outcome)
            visits[role].append(frozenset(agents_by_role[role].positions))

    return dict(outcomes), dict(visits)


def score_population(genomes, role, other_agents, layout, n_games, first_seed,
                     novelty_weight, k_neighbours, make_agent=NeatAgent):
    """Score every genome of one evolving population playing `role` against `other_agents`.

    genomes      : the population, in order.
    role         : the role the genomes play, "player" or "enemy_<i>".
    other_agents : role -> agent for the other roles. They play but do not evolve.
                   Raises ValueError when it also holds `role`.
    make_agent   : genome -> the agent that plays it (NeatAgent).
    Every genome plays the same games (the same seeds, the same layout) against its OWN clones of
    the other agents, so what an opponent does never depends on which genomes played before.

    Returns (rates, novelty, fitness, cells), one entry per genome, in the order of `genomes`:
        rates   : the share of games in which the genome reached its goal
        novelty : how different its movement is from the other genomes' (see novelty.py)
        fitness : (1 - novelty_weight) * rate + novelty_weight * novelty / biggest novelty
        cells   : average number of cells it visited per game
    """
    if role in other_agents:
        raise ValueError(f"role {role!r} is evolving, it cannot also be fixed")

    rates, behaviours = [], []

    for genome in genomes:
        agents_by_role = {r: agent.clone() for r, agent in other_agents.items()}
        agents_by_role[role] = make_agent(genome)

        outcomes, visits = play_with_visits(layout, agents_by_role, n_games, first_seed)

        rates.append(goal_rate(outcomes[role]))
        behaviours.append(visits[role])

    novelty = novelty_scores(behaviours, k_neighbours)
    fitness = mix_fitness(rates, novelty, novelty_weight)
    cells = [sum(map(len, visits)) / len(visits) for visits in behaviours]

    return rates, novelty, fitness, cells