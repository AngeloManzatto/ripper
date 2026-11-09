"""
Created on Thu Nov  5 07:01:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""Choosing the champion of a population: the genome that plays this role for the others.

The best genome of a generation is a poor choice on its own: it was picked from many on the same
games, so part of its score is luck (the winner's curse). The choice is made in two stages, and the
second one is played on games nobody was chosen on.
"""

from agents.neat_agent.agent import NeatAgent
from experiments.poet.neat_scoring import score_population
from poet.games import goal_rate, play_games

###############################################################################
# Pick Champion
###############################################################################

def pick_champion(genomes, champion, role, other_agents, layout, first_seed,
                  novelty_weight, k_neighbours, config, make_agent=NeatAgent):
    """The champion of `genomes` for `role`, measured on games it was not chosen on.

    genomes      : the population.
    champion     : the genome that is champion now. It is always one of the finalists, so a new
                   champion has to beat it.
    role         : the role the genomes play.
    other_agents : role -> agent for the other roles (what the champion plays against).
    first_seed   : first seed of this choice; stage 1 uses n_score_games seeds from here and
                   stage 2 uses n_final_games seeds right after them.
    novelty_weight, k_neighbours : the mix that evolution uses (see score_population).
    config       : a ChampionConfig (n_finalists, n_score_games, n_final_games).
    make_agent   : genome -> the agent that plays it.

    Stage 1: the population is scored on n_score_games games and the n_finalists genomes with the
             best fitness (the mix with novelty, what evolution uses) become finalists.
    Stage 2: the current champion and the finalists play n_final_games fresh games against clones of
             the same opponents; the highest goal rate wins. A tie goes to the champion (it is
             listed first), and then to the better fitness of stage 1.

    Returns (genome, rate): the champion and its goal rate in stage 2. Nothing is modified.
    """
    if len(genomes) == 0:
        raise ValueError("pick_champion needs at least one genome")

    _, _, fitness, _ = score_population(
        genomes, role, other_agents, layout, config.n_score_games, first_seed,
        novelty_weight, k_neighbours, make_agent=make_agent)

    ranked = sorted(range(len(genomes)), key=lambda i: -fitness[i])[:config.n_finalists]
    finalists = [champion] + [genomes[i] for i in ranked]

    final_seed = first_seed + config.n_score_games
    rates = []
    for genome in finalists:
        agents_by_role = {r: agent.clone() for r, agent in other_agents.items()}
        agents_by_role[role] = make_agent(genome)

        outcomes = play_games(layout, agents_by_role, n_games=config.n_final_games, first_seed=final_seed)
        rates.append(goal_rate(outcomes[role]))

    best = max(range(len(finalists)), key=rates.__getitem__)       # the first maximum: ties keep the champion
    return finalists[best], rates[best]