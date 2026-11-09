"""
Created on Wed Oct 21 10:50:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from agents.neat_agent.agent import NeatAgent
from engine import GenerationConfig, World, generate_valid_layout
from experiments.neat_player.novelty import novelty_scores, mix_fitness
from runtime.evaluate import evaluate
from runtime.runner import run_episode

###############################################################################
# Genome fitness
###############################################################################

def genome_summary(
    genome: dict,
    n_episodes: int,
    seed: int,
    generation_kwargs: dict | None = None,
    world_config=None,
):
    """
    Play one genome as the player and return the whole EvalSummary (outcome counts,
    mean steps, ...). genome_fitness is the goal rate of this summary.

    The genome plays on the layouts seed, seed + 1, ..., seed + n_episodes - 1, using
    the same `evaluate` as every other agent. The layouts and the world seeds depend
    only on `seed`, so every genome given the same arguments plays exactly the same
    games. Invalid genomes raise, they are never scored.
    """

    agent = NeatAgent(genome)

    return evaluate(
        agent,
        n_episodes=n_episodes,
        seed=seed,
        generation_kwargs=generation_kwargs,
        world_config=world_config,
    )


def genome_fitness(
    genome: dict,
    n_episodes: int,
    seed: int,
    generation_kwargs: dict | None = None,
    world_config=None,
) -> float:
    """
    Fitness of one genome: the fraction of episodes in which the player reaches
    the goal, in [0.0, 1.0]. Only GoalReached counts; Trapped, Caught and Timeout
    all score zero.

    Use training seeds here, never the benchmark evaluation seeds.
    """

    summary = genome_summary(genome, n_episodes, seed, generation_kwargs, world_config)

    return float(summary.rate("GoalReached"))


###############################################################################
# Population fitness
###############################################################################

def evaluate_population(
    genomes: list,
    n_episodes: int,
    seed: int,
    generation_kwargs: dict | None = None,
    world_config=None,
) -> list:
    """
    Fitness of every genome, in the same order as `genomes`.

    All genomes play the same layouts (same `seed`), one after the other in this
    process. If one genome is invalid the error is raised and no list is returned.

    run_evolution wants evaluate(genomes) -> fitnesses, so give it
        lambda genomes: evaluate_population(genomes, N_EPISODES, SEED)
    """

    if len(genomes) == 0:
        raise ValueError("evaluate_population needs at least one genome")

    return [
        genome_fitness(genome, n_episodes, seed, generation_kwargs, world_config)
        for genome in genomes
    ]


###############################################################################
# Fitness with novelty
###############################################################################

class RecordingAgent(NeatAgent):
    """A NeatAgent that also remembers the cell it stood on at every decision of the episode
    (self.positions, emptied at every reset)."""

    def __init__(self, genome, **kwargs):
        super().__init__(genome, **kwargs)
        self.positions = []

    def on_reset(self):
        self.positions = []

    def select_action(self, obs):
        self.positions.append(tuple(obs.position))
        return super().select_action(obs)


def play_with_positions(
    genome: dict,
    n_episodes: int,
    seed: int,
    generation_kwargs: dict | None = None,
    world_config=None,
):
    """
    Play one genome on the layouts seed, seed + 1, ..., seed + n_episodes - 1, built exactly as
    runtime.evaluate builds them, and record where it went.

    Returns (goal_rate, visits): the fraction of episodes that ended in GoalReached (the same
    number genome_fitness gives), and one frozenset of visited cells per episode, in order.
    """

    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    agent = RecordingAgent(genome)
    goals = 0
    visits = []

    for i in range(n_episodes):
        episode_seed = seed + i

        layout = generate_valid_layout(GenerationConfig(**(generation_kwargs or {}), seed=episode_seed))
        if layout is None:
            raise RuntimeError(f"Could not generate a valid layout for seed {episode_seed}")

        world = World(layout, world_config, episode_seed)
        player_id = world.reset().observation.player_id

        result = run_episode(world, {player_id: agent})

        goals += result.outcome == "GoalReached"
        visits.append(frozenset(agent.positions))

    return goals / n_episodes, visits


def evaluate_population_novelty(
    genomes: list,
    n_episodes: int,
    seed: int,
    weight: float,
    k: int,
    generation_kwargs: dict | None = None,
    world_config=None,
):
    """
    Fitness of every genome when novelty is mixed with the goal rate.

    Every genome plays the same layouts (as in evaluate_population). Returns (fitnesses,
    goal_rates), both in the order of `genomes`:
        goal_rates : the goal rate of each genome (identical to evaluate_population's result)
        fitnesses  : (1 - weight) * goal rate + weight * novelty / biggest novelty, where the novelty
                     of a genome is the mean behaviour distance to its k nearest genomes
                     (see novelty.py). With weight 0 the fitnesses are the goal rates.

    Give the fitnesses to the evolution and use the goal rates for everything you report.
    Needs at least two genomes (novelty compares them).
    """

    if len(genomes) < 2:
        raise ValueError("evaluate_population_novelty needs at least two genomes")

    goal_rates, behaviours = [], []
    for genome in genomes:
        rate, visits = play_with_positions(genome, n_episodes, seed, generation_kwargs, world_config)
        goal_rates.append(rate)
        behaviours.append(visits)

    novelty = novelty_scores(behaviours, k)
    return mix_fitness(goal_rates, novelty, weight), goal_rates