"""
Created on Fri Oct 23 14:20:04 2026

@author: Angelo Antonio Manzatto

Tests for experiments/neat_player/fitness.py: genome_summary and evaluate_population.

genome_summary(genome, n_episodes, seed, generation_kwargs=None, world_config=None) -> EvalSummary
evaluate_population(genomes, n_episodes, seed, generation_kwargs=None, world_config=None) -> list[float]

    - genome_summary plays one genome with runtime.evaluate.evaluate and returns the
      summary object untouched, passing every argument through.
    - evaluate_population returns one fitness (the goal rate) per genome, in the order of
      the genomes, every genome playing with the same arguments.
    - An empty list or an invalid genome raises; no partial list is returned.
    - The genomes are not changed.

`evaluate` is replaced by a fake, as in test_fitness.py. Real games are checked by the
first real run.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import experiments.neat_player.fitness as fitness_module
from experiments.neat_player.fitness import genome_summary, evaluate_population
from agents.neat_agent.agent import N_INPUTS
from core.actions import N_ACTIONS
from neat.network import compile_genome
from runtime.evaluate import EvalSummary

###############################################################################
# helpers
###############################################################################

def genome(bias=0.0, n_inputs=N_INPUTS, n_outputs=N_ACTIONS):
    nodes = {i: {"type": "input"} for i in range(n_inputs)}
    for k in range(n_outputs):
        nodes[n_inputs + k] = {"type": "output", "bias": bias, "activation": "sigmoid"}
    return {"n_inputs": n_inputs, "n_outputs": n_outputs, "nodes": nodes, "connections": {}}


def summary_with(goal=0, other=0):
    counts = {"GoalReached": goal, "Trapped": other, "Caught": 0, "Timeout": 0}
    return EvalSummary(n_episodes=goal + other, outcome_counts=counts, mean_steps=10.0, results=[])


class FakeEvaluate:
    """Stands in for evaluate: returns the next summary of a list and records every call."""

    def __init__(self, summaries):
        self.summaries = list(summaries)
        self.calls = []

    def __call__(self, agent, n_episodes=100, seed=0, generation_kwargs=None, world_config=None):
        self.calls.append({"agent": agent, "n_episodes": n_episodes, "seed": seed,
                           "generation_kwargs": generation_kwargs, "world_config": world_config})
        return self.summaries[len(self.calls) - 1]


def with_fake(summaries, fn):
    original = fitness_module.evaluate
    fake = FakeEvaluate(summaries)
    fitness_module.evaluate = fake
    try:
        return fn(fake)
    finally:
        fitness_module.evaluate = original


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False

###############################################################################
# genome_summary
###############################################################################

def test_summary_is_the_object_evaluate_returned():
    s = summary_with(goal=2, other=2)
    result = with_fake([s], lambda fake: genome_summary(genome(), 4, 7))
    assert result is s


def test_summary_passes_every_argument_through():
    kwargs, config = {"n_walls": 3}, object()
    def go(fake):
        genome_summary(genome(), 12, 345, generation_kwargs=kwargs, world_config=config)
        return fake.calls[0]
    call = with_fake([summary_with(1, 1)], go)
    assert call["n_episodes"] == 12 and call["seed"] == 345
    assert call["generation_kwargs"] is kwargs and call["world_config"] is config


def test_summary_invalid_genome_raises():
    assert with_fake([summary_with()], lambda fake: raises(ValueError, genome_summary, genome(n_inputs=3), 1, 0))

###############################################################################
# evaluate_population
###############################################################################

def test_one_fitness_per_genome_in_order():
    summaries = [summary_with(1, 3), summary_with(3, 1), summary_with(0, 4)]
    result = with_fake(summaries, lambda fake: evaluate_population([genome(0.0), genome(1.0), genome(2.0)], 4, 0))
    assert result == [0.25, 0.75, 0.0]
    assert all(isinstance(x, float) for x in result)


def test_each_call_gets_its_own_genome_in_order():
    genomes = [genome(0.0), genome(1.0), genome(2.0)]
    def go(fake):
        evaluate_population(genomes, 4, 0)
        return fake.calls
    calls = with_fake([summary_with(1, 1)] * 3, go)
    assert len(calls) == 3
    for call, g in zip(calls, genomes):
        assert call["agent"].network == compile_genome(g)


def test_every_genome_plays_with_the_same_arguments():
    kwargs, config = {"n_walls": 3}, object()
    def go(fake):
        evaluate_population([genome(0.0), genome(1.0)], 12, 345, generation_kwargs=kwargs, world_config=config)
        return fake.calls
    for call in with_fake([summary_with(1, 1)] * 2, go):
        assert call["n_episodes"] == 12 and call["seed"] == 345
        assert call["generation_kwargs"] is kwargs and call["world_config"] is config


def test_empty_population_raises():
    assert raises(ValueError, evaluate_population, [], 4, 0)


def test_invalid_genome_raises_and_returns_nothing():
    genomes = [genome(0.0), genome(n_inputs=3), genome(1.0)]
    assert with_fake([summary_with(1, 1)] * 3, lambda fake: raises(ValueError, evaluate_population, genomes, 4, 0))


def test_genomes_are_not_changed():
    genomes = [genome(0.0), genome(1.0)]
    before = copy.deepcopy(genomes)
    with_fake([summary_with(1, 1)] * 2, lambda fake: evaluate_population(genomes, 4, 0))
    assert genomes == before


def test_works_with_a_tuple_of_genomes():
    result = with_fake([summary_with(1, 1)] * 2, lambda fake: evaluate_population((genome(0.0), genome(1.0)), 2, 0))
    assert result == [0.5, 0.5]

###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"evaluate_population: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()