"""
Created on Fri Oct 23 13:42:33 2026

@author: Angelo Antonio Manzatto
"""

"""
Tests for experiments/neat_player/fitness.py: genome_fitness.

genome_fitness(genome, n_episodes, seed, generation_kwargs=None, world_config=None) -> float

    Builds a NeatAgent from the genome, plays it with runtime.evaluate.evaluate
    on layouts seed ... seed + n_episodes - 1, and returns the fraction of
    episodes that ended with "GoalReached" (a float in [0, 1]).

    - The arguments are passed to evaluate unchanged (n_episodes, seed,
      generation_kwargs, world_config).
    - Only GoalReached counts: Trapped, Caught and Timeout all score zero.
    - The genome is not changed. Invalid genomes (wrong shape, cycle...) raise,
      they are never scored 0.
    - No randomness of its own: the same genome and arguments give the same
      number (evaluate fixes the layouts and world seeds from `seed`).

These tests replace `evaluate` with a fake that records how it was called and
returns chosen outcome counts, so no engine is needed. Playing real games is
checked by the first real run.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy

import experiments.neat_player.fitness as fitness_module
from experiments.neat_player.fitness import genome_fitness
from agents.neat_agent.agent import NeatAgent, N_INPUTS
from core.actions import N_ACTIONS
from runtime.evaluate import EvalSummary, OUTCOMES

###############################################################################
# helpers
###############################################################################

def genome(biases=(0.0, 0.0, 0.0, 0.0), n_inputs=N_INPUTS, n_outputs=N_ACTIONS):
    nodes = {i: {"type": "input"} for i in range(n_inputs)}
    for k in range(n_outputs):
        bias = biases[k] if k < len(biases) else 0.0
        nodes[n_inputs + k] = {"type": "output", "bias": bias, "activation": "sigmoid"}
    return {"n_inputs": n_inputs, "n_outputs": n_outputs, "nodes": nodes, "connections": {}}


def summary_with(goal=0, trapped=0, caught=0, timeout=0):
    counts = {"GoalReached": goal, "Trapped": trapped, "Caught": caught, "Timeout": timeout}
    n = goal + trapped + caught + timeout
    return EvalSummary(n_episodes=n, outcome_counts=counts, mean_steps=10.0, results=[])


class FakeEvaluate:
    """Stands in for runtime.evaluate.evaluate and records every call."""

    def __init__(self, summary):
        self.summary = summary
        self.calls = []

    def __call__(self, agent, n_episodes=100, seed=0, generation_kwargs=None, world_config=None):
        self.calls.append({"agent": agent, "n_episodes": n_episodes, "seed": seed,
                           "generation_kwargs": generation_kwargs, "world_config": world_config})
        return self.summary


def with_fake(summary, fn):
    """Run fn(fake) with evaluate replaced, always restoring the real one."""
    original = fitness_module.evaluate
    fake = FakeEvaluate(summary)
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
# the score
###############################################################################

def test_fitness_is_the_goal_rate():
    for goal, others, expected in ((0, 10, 0.0), (10, 0, 1.0), (3, 7, 0.3), (1, 3, 0.25), (37, 63, 0.37)):
        summary = summary_with(goal=goal, timeout=others)
        value = with_fake(summary, lambda fake: genome_fitness(genome(), 100, 0))
        assert abs(value - expected) < 1e-12, (goal, others)


def test_only_goal_reached_counts():
    for kind in ("trapped", "caught", "timeout"):
        summary = summary_with(**{kind: 10})
        assert with_fake(summary, lambda fake: genome_fitness(genome(), 10, 0)) == 0.0
    mixed = summary_with(goal=2, trapped=3, caught=2, timeout=3)
    assert abs(with_fake(mixed, lambda fake: genome_fitness(genome(), 10, 0)) - 0.2) < 1e-12


def test_returns_a_plain_float():
    value = with_fake(summary_with(goal=1, timeout=1), lambda fake: genome_fitness(genome(), 2, 0))
    assert type(value) is float


def test_fitness_is_in_zero_one():
    for goal in range(0, 11):
        value = with_fake(summary_with(goal=goal, timeout=10 - goal),
                          lambda fake: genome_fitness(genome(), 10, 0))
        assert 0.0 <= value <= 1.0

###############################################################################
# how evaluate is called
###############################################################################

def test_arguments_reach_evaluate_unchanged():
    kwargs = {"width": 7, "height": 7}
    config = object()

    def call(fake):
        genome_fitness(genome(), 25, 100_123, generation_kwargs=kwargs, world_config=config)
        return fake.calls

    calls = with_fake(summary_with(goal=1), call)
    assert len(calls) == 1
    assert calls[0]["n_episodes"] == 25
    assert calls[0]["seed"] == 100_123
    assert calls[0]["generation_kwargs"] is kwargs
    assert calls[0]["world_config"] is config


def test_defaults_for_generation_kwargs_and_world_config_are_none():
    calls = with_fake(summary_with(goal=1),
                      lambda fake: (genome_fitness(genome(), 5, 3), fake.calls)[1])
    assert calls[0]["generation_kwargs"] is None
    assert calls[0]["world_config"] is None


def test_the_agent_played_is_a_neat_agent_made_from_the_genome():
    g = genome(biases=(0.0, 5.0, 0.0, 0.0))
    calls = with_fake(summary_with(goal=1), lambda fake: (genome_fitness(g, 5, 3), fake.calls)[1])
    agent = calls[0]["agent"]
    assert isinstance(agent, NeatAgent)
    assert agent.network == NeatAgent(g).network


def test_each_call_builds_a_fresh_agent():
    def call(fake):
        genome_fitness(genome(), 5, 3)
        genome_fitness(genome(), 5, 3)
        return fake.calls

    calls = with_fake(summary_with(goal=1), call)
    assert calls[0]["agent"] is not calls[1]["agent"]


def test_different_genomes_play_with_their_own_agent():
    a, b = genome(biases=(5.0, 0.0, 0.0, 0.0)), genome(biases=(0.0, 0.0, 0.0, 5.0))

    def call(fake):
        genome_fitness(a, 5, 3)
        genome_fitness(b, 5, 3)
        return fake.calls

    calls = with_fake(summary_with(goal=1), call)
    assert calls[0]["agent"].network != calls[1]["agent"].network

###############################################################################
# safety
###############################################################################

def test_the_genome_is_not_changed():
    g = genome(biases=(0.5, 1.5, 2.5, 3.5))
    snapshot = copy.deepcopy(g)
    with_fake(summary_with(goal=1), lambda fake: genome_fitness(g, 5, 3))
    assert g == snapshot


def test_an_invalid_genome_raises_instead_of_scoring_zero():
    for bad in (genome(n_inputs=3), genome(n_outputs=2)):
        assert with_fake(summary_with(goal=1), lambda fake: raises(ValueError, genome_fitness, bad, 5, 3))
    cyclic = genome()
    cyclic["nodes"][N_INPUTS + 4] = {"type": "hidden", "bias": 0.0, "activation": "sigmoid"}
    cyclic["nodes"][N_INPUTS + 5] = {"type": "hidden", "bias": 0.0, "activation": "sigmoid"}
    cyclic["connections"] = {
        0: {"src": N_INPUTS + 4, "dst": N_INPUTS + 5, "weight": 1.0, "enabled": True},
        1: {"src": N_INPUTS + 5, "dst": N_INPUTS + 4, "weight": 1.0, "enabled": True},
        2: {"src": N_INPUTS + 4, "dst": N_INPUTS, "weight": 1.0, "enabled": True},
    }
    assert with_fake(summary_with(goal=1), lambda fake: raises(ValueError, genome_fitness, cyclic, 5, 3))


def test_an_invalid_genome_never_reaches_evaluate():
    def call(fake):
        raises(ValueError, genome_fitness, genome(n_inputs=3), 5, 3)
        return fake.calls

    assert with_fake(summary_with(goal=1), call) == []


def test_errors_from_evaluate_are_not_swallowed():
    original = fitness_module.evaluate

    def boom(*args, **kwargs):
        raise RuntimeError("Could not generate a valid layout")

    fitness_module.evaluate = boom
    try:
        assert raises(RuntimeError, genome_fitness, genome(), 5, 3)
    finally:
        fitness_module.evaluate = original


def test_same_inputs_same_fitness():
    summary = summary_with(goal=4, timeout=6)
    first = with_fake(summary, lambda fake: genome_fitness(genome(), 10, 7))
    second = with_fake(summary, lambda fake: genome_fitness(genome(), 10, 7))
    assert first == second == 0.4


def test_outcome_names_match_the_evaluator():
    assert "GoalReached" in OUTCOMES

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"fitness: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()