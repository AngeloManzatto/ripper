"""
Tests for experiments/neat_player/fitness.py: play_with_positions and evaluate_population_novelty.
No engine is needed: for evaluate_population_novelty play_with_positions is replaced by a fake; for
play_with_positions the engine pieces (generate_valid_layout, GenerationConfig, World, run_episode)
are replaced by fakes that walk the agent through scripted positions.

play_with_positions(genome, n_episodes, seed, generation_kwargs=None, world_config=None)
    -> (goal_rate, visits)

    - episode i uses the layout of seed + i, built with generation_kwargs and that seed, and a World
      made with world_config and that seed;
    - goal_rate is the fraction of episodes with outcome GoalReached;
    - visits has one frozenset of the cells the agent stood on per episode, in order, and nothing is
      carried from one episode to the next;
    - n_episodes < 1 raises ValueError; a layout that cannot be generated raises RuntimeError.

evaluate_population_novelty(genomes, n_episodes, seed, weight, k, generation_kwargs=None, world_config=None)
    -> (fitnesses, goal_rates)

    - play_with_positions is called once per genome, in order, with the same arguments;
    - goal_rates are the goal rates it returned;
    - fitnesses = mix_fitness(goal_rates, novelty_scores(visits, k), weight);
    - weight 0 gives fitnesses equal to the goal rates;
    - fewer than two genomes raises, and so does an error of play_with_positions;
    - the genomes are not changed.

Real games are checked by the first real run.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy
from types import SimpleNamespace

import experiments.neat_player.fitness as fitness_module
from agents.neat_agent.agent import N_INPUTS
from core.actions import N_ACTIONS
from core.cells import FREE
from experiments.neat_player.fitness import evaluate_population_novelty, play_with_positions

###############################################################################
# helpers
###############################################################################

class FakePlay:
    """Stands in for play_with_positions: returns the next (rate, visits) of a list, records every call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, genome, n_episodes, seed, generation_kwargs=None, world_config=None):
        self.calls.append({"genome": genome, "n_episodes": n_episodes, "seed": seed,
                           "generation_kwargs": generation_kwargs, "world_config": world_config})
        return self.outputs[len(self.calls) - 1]


def with_fake(outputs, fn):
    fake = FakePlay(outputs)
    original = fitness_module.play_with_positions
    fitness_module.play_with_positions = fake
    try:
        return fn(fake)
    finally:
        fitness_module.play_with_positions = original


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


# three genomes, two games each: g0 and g1 go to the same cells, g2 goes elsewhere
OUTPUTS = [
    (0.5, [frozenset({1, 2}), frozenset({5})]),
    (0.25, [frozenset({1, 2}), frozenset({5})]),
    (0.0, [frozenset({8, 9}), frozenset({7})]),
]


def genomes():
    return [{"id": 0}, {"id": 1}, {"id": 2}]

def small_genome():
    nodes = {i: {"type": "input"} for i in range(N_INPUTS)}
    for k in range(N_ACTIONS):
        nodes[N_INPUTS + k] = {"type": "output", "bias": 0.0, "activation": "identity"}
    return {"n_inputs": N_INPUTS, "n_outputs": N_ACTIONS, "nodes": nodes, "connections": {}}


def obs_at(position, size=5):
    return SimpleNamespace(grid=[[FREE] * size for _ in range(size)], position=position,
                           kind="player", status="Alive", id=0)


class FakeEngine:
    """Replaces the engine pieces used by play_with_positions.

    script: {episode seed: (outcome, [positions the agent stands on, in order])}
    Records the layout requests and the World arguments."""

    def __init__(self, script, missing_layout_seed=None):
        self.script = script
        self.missing_layout_seed = missing_layout_seed
        self.layout_requests = []
        self.worlds = []

    def generation_config(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def generate_valid_layout(self, config):
        self.layout_requests.append(config)
        if config.seed == self.missing_layout_seed:
            return None
        return {"seed": config.seed}

    def world(self, layout, config, seed):
        self.worlds.append({"layout": layout, "config": config, "seed": seed})
        return SimpleNamespace(
            seed=seed, reset=lambda: SimpleNamespace(observation=SimpleNamespace(player_id=7)))

    def run_episode(self, world, agents):
        agent = agents[7]
        agent.reset()
        outcome, positions = self.script[world.seed]
        for position in positions:
            agent.act(obs_at(position))
        return SimpleNamespace(outcome=outcome, steps=len(positions))


def with_engine(engine, fn):
    names = {"GenerationConfig": engine.generation_config, "generate_valid_layout": engine.generate_valid_layout,
             "World": engine.world, "run_episode": engine.run_episode}
    original = {n: getattr(fitness_module, n) for n in names}
    for n, f in names.items():
        setattr(fitness_module, n, f)
    try:
        return fn()
    finally:
        for n, f in original.items():
            setattr(fitness_module, n, f)

###############################################################################
# tests
###############################################################################

def test_goal_rates_are_those_of_play_with_positions_in_order():
    fitnesses, goal_rates = with_fake(OUTPUTS, lambda fake: evaluate_population_novelty(genomes(), 2, 10, 0.2, 1))
    assert goal_rates == [0.5, 0.25, 0.0]


def test_weight_zero_gives_the_goal_rates():
    fitnesses, goal_rates = with_fake(OUTPUTS, lambda fake: evaluate_population_novelty(genomes(), 2, 10, 0.0, 1))
    assert fitnesses == goal_rates == [0.5, 0.25, 0.0]


def test_novelty_is_mixed_in():
    # k=1: g0 and g1 have distance 0 to each other, so novelty 0; g2 has distance 1.0 to both
    fitnesses, goal_rates = with_fake(OUTPUTS, lambda fake: evaluate_population_novelty(genomes(), 2, 10, 0.2, 1))
    assert close(fitnesses[0], 0.8 * 0.5) and close(fitnesses[1], 0.8 * 0.25) and close(fitnesses[2], 0.2 * 1.0)


def test_weight_one_ranks_only_by_novelty():
    fitnesses, _ = with_fake(OUTPUTS, lambda fake: evaluate_population_novelty(genomes(), 2, 10, 1.0, 1))
    assert fitnesses == [0.0, 0.0, 1.0]


def test_k_is_passed_to_the_novelty():
    # with k=2 the novelty of g0 is the mean of its distances to g1 (0.0) and g2 (1.0) = 0.5
    fitnesses, _ = with_fake(OUTPUTS, lambda fake: evaluate_population_novelty(genomes(), 2, 10, 1.0, 2))
    assert close(fitnesses[0], 0.5 / 1.0) and close(fitnesses[2], 1.0)


def test_every_genome_plays_with_the_same_arguments_in_order():
    gs = genomes()
    kwargs, config = {"wall_density": 0.2}, object()
    fake = with_fake(OUTPUTS, lambda f: (evaluate_population_novelty(gs, 2, 10, 0.2, 1, kwargs, config), f)[1])
    assert [c["genome"]["id"] for c in fake.calls] == [0, 1, 2]
    for call in fake.calls:
        assert call["n_episodes"] == 2 and call["seed"] == 10
        assert call["generation_kwargs"] is kwargs and call["world_config"] is config


def test_fewer_than_two_genomes_raises():
    assert with_fake([OUTPUTS[0]], lambda fake: raises(ValueError, evaluate_population_novelty, [{"id": 0}], 2, 10, 0.2, 1))
    assert with_fake([], lambda fake: raises(ValueError, evaluate_population_novelty, [], 2, 10, 0.2, 1))


def test_an_error_while_playing_is_raised_not_hidden():
    def boom(*args, **kwargs):
        raise RuntimeError("invalid genome")

    original = fitness_module.play_with_positions
    fitness_module.play_with_positions = boom
    try:
        assert raises(RuntimeError, evaluate_population_novelty, genomes(), 2, 10, 0.2, 1)
    finally:
        fitness_module.play_with_positions = original


def test_genomes_are_not_changed():
    gs = genomes()
    before = copy.deepcopy(gs)
    with_fake(OUTPUTS, lambda fake: evaluate_population_novelty(gs, 2, 10, 0.2, 1))
    assert gs == before

###############################################################################
# play_with_positions
###############################################################################

SCRIPT = {
    10: ("GoalReached", [(1, 1), (1, 2), (1, 1)]),
    11: ("Timeout", [(3, 3), (3, 3)]),
    12: ("GoalReached", [(0, 0)]),
    13: ("Caught", [(2, 2), (2, 3)]),
}


def test_play_goal_rate_and_visits():
    engine = FakeEngine(SCRIPT)
    rate, visits = with_engine(engine, lambda: play_with_positions(small_genome(), 4, 10))
    assert rate == 0.5
    assert visits == [frozenset({(1, 1), (1, 2)}), frozenset({(3, 3)}), frozenset({(0, 0)}), frozenset({(2, 2), (2, 3)})]
    assert all(isinstance(v, frozenset) for v in visits)


def test_play_uses_layout_and_world_of_seed_plus_i():
    engine = FakeEngine(SCRIPT)
    kwargs, config = {"wall_density": 0.2, "num_enemies": 1}, object()
    with_engine(engine, lambda: play_with_positions(small_genome(), 3, 10, kwargs, config))
    assert [c.seed for c in engine.layout_requests] == [10, 11, 12]
    assert all(c.wall_density == 0.2 and c.num_enemies == 1 for c in engine.layout_requests)
    assert [w["seed"] for w in engine.worlds] == [10, 11, 12]
    assert [w["layout"] for w in engine.worlds] == [{"seed": 10}, {"seed": 11}, {"seed": 12}]
    assert all(w["config"] is config for w in engine.worlds)


def test_play_without_generation_kwargs_uses_only_the_seed():
    engine = FakeEngine(SCRIPT)
    with_engine(engine, lambda: play_with_positions(small_genome(), 1, 10))
    assert vars(engine.layout_requests[0]) == {"seed": 10}


def test_play_episodes_do_not_share_cells():
    engine = FakeEngine({0: ("Timeout", [(1, 1)]), 1: ("Timeout", [(2, 2)])})
    _, visits = with_engine(engine, lambda: play_with_positions(small_genome(), 2, 0))
    assert visits == [frozenset({(1, 1)}), frozenset({(2, 2)})]


def test_play_all_episodes_lost_gives_rate_zero():
    engine = FakeEngine({0: ("Caught", [(1, 1)]), 1: ("Trapped", [(1, 1)])})
    rate, _ = with_engine(engine, lambda: play_with_positions(small_genome(), 2, 0))
    assert rate == 0.0


def test_play_errors():
    engine = FakeEngine(SCRIPT)
    assert with_engine(engine, lambda: raises(ValueError, play_with_positions, small_genome(), 0, 10))
    engine = FakeEngine(SCRIPT, missing_layout_seed=11)
    assert with_engine(engine, lambda: raises(RuntimeError, play_with_positions, small_genome(), 3, 10))

###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"evaluate_population_novelty: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()