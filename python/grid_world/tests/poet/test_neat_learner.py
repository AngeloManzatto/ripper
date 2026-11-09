"""
Created on Fri Nov  6 09:02:13 2026

@author: root
"""

"""Tests of experiments/poet/neat_learner.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.

Most tests use stand-ins for the NEAT library, the scoring and the champion choice: genomes are
small dicts, scoring gives scripted numbers. The last tests use the real library and the real
engine and are skipped when they cannot be imported.
"""

import copy

import random
import sys
import traceback
import unittest
from contextlib import contextmanager

from experiments.poet import neat_learner

from experiments.poet.learner_config import ChampionConfig, TrainConfig
from experiments.poet.neat_learner import NeatLearner
from poet.learner import Learner


@contextmanager
def raises(exception, match=None):
    """The block must raise `exception` (with `match` inside the message when given)."""
    try:
        yield
    except exception as error:
        if match is not None and match not in str(error):
            raise AssertionError(f"{exception.__name__} message {str(error)!r} does not contain {match!r}")
    else:
        raise AssertionError(f"{exception.__name__} was not raised")


@contextmanager
def patched(module, **replacements):
    """Replace attributes of `module` and ALWAYS put the originals back, even when a test fails."""
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


###############################################################################
# Stand-ins: genomes are dicts, the library and the scoring are scripted
###############################################################################

TRAIN = TrainConfig(population_size=6, n_generations=3, n_games=5, n_valid_games=20, k_neighbours=3)
CHAMPION = ChampionConfig(n_finalists=2, n_score_games=4, n_final_games=8)
NEAT_CONFIG = {"population_size": 999, "mutation_rate": 0.5}


class Agent:
    def __init__(self, genome):
        self.genome = genome


class Opponent:
    def __init__(self):
        self.clones = 0

    def clone(self):
        self.clones += 1
        return Opponent()


@contextmanager
def stand_ins(rates=None, fail_at=None):
    """Replace the library, the scoring and the champion choice. Yields the calls that were made.

    rates   : the goal rate of every genome in every scoring (default: the position / 10).
    fail_at : the number of the scoring call (1, 2, ...) that raises.
    """
    calls = {"init": [], "score": [], "evolve": [], "pick": []}

    def fake_init(n_inputs, n_outputs, rng, config):
        calls["init"].append(dict(n_inputs=n_inputs, n_outputs=n_outputs, draw=rng.integers(10**9), config=config))
        genomes = [{"id": i, "generation": 0} for i in range(config["population_size"])]
        return {"generation": 0, "genomes": genomes}

    def fake_score(genomes, role, other_agents, layout, n_games, first_seed, novelty_weight, k_neighbours,
                   make_agent=None):
        calls["score"].append(dict(genomes=genomes, role=role, other_agents=other_agents, layout=layout,
                                   n_games=n_games, first_seed=first_seed, novelty_weight=novelty_weight,
                                   k_neighbours=k_neighbours, make_agent=make_agent))
        if fail_at == len(calls["score"]):
            raise RuntimeError("scoring failed")
        scripted = list(rates) if rates is not None else [i / 10 for i in range(len(genomes))]
        return scripted, [1.0] * len(genomes), [float(i) for i in range(len(genomes))], [4.0] * len(genomes)

    def fake_evolve(state, fitness, rng, config):
        calls["evolve"].append(dict(state=state, fitness=fitness, draw=rng.integers(10**9), config=config))
        generation = state["generation"] + 1
        genomes = [{"id": i, "generation": generation} for i in range(len(state["genomes"]))]
        return {"generation": generation, "genomes": genomes}, {}

    def fake_pick(genomes, champion, role, other_agents, layout, first_seed, novelty_weight, k_neighbours, config,
                  make_agent=None):
        calls["pick"].append(dict(genomes=genomes, champion=champion, role=role, other_agents=other_agents,
                                  layout=layout, first_seed=first_seed, novelty_weight=novelty_weight,
                                  k_neighbours=k_neighbours, config=config, make_agent=make_agent))
        return genomes[1], 0.5

    with patched(neat_learner, init_population=fake_init, score_population=fake_score,
                 evolve_one_generation=fake_evolve, pick_champion=fake_pick):
        yield calls


def make_learner(role="player", seed=0, train=TRAIN, champion=CHAMPION):
    return NeatLearner(role, NEAT_CONFIG, train, champion, seed=seed, make_agent=Agent)


###############################################################################
# Building the learner
###############################################################################

def test_it_is_a_learner():
    with stand_ins():
        assert isinstance(make_learner(), Learner)


def test_an_unknown_role_is_refused():
    with stand_ins():
        with raises(ValueError, match="enemy_1"):
            NeatLearner("enemy_1", NEAT_CONFIG, TRAIN, CHAMPION, make_agent=Agent)


def test_the_champion_games_may_not_start_inside_the_validation_games():
    clash = ChampionConfig(champion_seed=210_005)          # the validation games are 210_000 ... 210_019
    with stand_ins():
        with raises(ValueError, match="champion_seed"):
            NeatLearner("player", NEAT_CONFIG, TRAIN, clash, make_agent=Agent)


def test_the_population_has_the_size_of_the_train_config_and_the_callers_dict_is_untouched():
    with stand_ins() as calls:
        learner = make_learner()

    assert calls["init"][0]["config"]["population_size"] == 6
    assert calls["init"][0]["config"]["mutation_rate"] == 0.5          # the other settings pass through
    assert NEAT_CONFIG["population_size"] == 999                      # not modified
    assert len(learner.state["genomes"]) == 6


def test_the_network_shape_comes_from_the_agent_and_the_actions():
    from agents.neat_agent.agent import N_INPUTS
    from core.actions import N_ACTIONS

    with stand_ins() as calls:
        make_learner()

    assert (calls["init"][0]["n_inputs"], calls["init"][0]["n_outputs"]) == (N_INPUTS, N_ACTIONS)


def test_the_same_seed_gives_the_same_population_and_another_seed_another():
    with stand_ins() as calls:
        make_learner(seed=3)
        make_learner(seed=3)
        make_learner(seed=4)

    first, second, third = (c["draw"] for c in calls["init"])
    assert first == second
    assert first != third


def test_the_champion_starts_as_the_first_genome_and_nothing_is_trained_yet():
    with stand_ins():
        learner = make_learner()

    assert learner.champion is learner.state["genomes"][0]
    assert learner.champion_rate is None
    assert learner.history == []


###############################################################################
# agent()
###############################################################################

def test_agent_plays_with_the_champion_and_is_a_new_object_every_time():
    with stand_ins():
        learner = make_learner()

    first, second = learner.agent(), learner.agent()
    assert first.genome is learner.champion and second.genome is learner.champion
    assert first is not second


def test_after_training_agent_plays_with_the_new_champion_not_with_the_first_genome():
    with stand_ins():
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))

    assert learner.champion is not learner.state["genomes"][0]          # the stand-in picks the second genome
    assert learner.agent().genome is learner.champion


###############################################################################
# train(): the generations
###############################################################################

def test_train_scores_and_evolves_n_generations_on_moving_seeds():
    with stand_ins() as calls:
        learner = make_learner()
        learner.train("LAYOUT", Opponent(), random.Random(0))

    assert len(calls["score"]) == 3 and len(calls["evolve"]) == 3
    # generation g plays the seeds train_seed + g * n_games
    assert [c["first_seed"] for c in calls["score"]] == [200_000, 200_005, 200_010]
    assert all(c["n_games"] == 5 and c["layout"] == "LAYOUT" for c in calls["score"])
    assert all(c["novelty_weight"] == 0.2 and c["k_neighbours"] == 3 for c in calls["score"])
    assert learner.state["generation"] == 3


def test_the_generations_keep_counting_across_calls():
    with stand_ins() as calls:
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))
        learner.train("L", Opponent(), random.Random(1))

    seeds = [c["first_seed"] for c in calls["score"]]
    assert seeds == [200_000 + 5 * g for g in range(6)], "no two generations may play the same games"
    assert learner.state["generation"] == 6


def test_every_generation_scores_the_population_of_that_generation():
    with stand_ins() as calls:
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))

    assert [c["genomes"][0]["generation"] for c in calls["score"]] == [0, 1, 2]
    assert calls["evolve"][0]["fitness"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]       # the fitness of the scoring


def test_the_player_plays_against_the_enemy_and_the_enemy_against_the_player():
    opponent = Opponent()
    with stand_ins() as calls:
        make_learner("player").train("L", opponent, random.Random(0))
        make_learner("enemy_0").train("L", opponent, random.Random(0))

    player_calls = [c for c in calls["score"] if c["role"] == "player"]
    enemy_calls = [c for c in calls["score"] if c["role"] == "enemy_0"]
    assert player_calls and enemy_calls
    assert all(c["other_agents"] == {"enemy_0": opponent} for c in player_calls)
    assert all(c["other_agents"] == {"player": opponent} for c in enemy_calls)
    assert all(c["make_agent"] is Agent for c in calls["score"])


def test_the_learner_never_changes_the_opponent():
    opponent = Opponent()
    with stand_ins():
        make_learner().train("L", opponent, random.Random(0))

    assert vars(opponent) == {"clones": 0}      # cloning for every genome is the scoring's job


def test_the_library_gets_its_randomness_from_the_rng_given_and_only_from_it():
    with stand_ins() as calls:
        make_learner().train("L", Opponent(), random.Random(7))
        make_learner().train("L", Opponent(), random.Random(7))
        make_learner().train("L", Opponent(), random.Random(8))

    draws = [c["draw"] for c in calls["evolve"]]
    assert draws[0:3] == draws[3:6], "the same rng must give the same evolution"
    assert draws[0:3] != draws[6:9], "another rng must give another one"


def test_train_uses_up_the_rng_it_is_given():
    rng = random.Random(7)
    with stand_ins():
        make_learner().train("L", Opponent(), rng)

    assert rng.random() != random.Random(7).random()


def test_train_returns_none():
    with stand_ins():
        assert make_learner().train("L", Opponent(), random.Random(0)) is None


###############################################################################
# train(): the report
###############################################################################

def test_history_has_one_row_per_generation_scored():
    with stand_ins(rates=[0.0, 0.2, 0.4, 0.0, 0.0, 0.2]) as calls:
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))
        learner.train("L", Opponent(), random.Random(1))

    assert [row["generation"] for row in learner.history] == [0, 1, 2, 3, 4, 5]
    row = learner.history[0]
    assert abs(row["goal_mean"] - 0.8 / 6) < 1e-12
    assert row["goal_best"] == 0.4
    assert row["novelty"] == 1.0 and row["cells"] == 4.0


###############################################################################
# train(): the champion
###############################################################################

def test_the_champion_is_chosen_from_the_evolved_population_against_the_same_opponent():
    opponent = Opponent()
    with stand_ins() as calls:
        learner = make_learner()
        first_champion = learner.champion
        learner.train("LAYOUT", opponent, random.Random(0))

    pick = calls["pick"][0]
    assert len(calls["pick"]) == 1
    assert pick["genomes"][0]["generation"] == 3                   # the population after the last evolution
    assert pick["champion"] is first_champion
    assert pick["role"] == "player" and pick["layout"] == "LAYOUT"
    assert pick["other_agents"] == {"enemy_0": opponent}
    assert pick["config"] is CHAMPION and pick["make_agent"] is Agent
    assert (pick["novelty_weight"], pick["k_neighbours"]) == (0.2, 3)

    assert learner.champion is learner.state["genomes"][1]         # what pick_champion returned
    assert learner.champion_rate == 0.5


def test_the_champion_games_follow_the_generation_and_never_repeat():
    with stand_ins() as calls:
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))
        learner.train("L", Opponent(), random.Random(1))

    first, second = (c["first_seed"] for c in calls["pick"])
    assert first == 240_000 + 3 * 12                               # generation 3, 4 + 8 games per choice
    assert second == 240_000 + 6 * 12
    assert second - first >= CHAMPION.games_per_round


###############################################################################
# train(): the guard and failures
###############################################################################

def test_training_may_reach_the_last_generation_below_the_validation_games():
    with stand_ins() as calls:
        learner = make_learner()
        last = TRAIN.last_generation()
        learner.state["generation"] = last - 2                        # 3 generations: last - 2, last - 1, last
        learner.train("L", Opponent(), random.Random(0))

    assert calls["score"][-1]["first_seed"] + TRAIN.n_games <= TRAIN.valid_seed


def test_training_that_would_reach_the_validation_seeds_is_refused_before_anything_changes():
    with stand_ins() as calls:
        learner = make_learner()
        learner.state["generation"] = TRAIN.last_generation() - 1     # one generation too many
        champion, state, history = learner.champion, learner.state, list(learner.history)

        with raises(ValueError, match="validation"):
            learner.train("L", Opponent(), random.Random(0))

    assert calls["score"] == [] and calls["evolve"] == [] and calls["pick"] == []
    assert learner.champion is champion and learner.state is state and learner.history == history


def test_the_games_of_the_last_generation_stay_below_valid_seed():
    last = TRAIN.last_generation()
    assert TRAIN.train_seed + (last + 1) * TRAIN.n_games <= TRAIN.valid_seed
    assert TRAIN.train_seed + (last + 2) * TRAIN.n_games > TRAIN.valid_seed


def test_a_failure_halfway_leaves_the_learner_as_it_was():
    with stand_ins(fail_at=2):
        learner = make_learner()
        champion, state = learner.champion, learner.state

        with raises(RuntimeError, match="scoring failed"):
            learner.train("L", Opponent(), random.Random(0))

    assert learner.champion is champion and learner.state is state
    assert learner.state["generation"] == 0 and learner.history == [] and learner.champion_rate is None


###############################################################################
# clone()
###############################################################################

def test_clone_is_the_same_learner_in_another_object():
    with stand_ins():
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))
        twin = learner.clone()

    assert twin is not learner
    assert twin.state == learner.state and twin.state is not learner.state
    assert twin.champion == learner.champion and twin.champion is not learner.champion
    assert twin.history == learner.history and twin.history is not learner.history
    assert twin.champion_rate == learner.champion_rate
    assert twin.agent() is not learner.agent()


def test_a_champion_that_is_in_the_population_is_one_object_in_the_clone_too():
    with stand_ins():
        learner = make_learner()
        twin = learner.clone()

    assert twin.champion is twin.state["genomes"][0]


def test_training_the_clone_does_not_change_the_original_and_the_other_way_round():
    with stand_ins():
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))
        twin = learner.clone()
        before = copy.deepcopy((learner.state, learner.champion, learner.history))

        for _ in range(2):
            twin.train("L", Opponent(), random.Random(1))
        assert (learner.state, learner.champion, learner.history) == before, "the clone changed the original"

        before_twin = copy.deepcopy((twin.state, twin.champion, twin.history))
        learner.train("L", Opponent(), random.Random(2))
        assert (twin.state, twin.champion, twin.history) == before_twin, "the original changed the clone"


def test_clone_after_the_same_training_keeps_training_in_step_with_the_original():
    with stand_ins() as calls:
        learner = make_learner()
        learner.train("L", Opponent(), random.Random(0))
        twin = learner.clone()
        learner.train("L", Opponent(), random.Random(5))
        twin.train("L", Opponent(), random.Random(5))

    assert learner.state == twin.state and learner.champion == twin.champion
    assert learner.history == twin.history


###############################################################################
# The real NEAT library and the real engine (skipped when they cannot be imported)
###############################################################################

def _real():
    try:
        from engine import GenerationConfig, generate_valid_layout
        from neat.config import default_config
        from agents.heuristic_agent.agent import HeuristicAgent
        from poet.games import play_games
    except ImportError as error:
        raise unittest.SkipTest(f"real library not available here: {error}")

    layout = generate_valid_layout(GenerationConfig(width=12, height=12, wall_density=0.1, num_enemies=1,
                                                    num_traps=0, min_player_goal_distance=6))
    return layout, default_config(), HeuristicAgent, play_games


REAL_TRAIN = TrainConfig(population_size=8, n_generations=2, n_games=3, n_valid_games=10, k_neighbours=3)
REAL_CHAMPION = ChampionConfig(n_finalists=2, n_score_games=3, n_final_games=5)


def test_real_a_short_training_runs_and_the_champion_plays():
    layout, config, Heuristic, play_games = _real()

    learner = NeatLearner("player", config, REAL_TRAIN, REAL_CHAMPION, seed=0)
    learner.train(layout, Heuristic(), random.Random(0))

    assert learner.state["generation"] == 2
    assert len(learner.history) == 2 and learner.champion_rate is not None
    outcomes = play_games(layout, {"player": learner.agent(), "enemy_0": Heuristic()}, n_games=3, first_seed=230_000)
    assert len(outcomes["player"]) == 3


def test_real_the_enemy_role_trains_against_a_player():
    layout, config, Heuristic, play_games = _real()

    learner = NeatLearner("enemy_0", config, REAL_TRAIN, REAL_CHAMPION, seed=0)
    learner.train(layout, Heuristic(), random.Random(0))

    assert learner.state["generation"] == 2
    outcomes = play_games(layout, {"player": Heuristic(), "enemy_0": learner.agent()}, n_games=3, first_seed=230_000)
    assert len(outcomes["enemy_0"]) == 3


def test_real_the_same_seeds_give_the_same_learner():
    layout, config, Heuristic, play_games = _real()

    def trained():
        learner = NeatLearner("player", config, REAL_TRAIN, REAL_CHAMPION, seed=0)
        learner.train(layout, Heuristic(), random.Random(3))
        return learner

    first, second = trained(), trained()
    assert first.history == second.history
    assert first.champion_rate == second.champion_rate

    games = lambda learner: play_games(layout, {"player": learner.agent(), "enemy_0": Heuristic()},
                                        n_games=10, first_seed=230_000)
    assert games(first) == games(second)


def test_real_the_library_does_not_modify_the_state_it_is_given():
    # train() relies on this: it keeps the old state until everything has worked
    layout, config, Heuristic, play_games = _real()
    from neat.evolution import evolve_one_generation
    import numpy as np

    learner = NeatLearner("player", config, REAL_TRAIN, REAL_CHAMPION, seed=0)
    before = copy.deepcopy(learner.state)
    evolve_one_generation(learner.state, [0.0] * 8, np.random.default_rng(0), learner.neat_config)

    assert learner.state["generation"] == before["generation"] == 0
    assert learner.state["genomes"] == before["genomes"]


def test_real_a_played_learner_can_still_be_cloned():
    # The agents that played hold engine observations that cannot be copied; the learner holds none
    layout, config, Heuristic, play_games = _real()

    learner = NeatLearner("player", config, REAL_TRAIN, REAL_CHAMPION, seed=0)
    agent = learner.agent()
    play_games(layout, {"player": agent, "enemy_0": Heuristic()}, n_games=2, first_seed=230_000)
    learner.train(layout, Heuristic(), random.Random(0))

    twin = learner.clone()
    assert twin.champion == learner.champion


###############################################################################
# Runner
###############################################################################

def run_all():
    # Only the tests defined in THIS file: a console that stays open still holds the tests of earlier runs
    tests = [(name, test) for name, test in globals().items()
             if name.startswith("test_") and callable(test)]
    passed, skipped, failed = 0, 0, []

    for name, test in tests:
        try:
            test()
        except unittest.SkipTest as reason:
            skipped += 1
            print(f"skip  {name}  ({reason})")
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")

    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed" + (f": {failed}" if failed else ""))

    try:
        get_ipython()                    # inside Spyder / IPython: just report
    except NameError:
        if failed:
            sys.exit(1)                  # from a plain terminal: a failing exit code


if __name__ == "__main__":
    run_all()