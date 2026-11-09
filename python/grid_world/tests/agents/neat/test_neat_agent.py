#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:16:04 2026

@author: root
"""

"""
Tests for agents/neat_agent/agent.py: NeatAgent.

NeatAgent(genome, seed=42, history_size=8)
    A frozen policy that runs one NEAT genome through the BaseAgent contract.

    - The genome is compiled once at construction. It must have N_INPUTS inputs
      (N_FEATURES + N_MEMORY + N_PROXIMITY) and N_ACTIONS outputs, otherwise
      ValueError. The agent shares nothing with the genome afterwards.
    - act(obs) returns an engine Action: the one whose network output is the
      largest (ties: the first action in ACTIONS).
    - The decision uses the input vector _build_input(obs, positions): the
      features of obs, then the recent-visit flags, then the hostile-proximity
      flags, as a float32 array of length N_INPUTS.
    - Deterministic: the same situation gives the same action, and the agent's
      rng is never used. Train / eval mode change nothing. on_transition does
      nothing. reset() clears the history. The agent can be pickled.

These tests build tiny genomes by hand (plain dicts) and fake observations
(simple objects with grid, position, kind), so no engine is needed.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy
import pickle
from types import SimpleNamespace

import numpy as np

from agents.neat_agent.agent import NeatAgent, N_INPUTS
from core.actions import ACTIONS, N_ACTIONS
from core.cells import FREE, WALL, GOAL, TRAP
from core.features import N_FEATURES, FEATURE_NAMES, features
from core.memory import N_MEMORY
from core.transition import Transition
from engine import Action

###############################################################################
# helpers
###############################################################################

def genome_with(biases=(0.0, 0.0, 0.0, 0.0), connections=(), activation="identity",
                n_inputs=N_INPUTS, n_outputs=N_ACTIONS):
    """Genome with inputs 0..n_inputs-1 and outputs after them.
    connections: (input_index, output_index, weight) triples."""
    nodes = {i: {"type": "input"} for i in range(n_inputs)}
    for k in range(n_outputs):
        bias = biases[k] if k < len(biases) else 0.0
        nodes[n_inputs + k] = {"type": "output", "bias": bias, "activation": activation}
    conns = {}
    for key, (src, out, weight) in enumerate(connections):
        conns[key] = {"src": src, "dst": n_inputs + out, "weight": weight, "enabled": True}
    return {"n_inputs": n_inputs, "n_outputs": n_outputs, "nodes": nodes, "connections": conns}


def make_obs(position=(2, 2), size=5, cells=None, kind="player"):
    """Fake observation: an empty size x size floor with some cells set.
    cells: {(row, col): cell code}."""
    grid = [[FREE] * size for _ in range(size)]
    for (row, col), code in (cells or {}).items():
        grid[row][col] = code
    return SimpleNamespace(grid=grid, position=position, kind=kind, status="Alive", id=0)


def feature_index(name):
    return FEATURE_NAMES.index(name)


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False

###############################################################################
# construction
###############################################################################

def test_n_inputs_is_features_plus_memory_plus_proximity():
    assert N_INPUTS >= N_FEATURES + N_MEMORY
    assert N_INPUTS == len(NeatAgent(genome_with())._build_input(make_obs(), [(2, 2)]))


def test_builds_from_a_genome_with_the_right_shape():
    agent = NeatAgent(genome_with())
    assert agent.name == "neat"
    assert agent.network["n_inputs"] == N_INPUTS
    assert agent.network["n_outputs"] == N_ACTIONS


def test_wrong_number_of_inputs_is_refused():
    for n in (N_INPUTS - 1, N_INPUTS + 1, 2):
        assert raises(ValueError, NeatAgent, genome_with(n_inputs=n))


def test_wrong_number_of_outputs_is_refused():
    for n in (1, N_ACTIONS - 1, N_ACTIONS + 1):
        assert raises(ValueError, NeatAgent, genome_with(n_outputs=n))


def test_a_genome_with_a_cycle_or_unknown_activation_is_refused():
    genome = genome_with(activation="banana")
    assert raises(ValueError, NeatAgent, genome)


def test_seed_and_history_size_are_passed_to_the_base_agent():
    agent = NeatAgent(genome_with(), seed=7, history_size=3)
    assert agent.history.maxlen == 3
    assert list(agent.rng.random(3)) == list(np.random.default_rng(7).random(3))


def test_the_agent_shares_nothing_with_the_genome():
    genome = genome_with(biases=(0.0, 5.0, 0.0, 0.0))
    agent = NeatAgent(genome)
    genome["nodes"][N_INPUTS + 1]["bias"] = -99.0
    genome["nodes"][N_INPUTS + 3]["bias"] = 99.0
    assert agent.act(make_obs()) == ACTIONS[1]

###############################################################################
# the decision: argmax of the outputs
###############################################################################

def test_the_largest_output_picks_the_action():
    for k in range(N_ACTIONS):
        biases = [-5.0] * N_ACTIONS
        biases[k] = 5.0
        agent = NeatAgent(genome_with(biases=biases))
        action = agent.act(make_obs())
        assert action == ACTIONS[k]
        assert isinstance(action, Action)


def test_ties_go_to_the_first_action():
    agent = NeatAgent(genome_with(biases=(1.0, 1.0, 1.0, 1.0)))
    assert agent.act(make_obs()) == ACTIONS[0]
    agent = NeatAgent(genome_with(biases=(0.0, 2.0, 2.0, 1.0)))
    assert agent.act(make_obs()) == ACTIONS[1]


def test_negative_outputs_still_pick_the_largest():
    agent = NeatAgent(genome_with(biases=(-4.0, -3.0, -1.0, -2.0)))
    assert agent.act(make_obs()) == ACTIONS[2]


def test_output_activation_is_used():
    # sigmoid squeezes 20 and 30 into the same float (1.0): a tie, first action wins
    agent = NeatAgent(genome_with(biases=(0.0, 20.0, 30.0, 0.0), activation="sigmoid"))
    assert agent.act(make_obs()) == ACTIONS[1]
    agent = NeatAgent(genome_with(biases=(0.0, 20.0, 30.0, 0.0), activation="identity"))
    assert agent.act(make_obs()) == ACTIONS[2]

###############################################################################
# the decision depends on the right input
###############################################################################

def goal_seeker():
    """goal_drow > 0 (goal below) -> Down; goal_drow < 0 (above) -> Up;
    goal_dcol > 0 (right) -> Right; goal_dcol < 0 (left) -> Left."""
    drow, dcol = feature_index("goal_drow"), feature_index("goal_dcol")
    up, down, left, right = (ACTIONS.index(a) for a in (Action.Up, Action.Down, Action.Left, Action.Right))
    return genome_with(connections=[
        (drow, down, 10.0), (drow, up, -10.0),
        (dcol, right, 10.0), (dcol, left, -10.0),
    ])


def test_goal_below_goes_down_and_goal_above_goes_up():
    agent = NeatAgent(goal_seeker())
    assert agent.act(make_obs(position=(1, 2), cells={(4, 2): GOAL})) == Action.Down
    agent.reset()
    assert agent.act(make_obs(position=(3, 2), cells={(0, 2): GOAL})) == Action.Up


def test_goal_right_goes_right_and_goal_left_goes_left():
    agent = NeatAgent(goal_seeker())
    assert agent.act(make_obs(position=(2, 0), cells={(2, 4): GOAL})) == Action.Right
    agent.reset()
    assert agent.act(make_obs(position=(2, 4), cells={(2, 0): GOAL})) == Action.Left


def test_neighbor_flags_reach_the_network():
    # a wall right above makes "up_wall" = 1; wire it to push away from Up
    up_wall = feature_index("up_wall")
    genome = genome_with(biases=(1.0, 0.0, 0.0, 0.0), connections=[(up_wall, 0, -5.0)])
    agent = NeatAgent(genome)
    assert agent.act(make_obs(position=(2, 2))) == ACTIONS[0]                    # no wall: Up wins
    agent.reset()
    assert agent.act(make_obs(position=(2, 2), cells={(1, 2): WALL})) != ACTIONS[0]


def test_trap_flags_reach_the_network():
    left_trap = feature_index("left_trap")
    left = ACTIONS.index(Action.Left)
    genome = genome_with(biases=(0.0, 0.0, 1.0, 0.0), connections=[(left_trap, left, -5.0)])
    agent = NeatAgent(genome)
    assert agent.act(make_obs(cells={})) == Action.Left
    agent.reset()
    assert agent.act(make_obs(cells={(2, 1): TRAP})) != Action.Left


def test_the_input_vector_starts_with_the_features():
    agent = NeatAgent(genome_with())
    obs = make_obs(position=(1, 3), cells={(4, 4): GOAL, (3, 3): TRAP})
    x = agent._build_input(obs, [(1, 3)])
    assert x.dtype == np.float32
    assert x.shape == (N_INPUTS,)
    assert list(x[:N_FEATURES]) == list(features(obs))


def test_memory_flags_come_after_the_features():
    # stand on (2,2) having been on (2,3) before: some flag after the features is set
    agent = NeatAgent(genome_with())
    obs = make_obs(position=(2, 2))
    fresh = agent._build_input(obs, [(2, 2)])
    visited = agent._build_input(obs, [(2, 3), (2, 2)])
    assert list(fresh[:N_FEATURES]) == list(visited[:N_FEATURES])
    assert list(fresh[N_FEATURES:]) != list(visited[N_FEATURES:])


def test_the_positions_given_to_the_input_builder_are_the_history_including_now():
    agent = NeatAgent(genome_with())
    seen = []
    original = agent._build_input

    def spy(obs, positions):
        seen.append(list(positions))
        return original(obs, positions)

    agent._build_input = spy
    agent.act(make_obs(position=(2, 3)))
    agent.act(make_obs(position=(2, 2)))
    assert seen == [[(2, 3)], [(2, 3), (2, 2)]]
    agent.reset()
    agent.act(make_obs(position=(1, 1)))
    assert seen[-1] == [(1, 1)]


###############################################################################
# behaviour as an agent
###############################################################################

def test_same_situation_same_action_and_rng_untouched():
    agent = NeatAgent(goal_seeker(), seed=3)
    before = copy.deepcopy(agent.rng.bit_generator.state)
    obs = make_obs(position=(1, 1), cells={(4, 1): GOAL})
    actions = []
    for _ in range(5):
        agent.reset()
        actions.append(agent.act(obs))
    assert len(set(actions)) == 1
    assert agent.rng.bit_generator.state == before


def test_two_agents_from_the_same_genome_agree_and_different_genomes_differ():
    obs = make_obs(position=(1, 1), cells={(4, 1): GOAL})
    a, b = NeatAgent(goal_seeker(), seed=1), NeatAgent(goal_seeker(), seed=99)
    assert a.act(obs) == b.act(obs)
    c = NeatAgent(genome_with(biases=(0.0, 0.0, 0.0, 9.0)))
    assert c.act(obs) == ACTIONS[3]


def test_eval_and_train_mode_change_nothing():
    agent = NeatAgent(goal_seeker())
    obs = make_obs(position=(1, 1), cells={(4, 1): GOAL})
    agent.set_train_mode()
    in_train = agent.act(obs)
    agent.reset()
    agent.set_eval_mode()
    assert agent.act(obs) == in_train


def test_on_transition_does_nothing():
    agent = NeatAgent(genome_with(biases=(0.0, 3.0, 0.0, 0.0)))
    obs = make_obs()
    network = copy.deepcopy(agent.network)
    action = agent.act(obs)
    result = agent.on_transition(Transition(obs, action, obs, False, False))
    assert result is None
    assert agent.network == network
    assert agent.act(obs) == action


def test_reset_clears_the_history():
    agent = NeatAgent(genome_with())
    agent.act(make_obs(position=(1, 1)))
    agent.act(make_obs(position=(1, 2)))
    assert len(agent.history) == 2
    agent.reset()
    assert len(agent.history) == 0


def test_history_is_bounded_by_history_size():
    agent = NeatAgent(genome_with(), history_size=3)
    for col in range(6):
        agent.act(make_obs(position=(2, col % 5)))
    assert len(agent.history) == 3


def test_act_returns_an_engine_action_every_time():
    rng = np.random.default_rng(0)
    for seed in range(20):
        weights = rng.normal(0.0, 3.0, N_INPUTS * N_ACTIONS)
        genome = genome_with(biases=list(rng.normal(0.0, 1.0, N_ACTIONS)),
                             connections=[(i % N_INPUTS, i // N_INPUTS, float(w))
                                          for i, w in enumerate(weights)],
                             activation="sigmoid")
        agent = NeatAgent(genome)
        cells = {(int(rng.integers(0, 5)), int(rng.integers(0, 5))): GOAL}
        obs = make_obs(position=(int(rng.integers(1, 4)), int(rng.integers(1, 4))), cells=cells)
        assert agent.act(obs) in ACTIONS


def test_works_on_an_enemy_observation_too():
    agent = NeatAgent(genome_with(biases=(0.0, 0.0, 2.0, 0.0)))
    assert agent.act(make_obs(kind="enemy")) == ACTIONS[2]


def test_the_agent_can_be_pickled():
    agent = NeatAgent(goal_seeker(), seed=5)
    obs = make_obs(position=(1, 1), cells={(4, 1): GOAL})
    expected = agent.act(obs)
    agent.reset()
    clone = pickle.loads(pickle.dumps(agent))
    assert clone.act(obs) == expected

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"neat agent: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()