"""
Contract tests: every agent can be used in place of every other one.

The same checks run on RandomAgent, HeuristicAgent, LinearQAgent, NeatAgent and ScriptedAgent:

    - it is a BaseAgent with a non-empty name, and the names are all different;
    - act(obs) returns an engine Action and leaves the observation unchanged;
    - the history holds the observations seen so far, bounded by history_size, and reset() empties it;
    - set_train_mode / set_eval_mode switch `training`;
    - on_transition accepts a Transition in both modes and returns None;
    - two fresh agents built with the same seed play the same episode the same way;
    - all of them share the same history_size and seed arguments;
    - the agent records its episode: positions (cell at each decision) and actions (what it chose),
      aligned, never truncated by history_size, emptied by reset(), not recorded for a refused action.
      An action is recorded as its index in ACTIONS (an int), so the record can be pickled and saved;
    - clone() is an independent agent that plays the same episode, with a clean history;
    - save / load: agents that store something (SAVES) write a file and read it back; every other
      agent refuses with NotImplementedError, so nothing fails silently.

The checks use fake observations (simple objects with grid, position, kind), so no engine is needed.
Add a new agent by adding one factory to AGENTS.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy
import os
import pickle
import tempfile
from types import SimpleNamespace

import agents.base_agent as base_agent_module
from agents.base_agent import BaseAgent
from agents.features.grid import FREE, GOAL, WALL
from agents.heuristic_agent.agent import HeuristicAgent
from agents.linear_q_agent.agent import LinearQAgent
from agents.neat_agent.agent import NeatAgent, N_INPUTS
from agents.random_agent.agent import RandomAgent
from agents.scripted_agent.agent import ScriptedAgent
from core.actions import ACTIONS, N_ACTIONS
from core.transition import Transition
from engine import Action

###############################################################################
# the agents under test
###############################################################################

def neat_genome():
    nodes = {i: {"type": "input"} for i in range(N_INPUTS)}
    for k in range(N_ACTIONS):
        nodes[N_INPUTS + k] = {"type": "output", "bias": float(k % 3), "activation": "sigmoid"}
    return {"n_inputs": N_INPUTS, "n_outputs": N_ACTIONS, "nodes": nodes, "connections": {}}


# name -> function(seed, history_size) -> agent
AGENTS = {
    "random":    lambda seed=42, history_size=8: RandomAgent(seed=seed, history_size=history_size),
    "heuristic": lambda seed=42, history_size=8: HeuristicAgent(seed=seed, history_size=history_size),
    "linear_q":  lambda seed=42, history_size=8: LinearQAgent(seed=seed, history_size=history_size),
    "neat":      lambda seed=42, history_size=8: NeatAgent(neat_genome(), seed=seed, history_size=history_size),
    "scripted":  lambda seed=42, history_size=8: ScriptedAgent([ACTIONS[i % N_ACTIONS] for i in range(50)],
                                                              seed=seed, history_size=history_size),
}

# agents that implement save / load. When another agent learns to save, add it here.
SAVES = {"neat"}

###############################################################################
# helpers
###############################################################################

def make_obs(position=(2, 2), size=6, cells=None, kind="player"):
    grid = [[FREE] * size for _ in range(size)]
    for (row, col), code in (cells or {}).items():
        grid[row][col] = code
    return SimpleNamespace(grid=grid, position=position, kind=kind, status="Alive", id=0)


EPISODE = [make_obs(position=(2, 2), cells={(5, 5): GOAL}),
           make_obs(position=(2, 3), cells={(5, 5): GOAL, (1, 3): WALL}),
           make_obs(position=(3, 3), cells={(5, 5): GOAL}),
           make_obs(position=(3, 4), cells={(5, 5): GOAL, (3, 5): WALL}),
           make_obs(position=(4, 4), cells={(5, 5): GOAL})]


def play(agent):
    agent.reset()
    return [agent.act(obs) for obs in EPISODE]

###############################################################################
# tests
###############################################################################

def test_every_agent_is_a_base_agent_with_its_own_name():
    names = []
    for label, build in AGENTS.items():
        agent = build()
        assert isinstance(agent, BaseAgent), label
        assert isinstance(agent.name, str) and agent.name, label
        names.append(agent.name)
    assert len(set(names)) == len(names)


def test_act_returns_an_engine_action_and_leaves_the_observation_alone():
    for label, build in AGENTS.items():
        agent = build()
        agent.reset()
        for obs in EPISODE:
            before = copy.deepcopy(obs)
            action = agent.act(obs)
            assert isinstance(action, Action) and action in ACTIONS, label
            assert obs.grid == before.grid and obs.position == before.position, label


def test_history_follows_the_episode_and_reset_empties_it():
    for label, build in AGENTS.items():
        agent = build(history_size=3)
        agent.reset()
        for k, obs in enumerate(EPISODE, start=1):
            agent.act(obs)
            assert len(agent.history) == min(k, 3), label
            assert agent.history[-1] is obs, label
        agent.reset()
        assert len(agent.history) == 0, label


def test_train_and_eval_modes_switch_the_flag():
    for label, build in AGENTS.items():
        agent = build()
        agent.set_eval_mode()
        assert agent.training is False, label
        agent.set_train_mode()
        assert agent.training is True, label


def test_on_transition_accepts_a_transition_in_both_modes():
    for label, build in AGENTS.items():
        for mode in ("train", "eval"):
            agent = build()
            agent.set_train_mode() if mode == "train" else agent.set_eval_mode()
            agent.reset()
            obs, next_obs = EPISODE[0], EPISODE[1]
            action = agent.act(obs)
            result = agent.on_transition(Transition(obs, action, next_obs, False, False))
            assert result is None, (label, mode)


def test_the_same_seed_gives_the_same_episode():
    for label, build in AGENTS.items():
        assert play(build(seed=5)) == play(build(seed=5)), label


def test_agents_can_replace_each_other_in_the_same_loop():
    # the loop below knows nothing about which agent it runs
    for label, build in AGENTS.items():
        agent = build()
        agent.reset()
        for obs, next_obs in zip(EPISODE, EPISODE[1:]):
            action = agent.act(obs)
            agent.on_transition(Transition(obs, action, next_obs, False, False))

def test_the_agent_records_positions_and_actions_of_the_episode():
    for label, build in AGENTS.items():
        agent = build(history_size=2)                 # smaller than the episode: the record must not be cut
        agent.reset()
        assert agent.positions == [] and agent.actions == [], label
        chosen = [agent.act(obs) for obs in EPISODE]
        assert agent.positions == [obs.position for obs in EPISODE], label
        assert all(isinstance(p, tuple) for p in agent.positions), label
        assert agent.actions == [ACTIONS.index(a) for a in chosen], label
        assert all(type(a) is int and 0 <= a < N_ACTIONS for a in agent.actions), label
        assert [ACTIONS[i] for i in agent.actions] == chosen, label
        assert len(agent.positions) == len(agent.actions) == len(EPISODE) > len(agent.history), label


def test_the_record_can_be_pickled():
    for label, build in AGENTS.items():
        agent = build()
        agent.reset()
        for obs in EPISODE:
            agent.act(obs)
        assert pickle.loads(pickle.dumps((agent.positions, agent.actions))) == (agent.positions, agent.actions), label


def test_an_action_outside_actions_is_refused_and_not_recorded():
    class Wander(BaseAgent):
        name = "wander"
        def select_action(self, obs):
            return Action.Right
    saved = base_agent_module.ACTIONS
    base_agent_module.ACTIONS = saved[:3]               # Right is no longer a known action
    try:
        agent = Wander()
        agent.reset()
        try:
            agent.act(EPISODE[0])
        except ValueError:
            pass
        else:
            raise AssertionError("an action outside ACTIONS was accepted")
        assert agent.positions == [] and agent.actions == []
    finally:
        base_agent_module.ACTIONS = saved


def test_reset_starts_a_new_record_and_the_old_one_stays_until_then():
    for label, build in AGENTS.items():
        agent = build()
        agent.reset()
        for obs in EPISODE:
            agent.act(obs)
        first = list(agent.positions)
        assert first == [obs.position for obs in EPISODE], label      # still there after the episode
        agent.reset()
        assert agent.positions == [] and agent.actions == [], label
        agent.act(EPISODE[2])
        assert agent.positions == [EPISODE[2].position], label


def test_a_refused_action_is_not_recorded():
    class Bad(BaseAgent):
        name = "bad"
        def select_action(self, obs):
            return "up"                                 # not an engine Action
    agent = Bad()
    agent.reset()
    try:
        agent.act(EPISODE[0])
    except TypeError:
        pass
    else:
        raise AssertionError("a wrong action type was accepted")
    assert agent.positions == [] and agent.actions == []


def test_clone_plays_the_same_episode_and_is_independent():
    for label, build in AGENTS.items():
        agent = build(seed=3)
        agent.reset()
        agent.act(EPISODE[0])
        other = agent.clone()
        assert type(other) is type(agent) and other is not agent, label
        assert len(other.history) == 0 and len(agent.history) == 1, label
        assert other.positions == [] and other.actions == [], label
        assert len(agent.positions) == 1 and other.positions is not agent.positions, label
        assert other.history is not agent.history and other.rng is not agent.rng, label
        assert play(other) == play(agent), label
        agent.act(EPISODE[1])
        assert len(other.history) == len(EPISODE), label     # the original moving on did not touch it


def test_clone_keeps_the_mode():
    for label, build in AGENTS.items():
        agent = build()
        agent.set_eval_mode()
        assert agent.clone().training is False, label
        agent.set_train_mode()
        assert agent.clone().training is True, label


def test_save_and_load_work_where_supported_and_refuse_loudly_elsewhere():
    folder = tempfile.mkdtemp()
    for label, build in AGENTS.items():
        agent, path = build(), os.path.join(folder, f"{label}.bin")
        if label in SAVES:
            agent.save(path)
            assert os.path.getsize(path) > 0, label
            agent.load(path)
            assert play(agent) == play(build()), label
        else:
            for call in (agent.save, agent.load):
                try:
                    call(path)
                except NotImplementedError:
                    continue
                raise AssertionError(f"{label}: {call.__name__} did not refuse")
            assert not os.path.exists(path), label


###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"agent contract: all {len(tests)} checks passed ({len(AGENTS)} agents)")


if __name__ == "__main__":
    run()