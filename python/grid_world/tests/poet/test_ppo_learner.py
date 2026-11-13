"""
Created on Fri Nov 13 11:23:02 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################


"""Tests of experiments/poet/ppo_learner.py.

Run this file with F5 in Spyder (or `runfile`): every test prints one line and a summary closes the
list. The same file also works under pytest.

Most tests replace play_games with a fake that drives the learning agent through scripted
transitions, so the REAL PpoAgent learns but no engine is needed. The last tests use the real
engine and are skipped when it cannot be imported.

PpoLearner(role, agent, games_per_call=500, train_seed=200_000, seed_span=10_000)

    - agent() is a NEW frozen copy of the learning agent on every call: eval mode, empty rollout.
      Playing with it never changes the learner, which keeps none of them.
    - train(layout, opponent, rng) plays games_per_call games with the learning agent (train mode)
      against a frozen clone of the opponent, on seeds train_seed + (games_played mod seed_span) ...
      in order, never past the span. rng is the only source of randomness. Training is in place:
      a failure halfway keeps what was learned, but games_played and history do not advance, and the
      agent holds no engine observation (the learner can be trained, cloned and pickled again).
    - clone() is independent: same networks, fresh optimizer and rollout, same games_played and history.
"""

import copy
import pickle
import random
import sys
import traceback
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from experiments.poet import ppo_learner

from agents.ppo_agent.agent import PpoAgent, build_actor_critic
from agents.features.grid import FREE as FLOOR
from core.actions import ACTIONS
from core.transition import Transition
from experiments.poet.ppo_learner import PpoLearner
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
# Stand-ins: scripted games that the real agent learns from
###############################################################################

class Opaque(SimpleNamespace):
    """Like the engine's EntityObservation: it can be read but neither copied nor pickled."""
    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy 'EntityObservation' object")

    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle 'EntityObservation' object")


def make_obs(position=(2, 2), size=5):
    grid = [[FLOOR] * size for _ in range(size)]
    return Opaque(grid=grid, position=position, kind="player", status="Alive", id=0)


class Reward:
    """reward_fn: 1.0 for the action with index 2, else 0.0."""
    def __call__(self, t):
        return 1.0 if t.action == ACTIONS[2] else 0.0

    def reset(self):
        pass


def new_agent(**kwargs):
    settings = dict(reward_fn=Reward(), rollout_steps=8, epochs=2, batch_size=4, lr=1e-2, gamma=0.9, seed=0)
    settings.update(kwargs)
    torch.manual_seed(0)
    return PpoAgent(build_actor_critic(hidden=8), **settings)


def new_learner(role="player", **kwargs):
    games = kwargs.pop("games_per_call", 6)
    return PpoLearner(role, new_agent(**kwargs), games_per_call=games)


class Opponent:
    """A frozen opponent that records how it is used."""
    def __init__(self):
        self.clones, self.eval_mode = [], False

    def clone(self):
        other = Opponent()
        self.clones.append(other)
        return other

    def set_eval_mode(self):
        self.eval_mode = True


@contextmanager
def games(fail_after=None):
    """Replace play_games: every game is one decision and one terminal transition of the agent of
    the learner's role. Yields the calls that were made."""
    calls = []

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        calls.append(dict(layout=layout, agents=dict(agents_by_role), n_games=n_games, first_seed=first_seed))
        outcomes = {role: [] for role in agents_by_role}
        for k in range(n_games):
            if fail_after is not None and sum(c["n_games"] for c in calls[:-1]) + k >= fail_after:
                raise RuntimeError("the engine failed")
            for role, agent in agents_by_role.items():
                if not isinstance(agent, PpoAgent):
                    continue
                agent.reset()
                obs = make_obs(position=(1 + (first_seed + k) % 3, 1))
                action = agent.act(obs)
                agent.on_transition(Transition(obs, action, make_obs(position=(2, 2)), True, False))
                outcomes[role].append("GoalReached" if action == ACTIONS[2] else "Timeout")
            for role, agent in agents_by_role.items():
                if not isinstance(agent, PpoAgent):
                    outcomes[role] = ["Timeout"] * n_games
        return outcomes

    with patched(ppo_learner, play_games=fake_play):
        yield calls


@contextmanager
def base_agent_without_pickle_support():
    """The BaseAgent of some versions has no __getstate__ / __setstate__: the DDQN code must not need them."""
    from agents.base_agent import BaseAgent
    saved = {n: BaseAgent.__dict__[n] for n in ("__getstate__", "__setstate__") if n in BaseAgent.__dict__}
    try:
        for name in saved:
            delattr(BaseAgent, name)
        yield
    finally:
        for name, value in saved.items():
            setattr(BaseAgent, name, value)


def weights(agent):
    return [p.detach().clone() for p in agent.net.parameters()]


def same_weights(a, b):
    return all(torch.equal(p, q) for p, q in zip(weights(a), weights(b)))

###############################################################################
# construction
###############################################################################

def test_it_is_a_learner():
    assert isinstance(new_learner(), Learner)


def test_a_bad_role_or_setting_is_refused():
    with raises(ValueError, "role"):
        PpoLearner("goblin", new_agent())
    with raises(ValueError, "games_per_call"):
        PpoLearner("player", new_agent(), games_per_call=0)
    with raises(ValueError, "seed_span"):
        PpoLearner("player", new_agent(), seed_span=0)


def test_the_learner_puts_its_agent_in_train_mode():
    agent = new_agent()
    agent.set_eval_mode()
    PpoLearner("player", agent)
    assert agent.training is True

###############################################################################
# agent(): the frozen copy
###############################################################################

def test_agent_is_a_frozen_copy_not_the_learning_agent():
    learner = new_learner()
    playing = learner.agent()
    assert isinstance(playing, PpoAgent) and playing is not learner._agent
    assert playing.training is False and playing.rollout_size == 0
    assert same_weights(playing, learner._agent)


def test_every_call_gives_a_new_frozen_agent_and_the_learner_keeps_none():
    learner = new_learner()
    first, second = learner.agent(), learner.agent()
    assert first is not second and first is not learner._agent and second is not learner._agent
    assert same_weights(first, second)
    assert not any(value is first or value is second for value in vars(learner).values())


def test_a_new_agent_after_training_has_the_new_weights():
    learner = new_learner(games_per_call=10)                  # 10 steps: one rollout of 8 is learned from
    before = learner.agent()
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    after = learner.agent()
    assert not same_weights(before, after) and same_weights(after, learner._agent)


def test_playing_with_the_agent_never_changes_the_learner():
    learner = new_learner()
    start = weights(learner._agent)
    playing = learner.agent()
    for k in range(30):
        playing.reset()
        obs = make_obs(position=(1 + k % 3, 1))
        action = playing.act(obs)
        playing.on_transition(Transition(obs, action, obs, True, False))     # eval mode: ignored
    assert playing.rollout_size == 0 and playing.updates == 0
    assert learner._agent.rollout_size == 0
    assert all(torch.equal(a, b) for a, b in zip(start, weights(learner._agent)))


def test_the_agent_after_training_has_what_was_learned_but_not_the_memory():
    learner = new_learner()
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    playing = learner.agent()
    assert same_weights(playing, learner._agent) and playing.rollout_size == 0
    assert learner._agent.rollout_size == 6 and learner._agent.updates == 0     # 6 steps: no rollout yet

###############################################################################
# train()
###############################################################################

def test_train_plays_the_games_with_the_learning_agent_against_a_frozen_opponent_clone():
    for role, other in (("player", "enemy_0"), ("enemy_0", "player")):
        learner = new_learner(role)
        opponent = Opponent()
        with games() as calls:
            result = learner.train("the layout", opponent, random.Random(0))
        assert result is None
        (call,) = calls
        assert call["layout"] == "the layout" and call["n_games"] == 6
        assert set(call["agents"]) == {role, other}
        assert call["agents"][role] is learner._agent and learner._agent.training is True
        assert len(opponent.clones) == 1 and call["agents"][other] is opponent.clones[0]
        assert call["agents"][other].eval_mode is True and opponent.eval_mode is False


def test_train_puts_the_learning_agent_back_in_train_mode_before_playing():
    learner = new_learner()
    learner._agent.set_eval_mode()                            # something else switched it
    modes = []

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        modes.append(agents_by_role["player"].training)
        return {"player": ["Timeout"] * n_games, "enemy_0": ["Timeout"] * n_games}

    with patched(ppo_learner, play_games=fake_play):
        learner.train("layout", Opponent(), random.Random(0))
    assert modes == [True]


def test_an_opponent_that_learns_does_not_learn_while_it_is_played_against():
    learner = new_learner("player")
    opponent = new_agent()                                    # a PpoAgent in train mode
    start = weights(opponent)
    with games() as calls:
        learner.train("layout", opponent, random.Random(0))
    assert calls[0]["agents"]["enemy_0"].training is False
    assert opponent.training is True and opponent.rollout_size == 0
    assert all(torch.equal(a, b) for a, b in zip(start, weights(opponent)))


def test_the_agent_learns_during_training():
    learner = new_learner(games_per_call=20)
    start = weights(learner._agent)
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    agent = learner._agent
    assert agent.updates == 2 and agent.rollout_size == 4                    # 20 steps = 2 rollouts of 8 + 4 left
    assert not all(torch.equal(a, b) for a, b in zip(start, weights(agent)))


def test_the_seeds_follow_the_games_already_played():
    learner = new_learner()
    with games() as calls:
        for _ in range(3):
            learner.train("layout", Opponent(), random.Random(0))
    assert [c["first_seed"] for c in calls] == [200_000, 200_006, 200_012]
    assert learner.games_played == 18


def test_the_seeds_start_again_at_the_first_instead_of_leaving_the_span():
    learner = PpoLearner("player", new_agent(), games_per_call=4, train_seed=1000, seed_span=10)
    with games() as calls:
        for _ in range(4):
            learner.train("layout", Opponent(), random.Random(0))
    # games 0-3, 4-7, 8-11 (8, 9 then 0, 1 again), 12-15 (2 ... 5)
    assert [(c["first_seed"], c["n_games"]) for c in calls] == [
        (1000, 4), (1004, 4), (1008, 2), (1000, 2), (1002, 4)]
    assert all(c["first_seed"] + c["n_games"] <= 1010 for c in calls)        # never past the span
    assert learner.games_played == 16


def test_a_call_longer_than_the_span_goes_around_more_than_once():
    learner = PpoLearner("player", new_agent(), games_per_call=25, train_seed=0, seed_span=10)
    with games() as calls:
        learner.train("layout", Opponent(), random.Random(0))
    assert [(c["first_seed"], c["n_games"]) for c in calls] == [(0, 10), (0, 10), (0, 5)]
    assert learner._agent.updates == 3 and learner._agent.rollout_size == 1


def test_the_history_has_a_row_per_call():
    learner = new_learner(games_per_call=10)
    with games():
        learner.train("layout", Opponent(), random.Random(0))
        learner.train("layout", Opponent(), random.Random(1))
    first, second = learner.history
    assert (first["call"], second["call"]) == (1, 2) and first["games"] == 10
    assert 0.0 <= first["goal_rate"] <= 1.0
    assert second["updates"] > first["updates"] > 0
    assert first["loss"] is not None and first["entropy"] > 0


def test_the_history_goal_rate_is_the_share_of_goals_of_the_games():
    learner = new_learner(games_per_call=10)

    def fake_play(layout, agents_by_role, n_games=100, first_seed=0):
        return {"player": ["GoalReached"] * 3 + ["Timeout"] * 7, "enemy_0": ["Alive"] * 10}

    with patched(ppo_learner, play_games=fake_play):
        learner.train("layout", Opponent(), random.Random(0))
    assert learner.history[0]["goal_rate"] == 0.3


def test_randomness_comes_from_rng_only():
    def trained(seed):
        learner = new_learner(games_per_call=20)
        with games():
            learner.train("layout", Opponent(), random.Random(seed))
        return learner._agent

    a, b, c = trained(3), trained(3), trained(4)
    assert same_weights(a, b)
    assert not same_weights(a, c)


def test_a_failure_halfway_keeps_the_count_and_leaves_a_learner_that_can_go_on():
    learner = new_learner(games_per_call=20)
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    agent = learner._agent
    count, history = learner.games_played, copy.deepcopy(learner.history)

    with games(fail_after=7):
        with raises(RuntimeError):
            learner.train("layout", Opponent(), random.Random(1))

    assert learner._agent is agent
    assert learner.games_played == count and learner.history == history       # nothing advanced
    assert len(agent.history) == 0 and agent.positions == [] and agent.actions == []   # holds no observation
    assert pickle.loads(pickle.dumps(learner)).games_played == count
    assert learner.clone().games_played == count
    with games():
        learner.train("layout", Opponent(), random.Random(2))                  # and it trains again
    assert learner.games_played == count + 20

###############################################################################
# clone()
###############################################################################

def test_clone_has_the_same_networks_and_history_but_a_fresh_memory():
    learner = new_learner(games_per_call=20)
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    twin = learner.clone()

    assert isinstance(twin, PpoLearner) and twin is not learner and twin.role == learner.role
    assert twin._agent is not learner._agent and same_weights(twin._agent, learner._agent)
    assert twin._agent.rollout_size == 0 and twin._agent.updates == 0
    assert learner._agent.rollout_size == 4 and learner._agent.updates == 2
    assert twin.games_played == 20 and twin.history == learner.history and twin.history is not learner.history
    assert twin._agent.training is True and twin.agent().training is False


def test_training_the_clone_never_changes_the_original_and_the_other_way_round():
    learner = new_learner(games_per_call=20)
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    twin = learner.clone()
    original = weights(learner._agent)

    with games():
        twin.train("layout", Opponent(), random.Random(1))
    assert all(torch.equal(a, b) for a, b in zip(original, weights(learner._agent)))
    assert learner.games_played == 20 and len(learner.history) == 1

    twin_weights = weights(twin._agent)
    with games():
        learner.train("layout", Opponent(), random.Random(2))
    assert all(torch.equal(a, b) for a, b in zip(twin_weights, weights(twin._agent)))
    assert twin.games_played == 40 and len(twin.history) == 2

###############################################################################
# Engine observations cannot be copied or pickled: the learner must hold none
###############################################################################

def test_a_learner_can_train_again_after_training():
    learner = new_learner(games_per_call=6)
    with games():
        learner.train("layout", Opponent(), random.Random(0))
        learner.train("layout", Opponent(), random.Random(1))       # copies the agent that just played
    assert learner.games_played == 12


def test_a_trained_learner_can_be_pickled_and_deep_copied():
    learner = new_learner()
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    again = pickle.loads(pickle.dumps(learner))
    assert again.games_played == 6 and same_weights(again._agent, learner._agent)
    assert copy.deepcopy(learner).games_played == 6


def test_a_learner_can_be_pickled_and_cloned_after_agents_played_for_POET():
    learner = new_learner()
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    playing = learner.agent()
    for k in range(5):                                              # what POET does: play with it
        playing.reset()
        playing.act(make_obs(position=(1 + k % 3, 1)))
    assert len(playing.history) == 1                                # it holds an observation now

    again = pickle.loads(pickle.dumps(learner))
    assert again.games_played == 6 and same_weights(again.agent(), learner.agent())
    assert learner.clone().games_played == 6


def test_all_of_it_also_works_with_a_base_agent_that_has_no_pickle_support():
    with base_agent_without_pickle_support():
        learner = new_learner(games_per_call=6)
        with games():
            learner.train("layout", Opponent(), random.Random(0))
            learner.train("layout", Opponent(), random.Random(1))
        playing = learner.agent()
        playing.reset()
        playing.act(make_obs())
        assert pickle.loads(pickle.dumps(learner)).games_played == 12
        assert copy.deepcopy(learner).games_played == 12
        assert learner.clone().games_played == 12


def test_the_learning_agent_holds_no_observation_after_training():
    learner = new_learner()
    with games():
        learner.train("layout", Opponent(), random.Random(0))
    agent = learner._agent
    assert len(agent.history) == 0 and agent.positions == [] and agent.actions == []

###############################################################################
# The real engine (skipped when it cannot be imported)
###############################################################################

def _real():
    try:
        from engine import GenerationConfig, generate_valid_layout
        from agents.heuristic_agent.agent import HeuristicAgent
        from poet.games import play_games
    except ImportError as error:
        raise unittest.SkipTest(f"real engine not available here: {error}")

    layout = generate_valid_layout(GenerationConfig(width=12, height=12, wall_density=0.1, num_enemies=1,
                                                    num_traps=0, min_player_goal_distance=6))
    return layout, HeuristicAgent, play_games


def _real_learner(role="player"):
    agent = new_agent(reward_fn=None, rollout_steps=16, batch_size=8, gamma=0.99)
    return PpoLearner(role, agent, games_per_call=3)


def test_real_a_short_training_runs_and_the_agent_plays():
    layout, Heuristic, play_games = _real()

    learner = _real_learner()
    learner.train(layout, Heuristic(), random.Random(0))

    assert learner.games_played == 3 and learner._agent.updates > 0
    outcomes = play_games(layout, {"player": learner.agent(), "enemy_0": Heuristic()}, n_games=3, first_seed=230_000)
    assert len(outcomes["player"]) == 3


def test_real_the_enemy_role_trains_against_a_player():
    layout, Heuristic, play_games = _real()

    learner = _real_learner("enemy_0")
    learner.train(layout, Heuristic(), random.Random(0))

    assert learner.games_played == 3 and learner._agent.updates > 0


def test_real_the_same_rng_gives_the_same_learner():
    layout, Heuristic, play_games = _real()

    def trained():
        learner = _real_learner()
        learner.train(layout, Heuristic(), random.Random(3))
        return learner

    first, second = trained(), trained()
    assert same_weights(first._agent, second._agent) and first.history == second.history


def test_real_a_played_learner_can_still_be_cloned_and_trained():
    layout, Heuristic, play_games = _real()

    learner = _real_learner()
    play_games(layout, {"player": learner.agent(), "enemy_0": Heuristic()}, n_games=2, first_seed=230_000)
    learner.train(layout, Heuristic(), random.Random(0))
    twin = learner.clone()
    twin.train(layout, Heuristic(), random.Random(1))
    assert twin.games_played == 6 and learner.games_played == 3

def test_real_the_same_learner_trains_twice_and_can_be_pickled_after_playing():
    # The error seen with the real engine: the agent kept the observations of its last game
    layout, Heuristic, play_games = _real()

    learner = _real_learner()
    learner.train(layout, Heuristic(), random.Random(0))
    learner.train(layout, Heuristic(), random.Random(1))                 # copies the agent that just played
    play_games(layout, {"player": learner.agent(), "enemy_0": Heuristic()}, n_games=2, first_seed=230_000)

    again = pickle.loads(pickle.dumps(learner))
    assert again.games_played == 6 and same_weights(again._agent, learner._agent)
    assert copy.deepcopy(learner).games_played == 6
    assert learner.clone().games_played == 6

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