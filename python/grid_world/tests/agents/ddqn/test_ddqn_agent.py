"""
Tests for agents/ddqn_agent/agent.py: build_mlp and DdqnAgent (the playing half).

DdqnAgent(policy_net, epsilon=0.1, eval_epsilon=0.0, seed=42, history_size=8)

    - The network must take N_INPUTS numbers and give N_ACTIONS Q-values, else ValueError.
      epsilon and eval_epsilon must be in [0, 1], else ValueError.
    - act(obs) returns an engine Action: the one with the largest Q-value (ties: the first),
      decided on build_input(history) as a float32 batch of one.
    - With probability epsilon (training) or eval_epsilon (eval mode) the action is drawn from
      the agent's rng instead. An epsilon of 0 never touches the rng.
    - Same seed, same decisions. Deciding builds no gradient.
    - clone() is independent: own network (equal weights), same mode, same rng state, empty history.
    - The agent can be pickled.

    Learning (training mode; in eval mode on_transition does nothing):
    - on_transition stores (state, action, reward, next state, terminated) in a replay buffer. The reward
      is reward_fn(transition). The next state is built from the history moved one step forward.
    - Nothing is learned before min_buffer_size transitions are stored; after that every transition
      does one update on a batch drawn with the agent's rng.
    - td_targets is the Double DQN target: the policy network chooses the next action, the target
      network values it, a terminated game has no future, a truncated one still bootstraps.
    - Epsilon decays by epsilon_decay per update, never below epsilon_min. The target network is
      copied from the policy network every target_update_freq updates.
    - clone(): copies of both networks, current epsilon, mode and rng state; fresh optimizer,
      empty buffer, zero updates.

Fake networks and fake observations are used, so no engine is needed.
Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import agents.ddqn_agent.agent as agent_module
from agents.ddqn_agent.agent import DdqnAgent, build_mlp, td_targets
from agents.ddqn_agent.replay_buffer import ReplayBuffer
from agents.features.grid import FREE as FLOOR
from agents.neat_agent.agent import N_INPUTS
from core.actions import ACTIONS, N_ACTIONS
from core.transition import Transition
from engine import Action

###############################################################################
# helpers
###############################################################################

class FixedQ(nn.Module):
    """Ignores the input and always gives the same Q-values."""
    def __init__(self, q):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q, dtype=torch.float32))

    def forward(self, x):
        return self.q.expand(x.shape[0], -1)


def prefers(index, n_actions=N_ACTIONS):
    q = [0.0] * n_actions
    q[index] = 1.0
    return FixedQ(q)


def make_obs(position=(2, 2), size=5, kind="player"):
    grid = [[FLOOR] * size for _ in range(size)]
    return SimpleNamespace(grid=grid, position=position, kind=kind, status="Alive", id=0)


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def patched(name, value):
    """Context manager: replace agent_module.<name> for a while."""
    class Patch:
        def __enter__(self):
            self.old = getattr(agent_module, name)
            setattr(agent_module, name, value)
        def __exit__(self, *exc):
            setattr(agent_module, name, self.old)
    return Patch()


def actions_taken(agent, n):
    return [agent.act(make_obs()) for _ in range(n)]

###############################################################################
# build_mlp
###############################################################################

def test_build_mlp_maps_a_batch_of_inputs_to_one_q_value_per_action():
    out = build_mlp()(torch.zeros(3, N_INPUTS))
    assert tuple(out.shape) == (3, N_ACTIONS) and out.dtype == torch.float32


def test_build_mlp_takes_other_sizes():
    out = build_mlp(n_inputs=7, n_actions=2, hidden=5)(torch.zeros(4, 7))
    assert tuple(out.shape) == (4, 2)


def test_build_mlp_repeats_with_the_same_torch_seed():
    torch.manual_seed(3)
    a = build_mlp()
    torch.manual_seed(3)
    b = build_mlp()
    x = torch.ones(1, N_INPUTS)
    assert torch.equal(a(x), b(x))

###############################################################################
# construction
###############################################################################

def test_builds_from_a_network_with_the_right_shape():
    agent = DdqnAgent(build_mlp())
    assert agent.name == "ddqn" and agent.training is True


def test_wrong_number_of_outputs_is_refused():
    for n in (1, N_ACTIONS - 1, N_ACTIONS + 1):
        assert raises(ValueError, DdqnAgent, build_mlp(n_actions=n))


def test_wrong_number_of_inputs_is_refused():
    for n in (1, N_INPUTS - 1, N_INPUTS + 1):
        assert raises(ValueError, DdqnAgent, build_mlp(n_inputs=n))


def test_an_epsilon_outside_0_1_is_refused():
    for bad in (-0.1, 1.1):
        assert raises(ValueError, DdqnAgent, prefers(0), epsilon=bad)
        assert raises(ValueError, DdqnAgent, prefers(0), eval_epsilon=bad)
    DdqnAgent(prefers(0), epsilon=0.0, eval_epsilon=1.0)       # the limits are allowed


def test_seed_and_history_size_are_passed_to_the_base_agent():
    agent = DdqnAgent(prefers(0), seed=7, history_size=3)
    assert agent.history.maxlen == 3
    assert list(agent.rng.random(3)) == list(np.random.default_rng(7).random(3))

###############################################################################
# the greedy decision
###############################################################################

def test_the_largest_q_value_picks_the_action():
    for i in range(N_ACTIONS):
        agent = DdqnAgent(prefers(i), epsilon=0.0)
        assert agent.act(make_obs()) == ACTIONS[i]


def test_ties_go_to_the_first_action():
    agent = DdqnAgent(FixedQ([0.5] * N_ACTIONS), epsilon=0.0)
    assert agent.act(make_obs()) == ACTIONS[0]
    agent = DdqnAgent(FixedQ([0.0, 2.0, 2.0, 1.0]), epsilon=0.0)
    assert agent.act(make_obs()) == ACTIONS[1]


def test_act_returns_an_engine_action():
    assert isinstance(DdqnAgent(build_mlp()).act(make_obs()), Action)


def test_the_network_gets_the_build_input_of_the_history_as_a_float32_batch_of_one():
    seen = []

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(N_ACTIONS))
        def forward(self, x):
            seen.append(x.clone())
            return self.w.expand(x.shape[0], -1)

    lengths = []
    def fake_build_input(history):
        lengths.append(len(history))
        return np.arange(N_INPUTS, dtype=np.float64)           # float64 on purpose

    agent = DdqnAgent(Spy(), epsilon=0.0)
    seen.clear()                                               # the check at construction
    with patched("build_input", fake_build_input):
        agent.act(make_obs())
        agent.act(make_obs(position=(1, 1)))

    assert lengths == [1, 2]                                   # the history grows with the episode
    assert all(x.dtype == torch.float32 and tuple(x.shape) == (1, N_INPUTS) for x in seen)
    assert torch.equal(seen[0][0], torch.arange(N_INPUTS, dtype=torch.float32))


def test_deciding_builds_no_gradient():
    net = build_mlp()
    agent = DdqnAgent(net, epsilon=0.0)
    agent.act(make_obs())
    assert all(p.grad is None for p in net.parameters())
    assert agent.q_values().shape == (N_ACTIONS,)

###############################################################################
# exploration
###############################################################################

def test_epsilon_1_explores_every_action_while_training():
    agent = DdqnAgent(prefers(0), epsilon=1.0)
    assert set(actions_taken(agent, 200)) == set(ACTIONS)


def test_epsilon_0_always_takes_the_best_action():
    agent = DdqnAgent(prefers(2), epsilon=0.0)
    assert set(actions_taken(agent, 100)) == {ACTIONS[2]}


def test_the_share_of_random_actions_follows_epsilon():
    # random draws pick the preferred action a quarter of the time: P(other) = eps * 3/4
    agent = DdqnAgent(prefers(0), epsilon=0.4, seed=1)
    others = sum(a != ACTIONS[0] for a in actions_taken(agent, 4000)) / 4000
    assert 0.26 < others < 0.34, others


def test_eval_mode_uses_eval_epsilon_not_epsilon():
    agent = DdqnAgent(prefers(1), epsilon=1.0, eval_epsilon=0.0)
    agent.set_eval_mode()
    assert set(actions_taken(agent, 100)) == {ACTIONS[1]}
    agent.set_train_mode()
    assert len(set(actions_taken(agent, 200))) == N_ACTIONS


def test_eval_epsilon_can_explore_too():
    agent = DdqnAgent(prefers(1), epsilon=0.0, eval_epsilon=1.0)
    agent.set_eval_mode()
    assert len(set(actions_taken(agent, 200))) == N_ACTIONS


def test_a_greedy_agent_never_touches_the_rng():
    agent = DdqnAgent(build_mlp(), epsilon=0.0, seed=5)
    actions_taken(agent, 20)
    assert list(agent.rng.random(3)) == list(np.random.default_rng(5).random(3))


def test_same_seed_same_decisions_and_another_seed_other_ones():
    net = prefers(0)
    a = actions_taken(DdqnAgent(net, epsilon=0.5, seed=1), 60)
    b = actions_taken(DdqnAgent(copy.deepcopy(net), epsilon=0.5, seed=1), 60)
    c = actions_taken(DdqnAgent(copy.deepcopy(net), epsilon=0.5, seed=2), 60)
    assert a == b and a != c

###############################################################################
# clone, pickle, on_transition
###############################################################################

def test_clone_has_its_own_network_with_equal_weights():
    agent = DdqnAgent(build_mlp(), epsilon=0.3, eval_epsilon=0.2)
    other = agent.clone()
    assert other.policy_net is not agent.policy_net
    assert all(torch.equal(p, q) for p, q in zip(agent.policy_net.parameters(), other.policy_net.parameters()))
    assert (other.epsilon, other.eval_epsilon) == (0.3, 0.2)

    with torch.no_grad():
        for p in other.policy_net.parameters():
            p.add_(1.0)
    assert not any(torch.equal(p, q) for p, q in zip(agent.policy_net.parameters(), other.policy_net.parameters()))


def test_clone_keeps_the_mode_and_the_rng_state_and_starts_a_clean_episode():
    agent = DdqnAgent(prefers(0), epsilon=0.5, seed=9, history_size=4)
    agent.set_eval_mode()
    agent.rng.random(5)
    for position in ((1, 1), (1, 2), (2, 2)):
        agent.act(make_obs(position=position))
    other = agent.clone()

    assert other.training is False
    assert len(other.history) == 0 and other.history.maxlen == 4 and other.positions == []
    assert len(agent.history) == 3                              # the original is untouched
    assert list(other.rng.random(3)) == list(agent.rng.random(3))


def test_clone_decides_like_the_original():
    agent = DdqnAgent(build_mlp(), epsilon=0.5, seed=4)
    other = agent.clone()
    assert actions_taken(agent, 40) == actions_taken(other, 40)


def test_the_agent_survives_pickle():
    agent = DdqnAgent(build_mlp(), epsilon=0.5, seed=4)
    other = pickle.loads(pickle.dumps(agent))
    assert actions_taken(agent, 40) == actions_taken(other, 40)


def test_on_transition_does_nothing_in_eval_mode():
    net = build_mlp()
    agent = DdqnAgent(net, min_buffer_size=1, batch_size=1)
    agent.set_eval_mode()
    before = copy.deepcopy(net.state_dict())
    obs = make_obs()
    agent.act(obs)
    assert agent.on_transition(Transition(obs, ACTIONS[0], obs, False, False)) is None
    assert len(agent.replay_buffer) == 0 and agent.learn_steps == 0
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())

###############################################################################
# replay buffer
###############################################################################

def fill(buffer, n, start=0):
    for i in range(start, start + n):
        buffer.push([i, i], i % 4, float(i), [i + 1, i + 1], i % 2 == 0)


def test_buffer_stores_and_counts_and_starts_empty():
    buffer = ReplayBuffer(5)
    assert len(buffer) == 0 and buffer._data is None           # nothing allocated yet
    fill(buffer, 3)
    assert len(buffer) == 3


def test_buffer_overwrites_the_oldest_when_full():
    buffer = ReplayBuffer(4)
    fill(buffer, 6)
    assert len(buffer) == 4
    states, actions, rewards, next_states, terminated = buffer.sample(4, np.random.default_rng(0))
    assert sorted(rewards.tolist()) == [2.0, 3.0, 4.0, 5.0]    # 0 and 1 are gone


def test_buffer_rows_stay_together():
    buffer = ReplayBuffer(10)
    fill(buffer, 10)
    states, actions, rewards, next_states, terminated = buffer.sample(10, np.random.default_rng(1))
    for s, a, r, n, t in zip(states, actions, rewards, next_states, terminated):
        i = int(r)
        assert s.tolist() == [i, i] and n.tolist() == [i + 1, i + 1]
        assert a == i % 4 and t == float(i % 2 == 0)


def test_buffer_sample_has_no_repeats_and_follows_the_rng():
    buffer = ReplayBuffer(20)
    fill(buffer, 20)
    a = buffer.sample(8, np.random.default_rng(3))[2]
    b = buffer.sample(8, np.random.default_rng(3))[2]
    c = buffer.sample(8, np.random.default_rng(4))[2]
    assert len(set(a.tolist())) == 8
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist()


def test_buffer_refuses_bad_use():
    assert raises(ValueError, ReplayBuffer, 0)
    buffer = ReplayBuffer(5)
    fill(buffer, 2)
    assert raises(ValueError, buffer.sample, 3, np.random.default_rng(0))      # more than stored
    assert raises(ValueError, buffer.sample, 0, np.random.default_rng(0))
    assert raises(ValueError, buffer.push, [1, 2, 3], 0, 0.0, [1, 2, 3], False)  # another state size

###############################################################################
# the Double DQN target
###############################################################################

def test_td_target_is_double_policy_chooses_target_values():
    policy = FixedQ([5.0, 0.0, 0.0, 0.0])          # prefers action 0
    target = FixedQ([1.0, 9.0, 0.0, 0.0])          # its own maximum is action 1 (9.0)
    rewards = torch.tensor([2.0])
    next_states = torch.zeros(1, N_INPUTS)
    out = td_targets(policy, target, rewards, next_states, torch.tensor([0.0]), gamma=0.5)
    assert torch.allclose(out, torch.tensor([2.0 + 0.5 * 1.0]))               # not 2 + 0.5 * 9


def test_td_target_of_a_terminated_game_is_the_reward_alone():
    net = FixedQ([3.0, 3.0, 3.0, 3.0])
    out = td_targets(net, net, torch.tensor([2.0, 2.0]), torch.zeros(2, N_INPUTS),
                     torch.tensor([1.0, 0.0]), gamma=0.5)
    assert torch.allclose(out, torch.tensor([2.0, 3.5]))


def test_td_target_has_no_gradient():
    policy, target = build_mlp(), build_mlp()
    out = td_targets(policy, target, torch.zeros(2), torch.zeros(2, N_INPUTS), torch.zeros(2), 0.9)
    assert not out.requires_grad

###############################################################################
# learning
###############################################################################

class TableReward:
    """reward_fn: a fixed reward per action index. Counts resets."""
    def __init__(self, table):
        self.table, self.resets = table, 0
    def __call__(self, t):
        return self.table[ACTIONS.index(t.action)]
    def reset(self):
        self.resets += 1


def play_step(agent, action_index, terminated=True, truncated=False, position=(2, 2)):
    """One decision plus its transition, from a fresh episode."""
    agent.reset()
    obs = make_obs(position=position)
    agent.act(obs)
    next_obs = make_obs(position=(position[0], position[1] + 1))
    agent.on_transition(Transition(obs, ACTIONS[action_index], next_obs, terminated, truncated))


def small_agent(**kwargs):
    settings = dict(reward_fn=TableReward([0.0] * N_ACTIONS), epsilon=0.0, batch_size=4, min_buffer_size=4,
                    buffer_capacity=100, target_update_freq=5, lr=1e-2, gamma=0.9)
    settings.update(kwargs)
    torch.manual_seed(0)
    return DdqnAgent(build_mlp(hidden=16), **settings)


def test_transitions_are_stored_with_the_reward_and_the_end_flag():
    agent = small_agent(reward_fn=TableReward([0.0, 0.0, 7.0, 0.0]), min_buffer_size=50)
    play_step(agent, 2, terminated=True)
    play_step(agent, 2, terminated=False, truncated=True)        # cut by the clock: NOT terminated
    states, actions, rewards, next_states, terminated = agent.replay_buffer.sample(2, np.random.default_rng(0))
    assert sorted(actions.tolist()) == [2, 2] and rewards.tolist() == [7.0, 7.0]
    assert sorted(terminated.tolist()) == [0.0, 1.0]


def test_the_next_state_is_built_from_the_history_moved_one_step_forward():
    seen = []
    def spy(history):
        seen.append([o.position for o in history])
        return np.zeros(N_INPUTS, dtype=np.float32)

    agent = small_agent(min_buffer_size=50, history_size=2)
    with patched("build_input", spy):
        agent.reset()
        agent.act(make_obs(position=(1, 1)))
        agent.act(make_obs(position=(1, 2)))
        agent.on_transition(Transition(make_obs(position=(1, 2)), ACTIONS[0], make_obs(position=(1, 3)), False, False))
    # two decisions, then: the state acted in, and the same window with the new observation (size 2)
    assert seen[-2] == [(1, 1), (1, 2)] and seen[-1] == [(1, 2), (1, 3)]


def test_nothing_is_learned_before_the_buffer_is_ready_then_every_transition_updates():
    agent = small_agent(min_buffer_size=6)
    start = copy.deepcopy(agent.policy_net.state_dict())
    for _ in range(5):
        play_step(agent, 1)
    assert agent.learn_steps == 0 and agent.last_loss is None
    assert all(torch.equal(start[k], v) for k, v in agent.policy_net.state_dict().items())
    play_step(agent, 1)
    assert agent.learn_steps == 1 and agent.last_loss is not None
    play_step(agent, 1)
    assert agent.learn_steps == 2
    assert not all(torch.equal(start[k], v) for k, v in agent.policy_net.state_dict().items())


def test_it_learns_which_action_pays():
    agent = small_agent(reward_fn=TableReward([-1.0, 0.0, 1.0, 0.0]))
    for i in range(300):
        play_step(agent, 2 if i % 2 == 0 else 0)
    agent.reset()
    agent.act(make_obs())
    q = agent.q_values()
    assert abs(q[2] - 1.0) < 0.25 and abs(q[0] + 1.0) < 0.25, q
    assert int(np.argmax(q)) == 2


def test_the_target_network_is_copied_every_target_update_freq_updates():
    agent = small_agent(target_update_freq=5)
    for _ in range(4):
        play_step(agent, 1)                                       # 4 transitions: ready, 1 update
    assert agent.learn_steps == 1
    differs = lambda: any(not torch.equal(p, q) for p, q in zip(agent.policy_net.parameters(),
                                                                 agent.target_net.parameters()))
    assert differs()
    for _ in range(3):
        play_step(agent, 1)
    assert agent.learn_steps == 4 and differs()                    # not yet
    play_step(agent, 1)
    assert agent.learn_steps == 5 and not differs()                # copied at the 5th update


def test_every_update_clips_the_gradient_norm_to_1():
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = lambda params, max_norm, **kw: calls.append(max_norm) or original(params, max_norm, **kw)
    try:
        agent = small_agent()
        for _ in range(6):
            play_step(agent, 1)
    finally:
        torch.nn.utils.clip_grad_norm_ = original
    assert agent.learn_steps == 3 and calls == [1.0, 1.0, 1.0]


def test_the_target_network_does_not_train():
    agent = small_agent()
    assert not any(p.requires_grad for p in agent.target_net.parameters())
    assert agent.target_net is not agent.policy_net


def test_epsilon_decays_per_update_down_to_the_minimum():
    agent = small_agent(epsilon=0.5, epsilon_min=0.2, epsilon_decay=0.5)
    for _ in range(3):
        play_step(agent, 1)
    assert agent.epsilon == 0.5                                    # still filling the buffer
    play_step(agent, 1)                                            # first update
    assert agent.epsilon == 0.25
    play_step(agent, 1)
    assert agent.epsilon == 0.2                                    # not 0.125
    play_step(agent, 1)
    assert agent.epsilon == 0.2


def test_epsilon_below_the_minimum_is_left_alone():
    agent = small_agent(epsilon=0.01, epsilon_min=0.2, epsilon_decay=0.5)
    for _ in range(8):
        play_step(agent, 1)
    assert agent.epsilon == 0.01


def test_reset_resets_the_reward():
    reward = TableReward([0.0] * N_ACTIONS)
    agent = small_agent(reward_fn=reward)
    agent.reset()
    agent.reset()
    assert reward.resets == 2


def test_same_seed_same_transitions_same_weights():
    torch.manual_seed(1)
    net = build_mlp(hidden=8)
    agents = [DdqnAgent(copy.deepcopy(net), reward_fn=TableReward([0.0, 1.0, 0.0, 0.0]), epsilon=0.0,
                        batch_size=4, min_buffer_size=4, seed=3) for _ in range(2)]
    for agent in agents:
        for i in range(30):
            play_step(agent, i % 4, position=(1 + i % 3, 1))
    assert all(torch.equal(p, q) for p, q in zip(agents[0].policy_net.parameters(), agents[1].policy_net.parameters()))


def test_construction_refuses_bad_settings():
    net = build_mlp()
    for bad in (dict(gamma=1.5), dict(epsilon_min=-0.1), dict(epsilon_decay=0.0), dict(epsilon_decay=1.1),
                dict(lr=0.0), dict(batch_size=0), dict(batch_size=10, min_buffer_size=5),
                dict(min_buffer_size=100, buffer_capacity=50), dict(target_update_freq=0)):
        assert raises(ValueError, DdqnAgent, net, **bad), bad


def test_a_reward_with_another_gamma_or_window_is_refused():
    assert raises(ValueError, DdqnAgent, build_mlp(), reward_fn=SimpleNamespace(gamma=0.5), gamma=0.99)
    assert raises(ValueError, DdqnAgent, build_mlp(), reward_fn=SimpleNamespace(window=3), history_size=8)

###############################################################################
# clone after learning
###############################################################################

def test_clone_carries_both_networks_and_epsilon_but_not_the_memory():
    agent = small_agent(epsilon=0.6, epsilon_min=0.1, epsilon_decay=0.9)
    for i in range(20):
        play_step(agent, i % 4)
    assert len(agent.replay_buffer) == 20 and agent.learn_steps > 0

    other = agent.clone()
    same = lambda a, b: all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert same(agent.policy_net, other.policy_net) and same(agent.target_net, other.target_net)
    assert other.policy_net is not agent.policy_net and other.target_net is not agent.target_net
    assert other.epsilon == agent.epsilon
    assert len(other.replay_buffer) == 0 and other.learn_steps == 0 and other.last_loss is None
    assert len(agent.replay_buffer) == 20                          # the original keeps its memory
    assert other.optimizer is not agent.optimizer and len(other.optimizer.state) == 0


def test_training_the_clone_leaves_the_original_alone():
    agent = small_agent()
    for i in range(8):
        play_step(agent, i % 4)
    other = agent.clone()
    before = copy.deepcopy(agent.policy_net.state_dict())
    for i in range(12):
        play_step(other, 1)
    assert other.learn_steps > 0
    assert all(torch.equal(before[k], v) for k, v in agent.policy_net.state_dict().items())


def test_clone_keeps_the_reward_settings_and_does_not_share_the_reward_object():
    reward = TableReward([0.0, 1.0, 0.0, 0.0])
    agent = small_agent(reward_fn=reward)
    other = agent.clone()
    assert other.reward_fn is not reward and other.reward_fn.table == reward.table


def test_pickle_keeps_the_learning_state():
    agent = small_agent()
    for i in range(12):
        play_step(agent, i % 4)
    other = pickle.loads(pickle.dumps(agent))
    assert len(other.replay_buffer) == 12 and other.learn_steps == agent.learn_steps
    play_step(agent, 1)
    play_step(other, 1)
    assert all(torch.equal(p, q) for p, q in zip(agent.policy_net.parameters(), other.policy_net.parameters()))

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"ddqn agent: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()