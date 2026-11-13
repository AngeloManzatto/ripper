"""
Created on Fri Nov 13 11:17:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
Tests for agents/ppo_agent/agent.py: ActorCritic, compute_gae and PpoAgent.

PpoAgent(net, reward_fn=None, eval_epsilon=0.0, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
         entropy_coef=0.01, value_coef=0.5, lr=3e-4, rollout_steps=512, epochs=4, batch_size=64,
         seed=42, history_size=8)

    - The network must take N_INPUTS numbers and give (N_ACTIONS scores, one value), else ValueError.
      Probabilities and coefficients are checked, else ValueError.
    - Training: the action is SAMPLED from softmax(scores) with the agent's rng (one number per decision).
      Eval mode: the largest score (ties: the first); with probability eval_epsilon a random action.
      A greedy eval agent never touches the rng. Deciding builds no gradient.
    - compute_gae: delta = r + gamma * next_value * (1 - terminated) - value;
      advantage = delta + gamma * lambda * (1 - ended) * next advantage; return = advantage + value.

    Learning (training mode; in eval mode on_transition does nothing):
    - on_transition stores one step: state, action, its log-probability, the critic value, the reward
      (reward_fn), the next state (history moved one step forward), terminated, episode ended.
    - Nothing is learned before rollout_steps steps are stored; then ONE update runs and the rollout is
      emptied. The last step of a rollout always closes the advantage chain.
    - A terminated game has no future; a truncated one still bootstraps from the value of the next state.
    - Every minibatch step clips the gradient norm to 1.
    - clone(): copy of the network, current mode and rng state; fresh optimizer, empty rollout, zero updates.

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

import agents.ppo_agent.agent as agent_module
from agents.ppo_agent.agent import ActorCritic, PpoAgent, build_actor_critic, compute_gae
from agents.features.grid import FREE as FLOOR
from agents.neat_agent.agent import N_INPUTS
from core.actions import ACTIONS, N_ACTIONS
from core.transition import Transition
from engine import Action

###############################################################################
# helpers
###############################################################################

class FixedNet(nn.Module):
    """Ignores the input: always the same scores and the same value."""
    def __init__(self, logits, value=0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1), self.value.expand(x.shape[0])


def prefers(index, strength=5.0):
    logits = [0.0] * N_ACTIONS
    logits[index] = strength
    return FixedNet(logits)


class Opaque(SimpleNamespace):
    """Like the engine's EntityObservation: it can be read but neither copied nor pickled."""
    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy 'EntityObservation' object")

    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle 'EntityObservation' object")


def make_obs(position=(2, 2), size=5, kind="player"):
    grid = [[FLOOR] * size for _ in range(size)]
    return Opaque(grid=grid, position=position, kind=kind, status="Alive", id=0)


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


def same_weights(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


class TableReward:
    """reward_fn: a fixed reward per action index. Counts resets."""
    def __init__(self, table):
        self.table, self.resets = table, 0
    def __call__(self, t):
        return self.table[ACTIONS.index(t.action)]
    def reset(self):
        self.resets += 1


def play_step(agent, action_index, terminated=True, truncated=False, position=(2, 2)):
    """One decision plus its transition, from a fresh episode (the action is given, not sampled)."""
    agent.reset()
    obs = make_obs(position=position)
    agent.act(obs)
    next_obs = make_obs(position=(position[0], position[1] + 1))
    agent.on_transition(Transition(obs, ACTIONS[action_index], next_obs, terminated, truncated))


def small_agent(**kwargs):
    settings = dict(reward_fn=TableReward([0.0] * N_ACTIONS), rollout_steps=8, epochs=2, batch_size=4,
                    lr=1e-2, gamma=0.9)
    settings.update(kwargs)
    torch.manual_seed(0)
    return PpoAgent(build_actor_critic(hidden=16), **settings)

###############################################################################
# the network
###############################################################################

def test_actor_critic_gives_scores_and_one_value_per_input():
    logits, value = build_actor_critic()(torch.zeros(3, N_INPUTS))
    assert tuple(logits.shape) == (3, N_ACTIONS) and tuple(value.shape) == (3,)
    assert logits.dtype == torch.float32 and value.dtype == torch.float32


def test_actor_critic_takes_other_sizes():
    logits, value = ActorCritic(n_inputs=7, n_actions=2, hidden=5)(torch.zeros(4, 7))
    assert tuple(logits.shape) == (4, 2) and tuple(value.shape) == (4,)


def test_actor_critic_repeats_with_the_same_torch_seed():
    torch.manual_seed(3)
    a = build_actor_critic()
    torch.manual_seed(3)
    b = build_actor_critic()
    x = torch.ones(1, N_INPUTS)
    assert all(torch.equal(u, v) for u, v in zip(a(x), b(x)))

###############################################################################
# compute_gae
###############################################################################

def test_gae_of_one_terminated_step_is_reward_minus_value():
    adv, ret = compute_gae([1.0], [0.4], [9.0], [1.0], [1.0], 0.9, 0.95)
    assert np.allclose(adv, [0.6]) and np.allclose(ret, [1.0])           # the next value (9) is ignored


def test_gae_bootstraps_from_the_next_value_when_not_terminated():
    adv, ret = compute_gae([0.0], [0.5], [2.0], [0.0], [1.0], 0.9, 0.95)
    assert np.allclose(adv, [0.0 + 0.9 * 2.0 - 0.5])                     # a game cut by the clock bootstraps
    assert np.allclose(ret, adv + 0.5)


def test_gae_lambda_0_is_the_one_step_error_and_lambda_1_is_the_whole_return():
    r, v, nv = [1.0, 0.0, 2.0], [0.2, 0.5, 0.1], [0.5, 0.1, 9.0]
    term, ended = [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
    g = 0.9

    adv0, _ = compute_gae(r, v, nv, term, ended, g, 0.0)
    assert np.allclose(adv0, [1.0 + g * 0.5 - 0.2, 0.0 + g * 0.1 - 0.5, 2.0 - 0.1])

    adv1, ret1 = compute_gae(r, v, nv, term, ended, g, 1.0)
    discounted = [1.0 + g * 0.0 + g * g * 2.0, 0.0 + g * 2.0, 2.0]       # the real return of each step
    assert np.allclose(ret1, discounted) and np.allclose(adv1, np.array(discounted) - v)


def test_gae_does_not_cross_an_episode_end():
    # step 0 ends its episode (truncated, so it still bootstraps); step 1 belongs to the next episode
    adv, _ = compute_gae([0.0, 5.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], 0.9, 1.0)
    assert np.allclose(adv, [0.9 * 1.0, 5.0])                            # step 1's reward does not leak into step 0


def test_gae_returns_float_arrays_of_the_same_length():
    adv, ret = compute_gae([1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], 0.9, 0.9)
    assert adv.shape == ret.shape == (3,) and adv.dtype == ret.dtype == np.float64

###############################################################################
# construction
###############################################################################

def test_builds_from_a_network_with_the_right_shape():
    agent = PpoAgent(build_actor_critic())
    assert agent.name == "ppo" and agent.training is True and agent.rollout_size == 0


def test_wrong_number_of_actions_is_refused():
    for n in (1, N_ACTIONS - 1, N_ACTIONS + 1):
        assert raises(ValueError, PpoAgent, build_actor_critic(n_actions=n))


def test_wrong_number_of_inputs_is_refused():
    for n in (1, N_INPUTS - 1, N_INPUTS + 1):
        assert raises(ValueError, PpoAgent, build_actor_critic(n_inputs=n))


def test_a_network_that_gives_no_value_is_refused():
    class OnlyScores(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(N_ACTIONS))
        def forward(self, x):
            return self.w.expand(x.shape[0], -1), torch.zeros(x.shape[0], 2)     # a value of the wrong shape
    assert raises(ValueError, PpoAgent, OnlyScores())
    assert raises(ValueError, PpoAgent, nn.Linear(N_INPUTS, N_ACTIONS))           # not (scores, value) at all


def test_construction_refuses_bad_settings():
    net = build_actor_critic()
    for bad in (dict(eval_epsilon=-0.1), dict(eval_epsilon=1.1), dict(gamma=1.5), dict(gae_lambda=-0.1),
                dict(clip_epsilon=0.0), dict(entropy_coef=-1.0), dict(value_coef=-1.0), dict(lr=0.0),
                dict(rollout_steps=1), dict(epochs=0), dict(batch_size=0)):
        assert raises(ValueError, PpoAgent, net, **bad), bad
    PpoAgent(net, eval_epsilon=1.0, gamma=1.0, gae_lambda=0.0)                    # the limits are allowed


def test_a_reward_with_another_gamma_or_window_is_refused():
    assert raises(ValueError, PpoAgent, build_actor_critic(), reward_fn=SimpleNamespace(gamma=0.5), gamma=0.99)
    assert raises(ValueError, PpoAgent, build_actor_critic(), reward_fn=SimpleNamespace(window=3), history_size=8)


def test_seed_and_history_size_are_passed_to_the_base_agent():
    agent = PpoAgent(prefers(0), seed=7, history_size=3)
    assert agent.history.maxlen == 3
    assert list(agent.rng.random(3)) == list(np.random.default_rng(7).random(3))

###############################################################################
# playing: training samples the policy
###############################################################################

def test_act_returns_an_engine_action():
    assert isinstance(PpoAgent(build_actor_critic()).act(make_obs()), Action)


def test_probabilities_are_the_softmax_of_the_scores():
    agent = PpoAgent(FixedNet([0.0, 1.0, 2.0, 3.0]))
    agent.act(make_obs())
    p = agent.action_probabilities()
    expected = np.exp([0.0, 1.0, 2.0, 3.0]) / np.exp([0.0, 1.0, 2.0, 3.0]).sum()
    assert np.allclose(p, expected) and abs(p.sum() - 1.0) < 1e-12


def test_training_samples_in_proportion_to_the_probabilities():
    agent = PpoAgent(FixedNet([0.0, 1.0, 2.0, 0.5]), seed=1)
    taken = [ACTIONS.index(a) for a in actions_taken(agent, 4000)]
    share = np.bincount(taken, minlength=N_ACTIONS) / 4000
    agent.act(make_obs())
    assert np.allclose(share, agent.action_probabilities(), atol=0.04), share


def test_a_certain_policy_always_takes_that_action():
    for i in range(N_ACTIONS):
        agent = PpoAgent(FixedNet([-1e4 if j != i else 1e4 for j in range(N_ACTIONS)]))
        assert all(a == ACTIONS[i] for a in actions_taken(agent, 30))


def test_training_uses_one_number_of_the_rng_per_decision():
    agent = PpoAgent(build_actor_critic(), seed=5)
    actions_taken(agent, 7)
    twin = np.random.default_rng(5)
    twin.random(7)
    assert agent.rng.random() == twin.random()


def test_same_seed_same_decisions_and_another_seed_other_ones():
    torch.manual_seed(0)
    net = build_actor_critic()
    a = PpoAgent(copy.deepcopy(net), seed=1)
    b = PpoAgent(copy.deepcopy(net), seed=1)
    c = PpoAgent(copy.deepcopy(net), seed=2)
    taken_a, taken_b, taken_c = (actions_taken(x, 60) for x in (a, b, c))
    assert taken_a == taken_b and taken_a != taken_c


def test_the_network_gets_the_build_input_of_the_history_as_a_float32_batch_of_one():
    seen = []

    class Spy(FixedNet):
        def forward(self, x):
            seen.append(x.clone())
            return super().forward(x)

    lengths = []
    def fake_build_input(history):
        lengths.append(len(history))
        return np.arange(N_INPUTS, dtype=np.float64)           # float64 on purpose

    agent = PpoAgent(Spy([0.0] * N_ACTIONS))
    seen.clear()                                               # the check at construction
    with patched("build_input", fake_build_input):
        agent.act(make_obs())
        agent.act(make_obs(position=(1, 1)))

    assert lengths == [1, 2]
    assert all(x.dtype == torch.float32 and tuple(x.shape) == (1, N_INPUTS) for x in seen)
    assert torch.equal(seen[0][0], torch.arange(N_INPUTS, dtype=torch.float32))


def test_deciding_builds_no_gradient():
    net = build_actor_critic()
    agent = PpoAgent(net)
    agent.act(make_obs())
    assert all(p.grad is None for p in net.parameters())

###############################################################################
# playing: eval mode
###############################################################################

def test_eval_mode_takes_the_largest_score():
    for i in range(N_ACTIONS):
        agent = PpoAgent(prefers(i, strength=0.1))
        agent.set_eval_mode()
        assert actions_taken(agent, 20) == [ACTIONS[i]] * 20


def test_eval_ties_go_to_the_first_action():
    agent = PpoAgent(FixedNet([0.5] * N_ACTIONS))
    agent.set_eval_mode()
    assert agent.act(make_obs()) == ACTIONS[0]
    agent = PpoAgent(FixedNet([0.0, 2.0, 2.0, 1.0]))
    agent.set_eval_mode()
    assert agent.act(make_obs()) == ACTIONS[1]


def test_a_greedy_eval_agent_never_touches_the_rng():
    agent = PpoAgent(build_actor_critic(), seed=3)
    agent.set_eval_mode()
    actions_taken(agent, 10)
    assert agent.rng.random() == np.random.default_rng(3).random()


def test_eval_epsilon_explores_in_eval_mode_only():
    agent = PpoAgent(prefers(0, strength=100.0), eval_epsilon=1.0, seed=1)
    assert all(a == ACTIONS[0] for a in actions_taken(agent, 30))                 # training: the policy, not epsilon
    agent.set_eval_mode()
    assert len(set(actions_taken(agent, 200))) == N_ACTIONS                      # eval: random actions appear


def test_the_share_of_random_eval_actions_follows_eval_epsilon():
    agent = PpoAgent(prefers(0, strength=100.0), eval_epsilon=0.3, seed=2)
    agent.set_eval_mode()
    off = sum(a != ACTIONS[0] for a in actions_taken(agent, 4000)) / 4000
    assert abs(off - 0.3 * (N_ACTIONS - 1) / N_ACTIONS) < 0.03, off

###############################################################################
# learning: what is stored
###############################################################################

def test_a_step_is_stored_with_its_log_probability_value_reward_and_flags():
    net = FixedNet([0.0, 1.0, 2.0, 3.0], value=0.7)
    agent = PpoAgent(net, reward_fn=TableReward([0.0, 0.0, 0.25, 0.0]), rollout_steps=100)
    play_step(agent, 2, terminated=False, truncated=True)

    assert agent.rollout_size == 1
    logp = np.log(np.exp(2.0) / np.exp([0.0, 1.0, 2.0, 3.0]).sum())
    assert agent._actions == [2] and abs(agent._log_probs[0] - logp) < 1e-6
    assert abs(agent._values[0] - 0.7) < 1e-6 and agent._rewards == [0.25]
    assert agent._terminated == [0.0] and agent._ended == [1.0]                  # truncated: ended, not terminated
    assert agent._states[0].shape == (N_INPUTS,) and agent._next_states[0].shape == (N_INPUTS,)


def test_the_next_state_is_built_from_the_history_moved_one_step_forward():
    seen = []
    def fake_build_input(history):
        seen.append([o.position for o in history])
        return np.zeros(N_INPUTS)

    agent = PpoAgent(prefers(0), rollout_steps=100, history_size=2)
    with patched("build_input", fake_build_input):
        agent.reset()
        for position in ((1, 1), (1, 2)):
            agent.act(make_obs(position=position))
        seen.clear()
        agent.on_transition(Transition(agent.history[-1], ACTIONS[0], make_obs(position=(1, 3)), False, False))

    assert seen == [[(1, 1), (1, 2)], [(1, 2), (1, 3)]]                          # state, then the window moved on


def test_on_transition_does_nothing_in_eval_mode():
    net = build_actor_critic()
    agent = PpoAgent(net, rollout_steps=2)
    agent.set_eval_mode()
    before = copy.deepcopy(net.state_dict())
    for _ in range(4):
        play_step(agent, 0)
    assert agent.rollout_size == 0 and agent.updates == 0
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())


def test_reset_resets_the_reward():
    reward = TableReward([0.0] * N_ACTIONS)
    agent = small_agent(reward_fn=reward)
    agent.reset()
    agent.reset()
    assert reward.resets == 2

###############################################################################
# learning: when it updates
###############################################################################

def test_nothing_is_learned_before_the_rollout_is_full_then_one_update_empties_it():
    agent = small_agent(rollout_steps=8)
    before = copy.deepcopy(agent.net.state_dict())
    for _ in range(7):
        play_step(agent, 1)
    assert agent.updates == 0 and agent.rollout_size == 7 and agent.last_loss is None
    assert all(torch.equal(before[k], v) for k, v in agent.net.state_dict().items())

    play_step(agent, 1)
    assert agent.updates == 1 and agent.rollout_size == 0 and agent.last_loss is not None
    assert not all(torch.equal(before[k], v) for k, v in agent.net.state_dict().items())

    for _ in range(8):
        play_step(agent, 1)
    assert agent.updates == 2


def test_the_rollout_may_hold_several_episodes():
    agent = small_agent(rollout_steps=6)
    for _ in range(3):
        play_step(agent, 0, terminated=True)
        play_step(agent, 1, terminated=False, truncated=True)
    assert agent.updates == 1 and agent.rollout_size == 0


def test_the_last_step_of_a_rollout_closes_the_advantage_chain_and_flags_are_passed_on():
    captured = {}
    original = compute_gae

    def spy(rewards, values, next_values, terminated, ended, gamma, gae_lambda):
        captured.update(terminated=list(terminated), ended=list(ended), n=len(rewards), gamma=gamma, lam=gae_lambda)
        return original(rewards, values, next_values, terminated, ended, gamma, gae_lambda)

    agent = small_agent(rollout_steps=4, gamma=0.9, gae_lambda=0.8)
    with patched("compute_gae", spy):
        play_step(agent, 0, terminated=True)
        play_step(agent, 0, terminated=False, truncated=True)
        play_step(agent, 0, terminated=False, truncated=False)         # the clock has not struck: no end ...
        play_step(agent, 0, terminated=False, truncated=False)         # ... but the rollout stops here
    assert captured["n"] == 4 and captured["terminated"] == [1.0, 0.0, 0.0, 0.0]
    assert captured["ended"] == [1.0, 1.0, 0.0, 1.0]
    assert captured["gamma"] == 0.9 and captured["lam"] == 0.8


def test_the_first_minibatch_has_ratio_1_so_the_policy_loss_is_zero_after_normalizing():
    # One epoch over one batch, no value or entropy term: the loss is -mean(normalized advantages) = 0
    agent = small_agent(rollout_steps=8, epochs=1, batch_size=8, value_coef=0.0, entropy_coef=0.0,
                        reward_fn=TableReward([1.0, 0.0, 0.0, 0.0]))
    for i in range(8):
        play_step(agent, i % 2)
    assert agent.updates == 1 and abs(agent.last_loss) < 1e-5, agent.last_loss


def test_every_minibatch_clips_the_gradient_norm_to_1():
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = lambda params, max_norm, **kw: calls.append(max_norm) or original(params, max_norm, **kw)
    try:
        agent = small_agent(rollout_steps=8, epochs=2, batch_size=4)
        for _ in range(8):
            play_step(agent, 1)
    finally:
        torch.nn.utils.clip_grad_norm_ = original
    assert calls == [1.0] * 4                                          # 2 epochs x 2 minibatches


def test_a_ratio_beyond_the_clip_gets_no_policy_gradient():
    # Step 0 has a positive advantage and a ratio far above 1 + clip; the others a negative advantage and a
    # ratio far below 1 - clip. In both cases the clipped term is the one taken: nothing moves the policy.
    def one_update(shift):
        agent = small_agent(rollout_steps=5, epochs=1, batch_size=4, value_coef=0.0, entropy_coef=0.0)
        for i in range(4):                                             # one short of a full rollout
            play_step(agent, i)
        agent._log_probs = [lp + d for lp, d in zip(agent._log_probs, shift)]
        before = copy.deepcopy(agent.net.state_dict())
        with patched("compute_gae", lambda *a: (np.array([3.0, -1.0, -1.0, -1.0]), np.zeros(4))):
            agent._update()
        return all(torch.equal(before[k], v) for k, v in agent.net.state_dict().items())

    assert one_update([-3.0, 3.0, 3.0, 3.0]) is True                   # far outside the clip: no move
    assert one_update([0.0, 0.0, 0.0, 0.0]) is False                   # ratio 1: the policy moves


def test_the_agent_learns_which_action_pays():
    agent = small_agent(reward_fn=TableReward([-1.0, 0.0, 1.0, 0.0]), rollout_steps=16, epochs=4, batch_size=8)
    for i in range(400):
        play_step(agent, i % N_ACTIONS)                                # every action tried equally
    agent.reset()
    agent.act(make_obs())
    p = agent.action_probabilities()
    assert int(np.argmax(p)) == 2 and p[2] > 0.5 and p[0] < 0.2, p


def test_the_critic_learns_the_value_of_a_state():
    agent = small_agent(reward_fn=TableReward([1.0] * N_ACTIONS), rollout_steps=16, epochs=4, batch_size=8, lr=3e-2)
    for i in range(300):
        play_step(agent, i % N_ACTIONS)                                # every step ends the game with reward 1
    agent.reset()
    agent.act(make_obs())
    x = torch.as_tensor(agent_module.build_input(agent.history), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, value = agent.net(x)
    assert abs(value.item() - 1.0) < 0.25, value.item()


def test_same_seed_same_transitions_same_weights():
    torch.manual_seed(1)
    net = build_actor_critic(hidden=8)
    agents = [PpoAgent(copy.deepcopy(net), reward_fn=TableReward([0.0, 1.0, 0.0, 0.0]), rollout_steps=8,
                       epochs=2, batch_size=4, seed=3) for _ in range(2)]
    for agent in agents:
        for i in range(40):
            play_step(agent, i % 4, position=(1 + i % 3, 1))
    assert same_weights(agents[0].net, agents[1].net)

###############################################################################
# clone
###############################################################################

def test_clone_has_its_own_network_with_equal_weights():
    agent = PpoAgent(build_actor_critic(), eval_epsilon=0.2)
    other = agent.clone()
    assert other.net is not agent.net and same_weights(agent.net, other.net) and other.eval_epsilon == 0.2
    with torch.no_grad():
        for p in other.net.parameters():
            p.add_(1.0)
    assert not any(torch.equal(p, q) for p, q in zip(agent.net.parameters(), other.net.parameters()))


def test_clone_keeps_the_mode_and_the_rng_state_and_starts_a_clean_episode():
    agent = PpoAgent(prefers(0), seed=9, history_size=4)
    agent.set_eval_mode()
    agent.rng.random(5)
    for position in ((1, 1), (1, 2), (2, 2)):
        agent.act(make_obs(position=position))
    other = agent.clone()

    assert other.training is False
    assert len(other.history) == 0 and other.history.maxlen == 4 and other.positions == []
    assert len(agent.history) == 3
    assert list(other.rng.random(3)) == list(agent.rng.random(3))


def test_clone_decides_like_the_original():
    agent = PpoAgent(build_actor_critic(), seed=4)
    other = agent.clone()
    assert actions_taken(agent, 40) == actions_taken(other, 40)


def test_clone_carries_the_network_but_not_the_memory():
    agent = small_agent(rollout_steps=8)
    for _ in range(11):
        play_step(agent, 1)
    assert agent.updates == 1 and agent.rollout_size == 3

    other = agent.clone()
    assert same_weights(agent.net, other.net) and other.net is not agent.net
    assert other.rollout_size == 0 and other.updates == 0 and other.last_loss is None
    assert agent.rollout_size == 3                                      # the original keeps its memory
    assert other.optimizer is not agent.optimizer and len(other.optimizer.state) == 0


def test_training_the_clone_leaves_the_original_alone():
    agent = small_agent()
    for i in range(8):
        play_step(agent, i % 4)
    other = agent.clone()
    before = copy.deepcopy(agent.net.state_dict())
    for i in range(12):
        play_step(other, 1)
    assert other.updates > 0
    assert all(torch.equal(before[k], v) for k, v in agent.net.state_dict().items())


def test_an_agent_that_has_played_can_be_cloned():
    # Its history holds engine observations that cannot be copied: clone() builds a new agent
    agent = small_agent()
    for k in range(6):
        play_step(agent, k % 4)
    agent.act(make_obs())                                               # an episode in progress
    assert len(agent.history) >= 1 and agent.positions

    other = agent.clone()
    assert len(other.history) == 0 and other.positions == [] and other.actions == []
    assert same_weights(agent.net, other.net)


def test_clone_keeps_the_reward_settings_and_does_not_share_the_reward_object():
    reward = TableReward([0.0, 1.0, 0.0, 0.0])
    agent = small_agent(reward_fn=reward)
    other = agent.clone()
    assert other.reward_fn is not reward and other.reward_fn.table == reward.table


def test_the_agent_survives_pickle():
    agent = small_agent()
    for i in range(12):
        play_step(agent, i % 4)
    agent.reset()                          # between episodes: a played agent holds engine observations
    other = pickle.loads(pickle.dumps(agent))
    assert other.rollout_size == agent.rollout_size and other.updates == agent.updates
    play_step(agent, 1)
    play_step(other, 1)
    assert other.rollout_size == agent.rollout_size
    assert np.array_equal(np.array(agent._states), np.array(other._states))

###############################################################################
# Direct run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"ppo agent: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()