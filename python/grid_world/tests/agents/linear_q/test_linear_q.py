"""
Created on Sat Oct 10 19:50:08 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from agents.linear_q_agent.linear_q import q_values, update, epsilon_greedy


def new_W():
    return np.zeros((4, 3))     # 4 actions, 2 features + bias


def test_initial_state():
    W = new_W()
    assert np.allclose(q_values(W, [1.0, 2.0]), np.zeros(4))


def test_terminal_update():
    W = new_W()
    x = [1.0, 2.0]                      # with bias: [1, 2, 1]
    delta = update(W, x, 0, 1.0, x, True, alpha=0.5, gamma=0.9)

    assert delta == 1.0                              # target 1 - Q 0
    assert np.allclose(W[0], [0.5, 1.0, 0.5])        # 0.5 * 1.0 * [1, 2, 1]
    assert np.all(W[1:] == 0.0)                      # other actions untouched
    assert np.isclose(q_values(W, x)[0], 3.0)        # 0.5 + 2.0 + 0.5


def test_bootstrap_from_next_state():
    W = new_W()
    W[1] = [1.0, 0.0, 0.0]
    x, next_x = [1.0, 2.0], [2.0, 0.0]   # next with bias: [2, 0, 1]
    # max Q(next) = action 1: 1*2 = 2.0 ; target = 0 + 0.5 * 2.0 = 1.0
    delta = update(W, x, 0, 0.0, next_x, False, alpha=0.1, gamma=0.5)

    assert np.isclose(delta, 1.0)
    assert np.allclose(W[0], [0.1, 0.2, 0.1])        # 0.1 * 1.0 * [1, 2, 1]
    assert np.allclose(W[1], [1.0, 0.0, 0.0])        # untouched


def test_terminated_cuts_the_bootstrap():
    W = new_W()
    W[1] = [1.0, 0.0, 0.0]
    x, next_x = [1.0, 2.0], [2.0, 0.0]
    delta = update(W, x, 0, 0.0, next_x, True, alpha=0.1, gamma=0.5)

    assert delta == 0.0                              # target 0, Q already 0
    assert np.all(W[0] == 0.0)


def test_update_moves_q_toward_the_target():
    W = new_W()
    W[0] = [1.0, 0.0, 0.0]
    x = [1.0, 2.0]
    assert np.isclose(q_values(W, x)[0], 1.0)

    delta = update(W, x, 0, 0.0, x, True, alpha=0.1, gamma=0.9)

    assert np.isclose(delta, -1.0)                   # target 0 - Q 1
    assert np.allclose(W[0], [0.9, -0.2, -0.1])      # 1 + 0.1 * -1 * [1, 2, 1]
    assert np.isclose(q_values(W, x)[0], 0.4)        # 0.9 - 0.4 - 0.1: closer to 0


def test_bad_inputs_raise():
    W = new_W()
    for bad in ([1.0], [1.0, 2.0, 3.0]):
        try:
            q_values(W, bad)
            assert False, "wrong feature length must raise"
        except ValueError:
            pass

    for bad_action in (4, -1):
        try:
            update(W, [1.0, 2.0], bad_action, 0.0, [1.0, 2.0], True, 0.1, 0.9)
            assert False, f"action {bad_action} must raise"
        except ValueError:
            pass

    assert np.all(W == 0.0)      # a rejected update must not change anything

###############################################################################
# Epsilon greedy test
###############################################################################

def counts(q, epsilon, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    out = [0, 0, 0, 0]
    for _ in range(n):
        out[epsilon_greedy(np.array(q, dtype=float), epsilon, rng)] += 1
    return out


def test_greedy_picks_the_best():
    assert counts([0.0, 2.0, 1.0, 0.0], 0.0, n=200) == [0, 200, 0, 0]
    # negative values: the best is the largest, not the one closest to zero
    assert counts([-3.0, -1.0, -2.0, -5.0], 0.0, n=200) == [0, 200, 0, 0]


def test_ties_are_broken_uniformly():
    c = counts([0.0, 0.0, 0.0, 0.0], 0.0)           # expected 1000 each, std ~27
    assert all(800 < k < 1200 for k in c), c

    c = counts([1.0, 1.0, 0.0, 0.0], 0.0)           # only the tied best, ~2000 each
    assert c[2] == 0 and c[3] == 0, c
    assert all(1700 < k < 2300 for k in c[:2]), c


def test_epsilon_one_is_uniform_even_if_q_has_a_favorite():
    c = counts([0.0, 5.0, 0.0, 0.0], 1.0)
    assert all(800 < k < 1200 for k in c), c


def test_epsilon_half_mixes_both():
    # P(action 1) = 0.5 + 0.5 * 0.25 = 0.625 -> expected 2500, std ~31
    c = counts([0.0, 5.0, 0.0, 0.0], 0.5)
    assert 2300 < c[1] < 2700, c
    assert all(k > 200 for k in (c[0], c[2], c[3])), c


def test_returns_plain_int_and_does_not_touch_q():
    q = np.array([0.0, 1.0, 0.0, 0.0])
    before = q.copy()
    a = epsilon_greedy(q, 0.3, np.random.default_rng(0))
    assert type(a) is int and 0 <= a < 4
    assert np.array_equal(q, before)


def test_same_seed_same_choices():
    def run_seq(seed):
        rng = np.random.default_rng(seed)
        q = np.zeros(4)
        return [epsilon_greedy(q, 0.5, rng) for _ in range(50)]

    assert run_seq(7) == run_seq(7)
    assert run_seq(7) != run_seq(8)


def test_bad_epsilon_raises():
    for bad in (-0.1, 1.1):
        try:
            epsilon_greedy(np.zeros(4), bad, np.random.default_rng(0))
            assert False, f"epsilon {bad} must raise"
        except ValueError:
            pass

def test_nan_in_q_raises():
    try:
        epsilon_greedy(np.array([0.0, np.nan, 0.0, 0.0]), 0.1, np.random.default_rng(0))
        assert False, "NaN in q must raise"
    except ValueError:
        pass
    
def run():
    test_initial_state()
    test_terminal_update()
    test_bootstrap_from_next_state()
    test_terminated_cuts_the_bootstrap()
    test_update_moves_q_toward_the_target()
    test_bad_inputs_raise()
    
    test_greedy_picks_the_best()
    test_ties_are_broken_uniformly()
    test_epsilon_one_is_uniform_even_if_q_has_a_favorite()
    test_epsilon_half_mixes_both()
    test_returns_plain_int_and_does_not_touch_q()
    test_same_seed_same_choices()
    test_bad_epsilon_raises()
    test_nan_in_q_raises()
    
    print("linear q: all checks passed")


if __name__ == "__main__":
    run()