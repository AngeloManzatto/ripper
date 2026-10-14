"""
Created on Tue Oct 13 13:09:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

from agents.linear_q_agent.linear_q import update, q_values

N_ACTIONS = 4


def fresh_W(n_features):
    return np.zeros((N_ACTIONS, n_features + 1))      # +1 for the bias column


def test_terminal_reward_gives_delta_equal_reward():
    W = fresh_W(3)
    x = np.array([1.0, 0.0, 2.0])

    delta = update(W, x, 1, 1.0, x, True, 0.1, 0.9)

    assert np.isclose(delta, 1.0)
    assert np.allclose(W[1], [0.1, 0.0, 0.2, 0.1])     # alpha * delta * [x, 1]
    assert np.all(W[[0, 2, 3]] == 0.0)                 # other actions untouched


def test_bootstrap_uses_max_q_of_next_state():
    W = fresh_W(3)
    W[0, -1] = 0.5                  # action 0 has Q = 0.5 in every state (bias only)
    x = np.array([1.0, 0.0, 2.0])

    delta = update(W, x, 1, 0.2, x, False, 0.1, 0.9)

    assert np.isclose(delta, 0.2 + 0.9 * 0.5 - 0.0)    # 0.65


def test_terminated_cuts_the_bootstrap():
    W = fresh_W(3)
    W[0, -1] = 0.5
    x = np.array([1.0, 0.0, 2.0])

    delta = update(W, x, 1, 0.2, x, True, 0.1, 0.9)

    assert np.isclose(delta, 0.2)                      # no gamma * max Q term


def test_q_change_equals_alpha_delta_norm_squared():
    """The identity from the toolkit, checked against the real update."""
    rng = np.random.default_rng(0)
    alpha, gamma = 0.05, 0.95

    for terminated in (False, True):
        W = rng.normal(size=(N_ACTIONS, 6))            # 5 features + bias
        x = rng.normal(size=5)
        x_next = rng.normal(size=5)

        q_before = q_values(W, x)[2]
        delta = update(W, x, 2, 0.3, x_next, terminated, alpha, gamma)
        q_after = q_values(W, x)[2]

        x_aug = np.append(x, 1.0)
        assert np.isclose(q_after - q_before, alpha * delta * (x_aug @ x_aug))


def run():
    test_terminal_reward_gives_delta_equal_reward()
    test_bootstrap_uses_max_q_of_next_state()
    test_terminated_cuts_the_bootstrap()
    test_q_change_equals_alpha_delta_norm_squared()
    print("td error: all checks passed")


if __name__ == "__main__":
    run()