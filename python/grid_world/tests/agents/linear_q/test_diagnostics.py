"""
Created on Tue Oct 13 13:12:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

from agents.linear_q_agent.linear_q import update, q_values
from agents.linear_q_agent.diagnostics import (
    action_margins,
    feature_second_moments,
    moment_eigenvalues,
    condition_number,
    binomial_se,
    expected_q_change,
)


def raises(error, fn, *args):
    try:
        fn(*args)
    except error:
        return True
    return False


def test_action_margins():
    Q = np.array([[0.1, 0.5, 0.3, 0.2],
                  [1.0, 1.0, 0.0, 0.0],
                  [-1.0, -3.0, -2.0, -5.0]])

    assert np.allclose(action_margins(Q), [0.2, 0.0, 1.0])
    assert raises(ValueError, action_margins, np.array([0.1, 0.2]))        # not 2D
    assert raises(ValueError, action_margins, np.array([[0.1], [0.2]]))    # one action


def test_feature_second_moments():
    X = np.array([[1.0, 0.0],
                  [1.0, 2.0]])

    assert np.allclose(feature_second_moments(X), [1.0, 2.0])
    assert raises(ValueError, feature_second_moments, np.array([1.0, 2.0]))
    assert raises(ValueError, feature_second_moments, np.zeros((0, 2)))


def test_moment_eigenvalues_sorted_largest_first():
    X = np.array([[1.0, 0.0],
                  [0.0, 0.1]])

    assert np.allclose(moment_eigenvalues(X), [0.5, 0.005])
    
def test_moment_eigenvalues_divides_by_T_not_by_shape():
    X = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])                  # T = 3, n_features = 2
    # X^T X / 3 = [[2/3, 1/3], [1/3, 2/3]]  ->  eigenvalues 1 and 1/3
    assert np.allclose(moment_eigenvalues(X), [1.0, 1.0 / 3.0])


def test_moment_eigenvalues_realistic_shape():
    X = np.random.default_rng(0).normal(size=(1000, 25))
    ev = moment_eigenvalues(X)

    assert ev.shape == (25,)
    assert np.all(np.diff(ev) <= 0)                # sorted largest first
    assert np.all(ev > 0)    


def test_condition_number():
    X = np.array([[1.0, 0.0],
                  [0.0, 0.1]])
    assert np.isclose(condition_number(X), 100.0)

    dead_column = np.array([[1.0, 0.0],
                            [1.0, 0.0]])
    assert np.isclose(condition_number(dead_column), 1.0)   # the zero eigenvalue is ignored

    assert raises(ValueError, condition_number, np.zeros((3, 2)))


def test_binomial_se():
    assert np.isclose(binomial_se(0.5, 100), 0.05)
    assert np.isclose(binomial_se(0.15, 100), 0.035707, atol=1e-5)
    assert binomial_se(0.0, 100) == 0.0
    assert raises(ValueError, binomial_se, 1.2, 100)
    assert raises(ValueError, binomial_se, -0.1, 100)
    assert raises(ValueError, binomial_se, 0.5, 0)


def test_expected_q_change_by_hand():
    x = np.array([1.0, 0.0, 2.0])             # x_aug . x_aug = 1 + 0 + 4 + 1 = 6
    assert np.isclose(expected_q_change(x, 1.0, 0.1), 0.6)


def test_expected_q_change_matches_the_real_update():
    rng = np.random.default_rng(1)
    alpha, gamma = 0.02, 0.95

    W = rng.normal(size=(4, 26))               # 25 features + bias, like the real agent
    x = rng.normal(size=25)
    x_next = rng.normal(size=25)

    q_before = q_values(W, x)[0]
    delta = update(W, x, 0, 1.0, x_next, False, alpha, gamma)
    q_after = q_values(W, x)[0]

    assert np.isclose(q_after - q_before, expected_q_change(x, delta, alpha))


def run():
    test_action_margins()
    test_feature_second_moments()
    test_moment_eigenvalues_sorted_largest_first()
    test_moment_eigenvalues_divides_by_T_not_by_shape()
    test_moment_eigenvalues_realistic_shape()
    test_condition_number()
    test_binomial_se()
    test_expected_q_change_by_hand()
    test_expected_q_change_matches_the_real_update()
    print("diagnostics: all checks passed")


if __name__ == "__main__":
    run()