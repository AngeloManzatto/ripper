import numpy as np
from core.danger import hostile_proximity, N_PROXIMITY


def test_nothing_visible_gives_zeros():
    assert hostile_proximity(False, 0, 0) == [0.0] * N_PROXIMITY
    assert hostile_proximity(False, -2, 3) == [0.0] * N_PROXIMITY   # found=False wins


def test_each_direction_lights_the_right_slot():
    # up, down, left, right, in DIRECTIONS order; 1 cell away -> 0.8
    assert np.allclose(hostile_proximity(True, -1, 0), [0.8, 0, 0, 0])
    assert np.allclose(hostile_proximity(True,  1, 0), [0, 0.8, 0, 0])
    assert np.allclose(hostile_proximity(True,  0, -1), [0, 0, 0.8, 0])
    assert np.allclose(hostile_proximity(True,  0,  1), [0, 0, 0, 0.8])


def test_closer_is_stronger():
    far  = hostile_proximity(True, -4, 0)[0]
    near = hostile_proximity(True, -2, 0)[0]
    assert np.isclose(far, 0.2) and np.isclose(near, 0.6) and near > far


def test_diagonal_lights_two_slots():
    assert np.allclose(hostile_proximity(True, -2, 3), [0.6, 0, 0, 0.4])


def test_radius_and_beyond_give_zero():
    assert np.allclose(hostile_proximity(True, -5, 0), [0, 0, 0, 0])
    assert np.allclose(hostile_proximity(True, -9, 0), [0, 0, 0, 0])