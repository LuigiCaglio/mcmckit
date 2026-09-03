"""The mac / mac_matrix helper."""

import numpy as np
import pytest

import mcmckit as mc


def test_identical_vectors_give_one():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(mc.mac(v, v), 1.0)


def test_orthogonal_vectors_give_zero():
    assert np.isclose(mc.mac([1.0, 0.0], [0.0, 1.0]), 0.0)


@pytest.mark.parametrize("factor", [-1.0, 2.5, -0.01, 1e6])
def test_invariant_to_sign_and_scale(factor):
    """The whole point of squaring the numerator.

    An eigensolver's sign and normalisation are arbitrary, so a mode compared
    against a rescaled or flipped copy of itself must still score 1.
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(10)
    assert np.isclose(mc.mac(v, factor * v), 1.0)


def test_complex_vectors_use_the_conjugate_inner_product():
    """Complex mode shapes arise from non-proportional damping and from OMA.

    Casting them to real would silently discard the imaginary part and give a
    wrong number rather than an error.
    """
    rng = np.random.default_rng(1)
    v = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    assert np.isclose(mc.mac(v, v), 1.0)
    assert np.isclose(mc.mac(v, (3.0 + 2.0j) * v), 1.0)   # complex scaling too


def test_result_is_bounded_by_zero_and_one():
    rng = np.random.default_rng(2)
    for _ in range(50):
        a, b = rng.standard_normal(7), rng.standard_normal(7)
        value = mc.mac(a, b)
        assert 0.0 <= value <= 1.0 + 1e-12


def test_is_symmetric():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(5), rng.standard_normal(5)
    assert np.isclose(mc.mac(a, b), mc.mac(b, a))


def test_zero_vector_gives_zero_rather_than_dividing_by_zero():
    assert mc.mac(np.zeros(4), np.ones(4)) == 0.0
    assert mc.mac(np.zeros(4), np.zeros(4)) == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        mc.mac([1.0, 2.0], [1.0, 2.0, 3.0])


def test_matrix_of_an_orthonormal_basis_is_the_identity():
    assert np.allclose(mc.mac_matrix(np.eye(4), np.eye(4)), np.eye(4))


def test_matrix_shape_and_entries():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((6, 3))
    B = rng.standard_normal((6, 2))
    M = mc.mac_matrix(A, B)
    assert M.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert np.isclose(M[i, j], mc.mac(A[:, i], B[:, j]))


def test_matrix_accepts_a_single_vector():
    v = np.array([1.0, 2.0, 3.0])
    assert mc.mac_matrix(v, v).shape == (1, 1)
    assert np.isclose(mc.mac_matrix(v, v)[0, 0], 1.0)


def test_matrix_rejects_mismatched_row_counts():
    with pytest.raises(ValueError, match="same number of rows"):
        mc.mac_matrix(np.eye(3), np.eye(4))


def test_matrix_pairs_a_shuffled_basis_back_up():
    """The practical use: work out which column matches which."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((8, 4))
    order = [2, 0, 3, 1]
    B = A[:, order] * rng.choice([-1.0, 1.0, 3.0], size=4)   # shuffled, flipped, rescaled
    assert list(np.argmax(mc.mac_matrix(A, B), axis=1)) == [order.index(i) for i in range(4)]
