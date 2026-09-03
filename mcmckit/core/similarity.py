"""Shape-similarity helpers.

A small, general utility: how well two vectors line up, ignoring their sign and
scale. Written for comparing mode shapes, where it is known as the Modal
Assurance Criterion, but nothing here is specific to structural dynamics - it is
just a squared normalised inner product.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mac", "mac_matrix"]


def mac(a, b) -> float:
    r"""Squared normalised inner product of two vectors.

    .. math::

        \mathrm{MAC}(a, b) = \frac{|a^H b|^2}{(a^H a)(b^H b)}

    1.0 when the two are parallel, 0.0 when orthogonal. Squaring the numerator
    is what makes the result invariant to both sign and scale, which matters
    because an eigensolver's choice of sign and normalisation is arbitrary.

    Complex input is handled with the conjugate inner product, so complex mode
    shapes - from non-proportional damping, or from operational modal analysis -
    give the right answer rather than silently losing their imaginary part.

    Parameters
    ----------
    a, b : array-like
        Vectors of the same length. Real or complex.

    Returns
    -------
    float
        In [0, 1]. Returns 0.0 if either vector is zero.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([1.0, 2.0, 3.0])
    >>> mac(v, v)
    1.0
    >>> bool(np.isclose(mac(v, -2.5 * v), 1.0))    # sign and scale invariant
    True
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"vectors must have the same shape, got {a.shape} and {b.shape}")
    num = abs(complex(np.vdot(a, b))) ** 2
    den = float(np.vdot(a, a).real) * float(np.vdot(b, b).real)
    return num / den if den > 1e-300 else 0.0


def mac_matrix(shapes_a, shapes_b) -> np.ndarray:
    """All pairwise :func:`mac` values between two sets of column vectors.

    Parameters
    ----------
    shapes_a : array-like, shape (n, n_a)
        Column vectors. A 1-D input is treated as a single column.
    shapes_b : array-like, shape (n, n_b)

    Returns
    -------
    np.ndarray, shape (n_a, n_b)
        ``M[i, j] = mac(shapes_a[:, i], shapes_b[:, j])``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.eye(3)
    >>> bool(np.allclose(mac_matrix(A, A), np.eye(3)))
    True
    """
    shapes_a = np.asarray(shapes_a)
    shapes_b = np.asarray(shapes_b)
    if shapes_a.ndim == 1:
        shapes_a = shapes_a[:, np.newaxis]
    if shapes_b.ndim == 1:
        shapes_b = shapes_b[:, np.newaxis]
    if shapes_a.shape[0] != shapes_b.shape[0]:
        raise ValueError(
            "both sets must have the same number of rows, got "
            f"{shapes_a.shape[0]} and {shapes_b.shape[0]}"
        )
    return np.array(
        [[mac(shapes_a[:, i], shapes_b[:, j]) for j in range(shapes_b.shape[1])]
         for i in range(shapes_a.shape[1])]
    )
