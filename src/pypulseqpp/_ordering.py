"""Traversal orders over a single encoded axis.

Each traversal is reached through :func:`calc_traversal_order`, which names
them, so the individual ones are private to this module.
"""

from __future__ import annotations

import numpy as np


def _count(n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return n


def _sequential(n: int) -> np.ndarray:
    return np.arange(_count(n), dtype=np.intp)


def _reverse(n: int) -> np.ndarray:
    return _sequential(n)[::-1].copy()


def _interleaved(n: int) -> np.ndarray:
    values = _sequential(n)
    return np.concatenate((values[::2], values[1::2]))


def _center_out(n: int) -> np.ndarray:
    """Order positions by distance from the centre, the lower index first on a tie."""
    n = _count(n)
    center = (n - 1) / 2.0
    return np.asarray(
        sorted(range(n), key=lambda index: (abs(index - center), index)), dtype=np.intp
    )


def _outside_in(n: int) -> np.ndarray:
    return _center_out(n)[::-1].copy()


def _random_order(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(_count(n))


_TRAVERSALS = {
    "sequential": _sequential,
    "reverse": _reverse,
    "interleaved": _interleaved,
    "center_out": _center_out,
    "outside_in": _outside_in,
}


def calc_traversal_order(
    n: int, order: str = "sequential", *, seed: int = 0
) -> np.ndarray:
    """Return a permutation of zero-based positions for a one-dimensional loop.

    Parameters
    ----------
    n : int
        Number of positions.
    order : {'sequential', 'reverse', 'interleaved', 'center_out', 'outside_in', 'random'}, default='sequential'
        Traversal scheme. ``'interleaved'`` takes even positions then odd ones
        (the standard multi-slice choice, maximising the gap between
        neighbours); ``'center_out'`` starts at the middle, which is the usual
        choice for a partition loop whose contrast is set by the first echoes.
    seed : int, default=0
        Seed for ``order='random'``.

    Returns
    -------
    numpy.ndarray
        Integer permutation of ``range(n)``.

    Raises
    ------
    ValueError
        If ``n`` is negative or ``order`` is unknown.

    Examples
    --------
    >>> from pypulseqpp._ordering import calc_traversal_order
    >>> calc_traversal_order(6, "interleaved").tolist()
    [0, 2, 4, 1, 3, 5]
    >>> calc_traversal_order(5, "center_out").tolist()
    [2, 1, 3, 0, 4]
    """
    if order == "random":
        return _random_order(n, seed)
    try:
        return _TRAVERSALS[order](n)
    except KeyError:
        raise ValueError(
            f"unknown order {order!r}; expected one of {', '.join(sorted((*_TRAVERSALS, 'random')))}"
        ) from None
