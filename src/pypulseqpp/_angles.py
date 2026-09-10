"""Projection angles in radians and rotation matrices for radial acquisitions."""

from __future__ import annotations

__all__ = [
    "calc_golden_angles",
    "calc_projection_shell",
    "calc_raga_angles",
    "calc_tiny_golden_angles",
    "calc_uniform_angles",
]

import math

import numpy as np
from scipy.spatial.transform import Rotation

#: The golden ratio, whose irrationality is what keeps any window of
#: consecutive spokes near-uniformly distributed.
_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _accumulated(n: int, step: float) -> np.ndarray:
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return np.mod(np.arange(n, dtype=float) * float(step), 2.0 * np.pi)


def _generalized_fibonacci(order, index):
    if order == 1:
        return 1
    previous, current = 1, int(index)
    for _ in range(2, order):
        previous, current = current, previous + current
    return current


def calc_golden_angles(n: int, *, full_circle: bool = False) -> np.ndarray:
    """Return golden-angle rotations in radians, accumulated modulo 2*pi.

    For diametric spokes the increment is ``pi / phi``; for full-circle
    arms it is ``2*pi / phi**2``, where phi is the golden ratio.

    Parameters
    ----------
    n : int
        Number of angles.
    full_circle : bool, optional
        ``False`` (the default) is the ``pi``-periodic radial/blade golden
        angle ``pi / phi``; ``True`` is the full-turn spiral golden angle
        ``2 * pi / phi**2``. Default is False.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, accumulated modulo ``2 pi``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> np.rad2deg(pp.calc_golden_angles(4)).round(2)
    array([  0.  , 111.25, 222.49, 333.74])

    >>> np.rad2deg(pp.calc_golden_angles(3, full_circle=True)).round(2)
    array([  0.  , 137.51, 275.02])

    References
    ----------
    Winkelmann et al., golden-ratio profile order, DOI ``10.1109/TMI.2006.885337``.

    See Also
    --------
    calc_tiny_golden_angles : smaller increments with the same uniformity.
    calc_raga_angles : rational, exactly repeatable approximation.
    """
    step = 2.0 * np.pi / _PHI**2 if full_circle else np.pi / _PHI
    return _accumulated(n, step)


def calc_raga_angles(
    n: int, *, tiny_index: int = 1, approximation_order: int = 13
) -> np.ndarray:
    """Return rational approximate golden-angle (RAGA) rotations.

    The finite support contains ``fib(approximation_order, tiny_index)``
    equidistant angles. Angles repeat when ``n`` exceeds the support size.

    Parameters
    ----------
    n : int
        Number of angles.
    tiny_index : int, optional
        Tiny-golden index the rational approximation is built from.
    approximation_order : int, optional
        Fibonacci order; sets the size of the angular support.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, drawn from a finite equidistant support.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> angles = pp.calc_raga_angles(1000, approximation_order=8)
    >>> len(np.unique(angles.round(9)))
    21

    References
    ----------
    Scholand et al., RAGA sampling, DOI ``10.1002/mrm.30254``.

    See Also
    --------
    calc_golden_angles : the irrational increment RAGA approximates.
    """
    tiny_index, approximation_order = int(tiny_index), int(approximation_order)
    if n < 0:
        raise ValueError("n must be nonnegative")
    if tiny_index < 1 or approximation_order < 2:
        raise ValueError("tiny_index must be >= 1 and approximation_order >= 2")

    # The support is a finite, equidistant set of Fibonacci-many angles; the
    # order they are visited in is what stays golden-like.
    support_size = _generalized_fibonacci(approximation_order, tiny_index)
    step = _generalized_fibonacci(approximation_order - 1, 1)
    support = np.arange(support_size) * (2.0 * np.pi) / support_size
    visited = (np.arange(n, dtype=np.intp) * step) % support_size
    return support[visited]


def calc_tiny_golden_angles(n: int, *, index: int = 2) -> np.ndarray:
    """Return tiny-golden rotations with increment ``pi / (phi + index - 1)``.

    ``index=1`` gives the radial golden-angle increment; larger indices
    reduce the angular step.

    Parameters
    ----------
    n : int
        Number of angles.
    index : int, optional
        Tiny-golden index ``N >= 1`` (default 2).

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, accumulated modulo ``2 pi``.

    Raises
    ------
    ValueError
        If ``index`` is smaller than 1.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> np.rad2deg(pp.calc_tiny_golden_angles(3, index=2)).round(2)
    array([  0.  ,  68.75, 137.51])

    References
    ----------
    Wundrak et al., tiny golden angles, DOI ``10.1002/mrm.25831``.

    See Also
    --------
    calc_golden_angles : the ``index=1`` case.
    """
    index = int(index)
    if index < 1:
        raise ValueError("index must be >= 1")
    return _accumulated(n, np.pi / (_PHI + index - 1))


def calc_uniform_angles(n: int, *, span: float = 2.0 * np.pi) -> np.ndarray:
    """Return equally spaced rotations in the half-open interval [0, span).

    Parameters
    ----------
    n : int
        Number of angles.
    span : float, optional
        Angular range the spokes are spread across, in radians. ``pi`` for
        diametric spokes, ``2 * pi`` (the default) for full-turn arms.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, spaced by ``span / n``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> np.rad2deg(pp.calc_uniform_angles(4))
    array([  0.,  90., 180., 270.])

    A half-turn span spaces diametric spokes without covering a direction
    twice:

    >>> np.rad2deg(pp.calc_uniform_angles(4, span=np.pi))
    array([  0.,  45.,  90., 135.])

    See Also
    --------
    calc_golden_angles : uniform in any temporal window instead.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return _accumulated(n, 0.0 if n == 0 else float(span) / n)


def calc_projection_shell(n_views: int, n_shots: int = 1, *, scheme: str = "spiral"):
    """Return a pole-to-pole base shell and z-axis rotations for each shot.

    Consecutive base directions have equal angular separation. Rotated shells
    share both poles; each nonpolar ring has one spoke per shot.

    Parameters
    ----------
    n_views : int
        Spokes in the base shell, at least three.
    n_shots : int, optional
        Rotated replays of that shell.
    scheme : {'spiral', 'meridian'}, optional
        ``'spiral'`` winds pole to pole across equal-area rings, so the shell
        alone is already near-uniform. ``'meridian'`` is a half great circle in
        the x-z plane at equal polar steps, which is simpler and oversamples
        the poles.

    Returns
    -------
    directions : numpy.ndarray
        Unit spoke directions of the base shell, shape ``(n_views, 3)``.
    rotations : numpy.ndarray
        Rotation matrices, shape ``(n_shots, 3, 3)``, each turning the whole
        shell to where that shot samples.

    Raises
    ------
    ValueError
        If a count is out of range or ``scheme`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> directions, rotations = pp.calc_projection_shell(32, n_shots=13)
    >>> directions.shape, rotations.shape
    ((32, 3), (13, 3, 3))

    The shell runs pole to pole, so the shot rotations leave its ends alone:

    >>> bool(np.allclose(directions[[0, -1]], [[0, 0, 1], [0, 0, -1]]))
    True

    Consecutive views are exactly one step apart, so every turn between them
    is the same slew:

    >>> steps = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
    >>> bool(np.ptp(steps) < 1e-9)
    True

    See Also
    --------
    calc_golden_angles : in-plane spoke angles, one per shot, for 2D radial.
    """
    n_views, n_shots = int(n_views), int(n_shots)
    if n_views < 3:
        raise ValueError("a pole-to-pole shell needs at least three views")
    if n_shots < 1:
        raise ValueError("n_shots must be positive")

    if scheme == "spiral":
        directions = _constant_step_spiral(n_views)
    elif scheme == "meridian":
        polar = np.linspace(0.0, np.pi, n_views)
        directions = np.column_stack((np.sin(polar), np.zeros(n_views), np.cos(polar)))
    else:
        raise ValueError(f"scheme must be 'spiral' or 'meridian', got {scheme!r}")
    return directions, _turns_about_z(
        2.0 * np.pi * np.arange(n_shots, dtype=float) / n_shots
    )


def _constant_step_spiral(n_views: int) -> np.ndarray:
    """Place equal-angle neighbours on equal-area polar rings.

    The polar gap next to a pole fixes the angular step. Solve each azimuth
    increment from the spherical cosine relation.
    """
    height = np.linspace(1.0, -1.0, n_views)
    polar = np.arccos(np.clip(height, -1.0, 1.0))
    step = float(np.max(np.diff(polar)))

    numerator = math.cos(step) - np.cos(polar[:-1]) * np.cos(polar[1:])
    denominator = np.sin(polar[:-1]) * np.sin(polar[1:])
    increment = np.zeros(n_views - 1)
    turning = denominator > 1e-12
    increment[turning] = np.arccos(
        np.clip(numerator[turning] / denominator[turning], -1.0, 1.0)
    )
    azimuth = np.concatenate(([0.0], np.cumsum(increment)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), height))


def _turns_about_z(angles: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(
        "z", np.asarray(angles, dtype=float).reshape(-1, 1)
    ).as_matrix()
