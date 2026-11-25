"""Non-Cartesian orientations: in-plane rotation angles and 3D projection directions.

The angle routines return one in-plane rotation angle in radians per shot
(spoke, interleaf or blade); :func:`calc_projection_shell` returns 3D unit
directions and rotation matrices. The scan loop applies them as rotations of
a readout trajectory.
"""

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

#: The golden ratio. Its irrationality keeps any window of consecutive
#: golden-angle spokes near-uniformly distributed.
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
    full_circle : bool, default=False
        ``False`` selects the ``pi``-periodic radial/blade increment
        ``pi / phi``; ``True`` selects the full-turn spiral increment
        ``2 * pi / phi**2``.

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
    tiny_index : int, default=1
        Tiny-golden index the rational approximation is built from.
    approximation_order : int, default=13
        Fibonacci order; sets the size of the angular support.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, drawn from a finite equidistant support.

    Raises
    ------
    ValueError
        If ``n`` is negative, ``tiny_index`` is below 1 or
        ``approximation_order`` is below 2.

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

    # The support is a finite, equidistant set of Fibonacci-many angles,
    # visited in a golden-angle-like order.
    support_size = _generalized_fibonacci(approximation_order, tiny_index)
    step = _generalized_fibonacci(approximation_order - 1, 1)
    support = np.arange(support_size) * (2.0 * np.pi) / support_size
    visited = (np.arange(n, dtype=np.intp) * step) % support_size
    return support[visited]


def calc_raga_size(n: int, *, tiny_index: int = 1) -> int:
    """Return the RAGA support size nearest ``n``.

    The support sizes are the generalized Fibonacci numbers of ``tiny_index``,
    ``1, N, N + 1, 2N + 1, ...``; on one of them the RAGA order visits every
    angle once and its first pass is the (tiny) golden-angle order.

    Parameters
    ----------
    n : int
        Wanted number of angles, at least 1.
    tiny_index : int, default=1
        Tiny-golden index ``N``, at least 1.

    Returns
    -------
    int
        The support size closest to ``n`` in ratio.

    Raises
    ------
    ValueError
        If ``n`` or ``tiny_index`` is below 1.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> pp.calc_raga_size(198), pp.calc_raga_size(198, tiny_index=5)
    (233, 191)
    """
    n, tiny_index = int(n), int(tiny_index)
    if n < 1 or tiny_index < 1:
        raise ValueError("n and tiny_index must be >= 1")
    order = 1
    while _generalized_fibonacci(order + 1, tiny_index) < n:
        order += 1
    below = _generalized_fibonacci(order, tiny_index)
    above = _generalized_fibonacci(order + 1, tiny_index)
    return below if n * n <= below * above else above


def _raga_order(n: int, tiny_index: int) -> int:
    order = 1
    while _generalized_fibonacci(order, tiny_index) < n:
        order += 1
    if order < 2 or _generalized_fibonacci(order, tiny_index) != n:
        raise ValueError(
            f"a RAGA order needs n_shots to be a generalized Fibonacci number "
            f"of tiny_index={tiny_index}; the nearest to {n} is "
            f"{calc_raga_size(n, tiny_index=tiny_index)}"
        )
    return order


def calc_tiny_golden_angles(n: int, *, index: int = 2) -> np.ndarray:
    """Return tiny-golden rotations with increment ``pi / (phi + index - 1)``.

    ``index=1`` gives the radial golden-angle increment; larger indices
    reduce the angular step.

    Parameters
    ----------
    n : int
        Number of angles.
    index : int, default=2
        Tiny-golden index, at least 1.

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
    span : float, default=2.0 * np.pi
        Angular range (rad) the angles are spread across: ``pi`` for
        diametric spokes, ``2 * pi`` for full-turn arms.

    Returns
    -------
    numpy.ndarray
        Angles (rad), length ``n``, spaced by ``span / n``.

    Raises
    ------
    ValueError
        If ``n`` is negative.

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


def calc_projection_shell(
    n_views: int,
    n_shots: int = 1,
    *,
    scheme: str = "spiral",
    ordering: str = "sequential",
    tiny_index: int = 1,
):
    """Return a pole-to-pole base shell and z-axis rotations for each shot.

    Rotated shells share both poles; each nonpolar ring has one spoke per
    shot.

    Parameters
    ----------
    n_views : int
        Spokes in the base shell, at least three.
    n_shots : int, default=1
        Number of rotated replays of the base shell, at least one.
    scheme : {'spiral', 'quiet', 'meridian'}, default='spiral'
        ``'spiral'`` winds pole to pole across equal-area rings, so the shell
        alone is already near-uniform. ``'quiet'`` keeps the spiral's rings and
        the directions the shots sample on them, but moves each ring's spoke by
        whole shot turns to the one nearest its predecessor: the shell becomes
        a near-meridian whose steps are about one ring gap, instead of a
        stride across many rings' worth of azimuth. ``'meridian'`` is a half
        great circle in the x-z plane at equal polar steps, which is simpler
        and oversamples the poles.
    ordering : {'sequential', 'golden', 'raga'}, default='sequential'
        Order of the shots. ``'sequential'`` turns each shot ``2*pi/n_shots``
        past the one before. ``'golden'`` plays the same set of turns with
        shot ``j`` on the grid turn ranked as ``frac(j / (phi + tiny_index -
        1))``, so the first shots, or any run of consecutive ones, spread
        near-uniformly in azimuth. ``'raga'`` is the rational approximation
        of the same golden step, as in :func:`calc_raga_angles`: ``n_shots``
        must be a generalized Fibonacci number of ``tiny_index`` (see
        :func:`calc_raga_size`), the grid is visited in steps of the preceding
        Fibonacci number, and the order is cyclic, so a run of consecutive
        shots that wraps past the last one into a repetition of the set is as
        uniform as any other. On such an ``n_shots`` the first pass is the
        ``'golden'`` order.
    tiny_index : int, default=1
        Tiny-golden index ``N`` of the golden step: 1 is the golden ratio,
        larger indices turn consecutive shots by less.

    Returns
    -------
    directions : numpy.ndarray
        Unit spoke directions of the base shell, shape ``(n_views, 3)``.
    rotations : numpy.ndarray
        Rotation matrices, shape ``(n_shots, 3, 3)``, in play order:
        rotations about the z axis by ``2*pi*k/n_shots``, each ``k`` in
        ``0..n_shots-1`` once. Each maps the whole base shell to the
        directions that shot samples.

    Raises
    ------
    ValueError
        If ``n_views`` is below 3, ``n_shots`` is below 1, ``scheme`` or
        ``ordering`` is unknown, or ``ordering`` is ``'raga'`` and ``n_shots``
        is not a generalized Fibonacci number of ``tiny_index``.

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

    Consecutive ``'spiral'`` views are exactly one step apart, so every turn
    between them is the same slew:

    >>> steps = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
    >>> bool(np.ptp(steps) < 1e-9)
    True

    ``'quiet'`` samples the same sphere with the shell's steps a ring gap long:

    >>> quiet, _ = pp.calc_projection_shell(32, n_shots=13, scheme="quiet")
    >>> quiet_steps = np.arccos(np.clip(np.sum(quiet[:-1] * quiet[1:], axis=1), -1, 1))
    >>> bool(np.median(quiet_steps) < np.median(steps) / 2)
    True

    Under ``'golden'`` the first quarter of the shots leaves no azimuth gap
    wider than twice their mean spacing:

    >>> _, golden = pp.calc_projection_shell(32, n_shots=40, ordering="golden")
    >>> azimuth = np.sort(np.arctan2(golden[:10, 1, 0], golden[:10, 0, 0]))
    >>> gaps = np.diff(np.append(azimuth, azimuth[0] + 2 * np.pi))
    >>> bool(gaps.max() < 2 * 2 * np.pi / 10)
    True

    See Also
    --------
    calc_golden_angles : in-plane spoke angles, one per shot, for 2D radial.
    """
    n_views, n_shots = int(n_views), int(n_shots)
    if ordering not in ("sequential", "golden", "raga"):
        raise ValueError(
            f"ordering must be 'sequential', 'golden' or 'raga', got {ordering!r}"
        )
    if int(tiny_index) < 1:
        raise ValueError("tiny_index must be >= 1")
    if n_views < 3:
        raise ValueError("a pole-to-pole shell needs at least three views")
    if n_shots < 1:
        raise ValueError("n_shots must be positive")

    if scheme in ("spiral", "quiet"):
        directions = _constant_step_spiral(
            n_views, 2.0 * np.pi / n_shots if scheme == "quiet" else None
        )
    elif scheme == "meridian":
        polar = np.linspace(0.0, np.pi, n_views)
        directions = np.column_stack((np.sin(polar), np.zeros(n_views), np.cos(polar)))
    else:
        raise ValueError(
            f"scheme must be 'spiral', 'quiet' or 'meridian', got {scheme!r}"
        )
    ratio = _PHI + int(tiny_index) - 1
    slots = np.arange(n_shots, dtype=float)
    if ordering == "golden":
        slots = np.argsort(np.argsort(np.mod(slots / ratio, 1.0), kind="stable"))
    elif ordering == "raga":
        order = _raga_order(n_shots, int(tiny_index))
        return directions, _turns_about_z(
            calc_raga_angles(n_shots, tiny_index=tiny_index, approximation_order=order)
        )
    return directions, _turns_about_z(2.0 * np.pi * slots / n_shots)


def _constant_step_spiral(n_views: int, shot_turn: float | None = None) -> np.ndarray:
    """Place equal-angle neighbours on equal-area polar rings.

    The polar gap next to a pole fixes the angular step. Solve each azimuth
    increment from the spherical cosine relation. With ``shot_turn``, each
    increment is reduced to the nearest one modulo the shot turn, which moves a
    spoke onto another shot's copy of it and leaves the shots' union alone.
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
    if shot_turn is not None:
        increment = (
            np.remainder(increment + 0.5 * shot_turn, shot_turn) - 0.5 * shot_turn
        )
    azimuth = np.concatenate(([0.0], np.cumsum(increment)))
    radius = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), height))


def _turns_about_z(angles: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(
        "z", np.asarray(angles, dtype=float).reshape(-1, 1)
    ).as_matrix()
