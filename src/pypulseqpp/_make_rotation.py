"""Rotation extension event constructor.

Attaching one of these to a block rotates every gradient in it, without
redesigning any waveform. That is what makes non-Cartesian readouts cheap: one
base spoke or interleaf is designed, and each shot is the same waveform under
a different rotation.

Which means a scan builds one of these per shot, hundreds of thousands of
times, so what it costs to build one is what a design loop spends. A rotation
about z is a cosine and a sine; asking SciPy for the same thing costs forty
microseconds a shot, ninety times what the block itself costs. So the angle
forms are worked out here and SciPy is asked only where it is actually needed,
which is turning a matrix into a quaternion.

The event carries whichever it was given: a rotation object, kept so that
``add_block`` keys its quaternion cache on that object's identity and asks
once per orientation rather than once per block; or the four numbers
themselves, worked out here.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = ["make_rotation"]

_TWO_PI = 2.0 * math.pi


def _from_axis_and_angle(axis, angle: float) -> np.ndarray:
    """Return the quaternion turning by @p angle about @p axis, scalar first."""
    direction = np.asarray(axis, dtype=float).reshape(-1)
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        raise ValueError("rotation axis vector norm must be non-zero.")
    if abs(angle) > math.pi:
        raise ValueError(
            f"rotation angle phi ({angle:.2f}) is invalid. "
            "should be within [0,pi] radians"
        )
    half = math.sin(angle / 2.0) / length
    return np.array(
        [math.cos(angle / 2.0), *(half * direction)],
        dtype=float,
    )


def _from_polar(phi: float, theta: float) -> np.ndarray:
    """Rz(phi) then Ry(theta), which is the MATLAB toolbox's `qz * qy`."""
    if not -math.pi <= phi < _TWO_PI:
        raise ValueError(
            f"rotation angle phi ({phi:.2f}) is invalid. "
            "should be within [-pi,2*pi) radians"
        )
    if not -math.pi <= theta <= math.pi:
        raise ValueError(
            f"rotation angle theta ({theta:.2f}) is invalid. "
            "should be within [-pi,pi] radians"
        )
    cz, sz = math.cos(phi / 2.0), math.sin(phi / 2.0)
    cy, sy = math.cos(theta / 2.0), math.sin(theta / 2.0)
    # (cz, 0, 0, sz) * (cy, 0, sy, 0), written out.
    return np.array([cz * cy, -sz * sy, cz * sy, sz * cy], dtype=float)


def _from_matrices(matrices: np.ndarray):
    """One quaternion per 3x3 matrix, scalar first.

    SciPy is asked here and only here: turning a matrix into a quaternion
    means projecting it onto a rotation first, and its answer is the one the
    toolbox's answer is.
    """
    from scipy.spatial.transform import Rotation

    quaternions = np.atleast_2d(
        Rotation.from_matrix(np.ascontiguousarray(matrices, dtype=float)).as_quat()
    )
    return [np.array([q[3], q[0], q[1], q[2]], dtype=float) for q in quaternions]


def _event(quaternion) -> SimpleNamespace:
    made = SimpleNamespace()
    made.type = "rot3D"
    made.rot_quaternion = quaternion
    return made


def make_rotation(*args: Any) -> SimpleNamespace | list[SimpleNamespace]:
    """Create a rotation extension event.

    Parameters
    ----------
    *args
        One of the forms the reference toolbox takes, or a rotation object:

        - ``make_rotation(rotation)`` — anything exposing ``as_quat``, which
          is a :class:`scipy.spatial.transform.Rotation`. Kept as it is.
        - ``make_rotation(phi)`` — a turn about z.
        - ``make_rotation(phi, theta)`` — ``Rz(phi)`` then ``Ry(theta)``.
        - ``make_rotation(axis, angle)`` — a turn about a three-vector.
        - ``make_rotation(quaternion)`` — four numbers, scalar first,
          normalised here.
        - ``make_rotation(matrix)`` — a 3x3 rotation matrix.
        - ``make_rotation(matrices)`` — an Nx3x3 stack, giving a list of
          events, one per matrix.

    Returns
    -------
    types.SimpleNamespace or list
        Rotation extension event (``type == 'rot3D'``), or one per matrix
        when given a stack of them.

    Notes
    -----
    The angle ranges are the toolbox's: ``phi`` within ``[-pi, 2*pi)``,
    ``theta`` within ``[-pi, pi]``, and an axis-angle turn within ``[0, pi]``.

    Reusing one rotation *object* across blocks costs one ``as_quat`` call for
    all of them; the angle forms cost no SciPy call at all.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> pp.make_rotation(np.pi / 4).type
    'rot3D'

    Rotate a radial base spoke over a golden-angle plan::

        for angle in pp.calc_golden_angles(377):
            readout(seq, rotation=pp.make_rotation(angle))

    See Also
    --------
    calc_projection_shell : directions spread over the sphere, to rotate onto.
    """
    if not args:
        raise ValueError(
            "make_rotation() takes a rotation: an angle, an axis and an "
            "angle, a quaternion, a matrix, or a rotation object"
        )

    first = args[0]

    # An angle is a number, and a scan turning a base spoke per shot passes
    # one per shot: recognised before anything is asked of NumPy.
    if isinstance(first, (float, int, np.floating, np.integer)) and not isinstance(
        first, bool
    ):
        if len(args) > 2:
            raise ValueError(
                "make_rotation(phi) or make_rotation(phi, theta) expected."
            )
        theta = 0.0 if len(args) == 1 else float(args[1])
        return _event(_from_polar(float(first), theta))

    if hasattr(first, "as_quat"):
        if len(args) != 1:
            raise ValueError("make_rotation(rotation) takes exactly one argument.")
        return _event(first)

    given = np.asarray(first, dtype=float)
    flat = given.reshape(-1)

    if given.ndim == 3 and given.shape[1:] == (3, 3):
        if len(args) != 1:
            raise ValueError("make_rotation(matrices) takes exactly one argument.")
        return [_event(quaternion) for quaternion in _from_matrices(given)]

    if given.shape == (3, 3):
        if len(args) != 1:
            raise ValueError("make_rotation(matrix) takes exactly one argument.")
        return _event(_from_matrices(given[np.newaxis])[0])

    if flat.size == 1:
        if len(args) > 2:
            raise ValueError(
                "make_rotation(phi) or make_rotation(phi, theta) expected."
            )
        theta = 0.0 if len(args) == 1 else float(args[1])
        return _event(_from_polar(float(flat[0]), theta))

    if flat.size == 3:
        if len(args) != 2:
            raise ValueError("make_rotation(axis, angle) expected for an axis.")
        return _event(_from_axis_and_angle(flat, float(args[1])))

    if flat.size == 4:
        if len(args) != 1:
            raise ValueError("make_rotation(quaternion) takes exactly one argument.")
        norm = float(np.linalg.norm(flat))
        if norm == 0.0:
            raise ValueError("quaternion norm must be non-zero.")
        return _event(flat / norm)

    raise ValueError(
        "make_rotation() did not recognise its arguments: expected an angle, "
        "an angle and a polar angle, an axis and an angle, a quaternion, a "
        "3x3 matrix, or a stack of them"
    )
