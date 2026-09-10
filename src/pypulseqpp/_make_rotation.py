"""Pulseq rotation extension events using scalar-first quaternions."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = ["make_rotation"]

_TWO_PI = 2.0 * math.pi


def _from_axis_and_angle(axis, angle: float) -> np.ndarray:
    """Return a scalar-first quaternion for an axis-angle rotation in radians."""
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
    """Compose ``Rz(phi) @ Ry(theta)``: apply the y rotation first."""
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
    """Return scalar-first quaternions using SciPy's matrix orthogonalisation."""
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
    """Create a rotation extension event for a block's gradients.

    Parameters
    ----------
    *args
        Accepted forms (angles in radians):

        - ``(rotation,)`` : a SciPy rotation object, retained by reference.
        - ``(phi,)`` : rotation about z.
        - ``(phi, theta)`` : ``Rz(phi) @ Ry(theta)``; y is applied first.
        - ``(axis, angle)`` : rotation about a nonzero three-vector.
        - ``(quaternion,)`` : four scalar-first components, normalised here.
        - ``(matrix,)`` : a 3-by-3 rotation matrix.
        - ``(matrices,)`` : an N-by-3-by-3 stack of rotation matrices.

    Returns
    -------
    types.SimpleNamespace or list of types.SimpleNamespace
        Event with ``type == "rot3D"``, or one event per stacked matrix.

    Raises
    ------
    ValueError
        For an unrecognised form, zero axis or quaternion, or an angle outside
        its accepted range: ``phi`` in ``[-pi, 2*pi)``, ``theta`` and
        axis-angle rotations in ``[-pi, pi]``.

    Notes
    -----
    Registration caches quaternions by rotation-object identity. Reuse the
    same object for blocks with the same orientation.
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
