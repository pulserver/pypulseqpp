"""Turning a block's gradients onto a rotated set of axes."""

from __future__ import annotations

__all__ = ["rotate_3d"]

import numbers as _numbers
from copy import deepcopy as _deepcopy

import numpy as _np
from pypulseq.add_gradients import add_gradients as _add_gradients
from pypulseq.scale_grad import scale_grad as _scale_grad
from scipy.spatial.transform import Rotation as _Rotation

from ._block_to_events import block_to_events as _block_to_events

_AXES = ("x", "y", "z")

#: How small a matrix entry has to be before the projection onto that axis is
#: no gradient at all, as a fraction of the strongest gradient given.
_NEGLIGIBLE = 1e-6


def _peak(grad) -> float:
    if grad.type == "trap":
        return abs(grad.amplitude)
    return float(_np.max(_np.abs(grad.waveform)))


def _matrix(rotation) -> _np.ndarray:
    given = _np.asarray(rotation, dtype=float)
    if given.shape == (3, 3):
        return given

    flat = given.ravel()
    if flat.size == 4:
        # Scalar-first, as Pulseq writes a quaternion; scipy reads them
        # scalar-last.
        return _Rotation.from_quat([flat[1], flat[2], flat[3], flat[0]]).as_matrix()

    if flat.size in (1, 2):
        phi = float(flat[0])
        if not -_np.pi <= phi < 2 * _np.pi:
            raise ValueError(
                f"rotation angle phi ({phi:.2f}) is outside [-pi, 2*pi) radians"
            )
        about_z = _Rotation.from_rotvec([0.0, 0.0, phi])
        if flat.size == 1:
            return about_z.as_matrix()
        theta = float(flat[1])
        if not -_np.pi <= theta <= _np.pi:
            raise ValueError(
                f"rotation angle theta ({theta:.2f}) is outside [-pi, pi] radians"
            )
        return (about_z * _Rotation.from_rotvec([0.0, theta, 0.0])).as_matrix()

    raise ValueError(
        "a rotation is a 3x3 matrix, a scalar-first quaternion, or one or two angles"
    )


def rotate_3d(rotation, *args, system=None) -> list:
    """Rotate and sum gradient projections onto the output axes.

    Non-gradient events are returned unchanged, before the rotated gradients.

    Parameters
    ----------
    rotation : array_like
        A 3x3 matrix, a scalar-first quaternion ``[w, x, y, z]``, one angle
        about z, or two angles read as ``Rz(phi) Ry(theta)``.
    *args : SimpleNamespace or list
        Block events, or one block. At most one gradient per axis.
    system : Opts, optional
        System limits, used when two projections are summed.

    Returns
    -------
    list
        The events that were not gradients, then the rotated gradients.

    Raises
    ------
    ValueError
        On a rotation that is none of the accepted forms, on a gradient
        outside x, y and z, or on two gradients on one axis.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> gx = pp.make_trapezoid("x", area=1000, duration=2e-3)

    A quarter turn about z carries a readout off x onto y, area and all:

    >>> about_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> (rotated,) = pp.rotate_3d(about_z, gx)
    >>> rotated.channel, round(float(rotated.area))
    ('y', 1000)

    The same turn, named as a quaternion and as an angle:

    >>> quarter = np.pi / 2
    >>> quaternion = [np.cos(quarter / 2), 0.0, 0.0, np.sin(quarter / 2)]
    >>> pp.rotate_3d(quaternion, gx)[0].channel
    'y'
    >>> pp.rotate_3d([quarter], gx)[0].channel
    'y'

    Anything that is not a gradient comes back untouched, and first:

    >>> adc = pp.make_adc(num_samples=64, duration=1e-3)
    >>> [event.type for event in pp.rotate_3d(about_z, gx, adc)]
    ['adc', 'trap']
    """
    matrix = _matrix(rotation)
    if not args:
        return []

    events = list(_block_to_events(*args))

    to_rotate: list = [None, None, None]
    bypass: list = []
    for event in events:
        if event is None:
            continue
        if isinstance(event, _numbers.Number) or getattr(event, "type", None) not in (
            "grad",
            "trap",
        ):
            bypass.append(event)
            continue
        if event.channel not in _AXES:
            raise ValueError(
                f"invalid gradient channel {event.channel!r}; expected x, y or z"
            )
        axis = _AXES.index(event.channel)
        if to_rotate[axis] is not None:
            raise ValueError(f"more than one gradient on axis {event.channel}")
        to_rotate[axis] = _deepcopy(event)

    strongest = max((_peak(g) for g in to_rotate if g is not None), default=0.0)
    keep_above = _NEGLIGIBLE * strongest

    rotated = []
    for out_axis in range(3):
        summed = None
        for in_axis in range(3):
            grad = to_rotate[in_axis]
            if grad is None or abs(matrix[out_axis, in_axis]) < _NEGLIGIBLE:
                continue
            projection = _scale_grad(grad, matrix[out_axis, in_axis])
            projection.channel = _AXES[out_axis]
            if summed is None:
                summed = projection
            elif system is None:
                summed = _add_gradients([summed, projection])
            else:
                summed = _add_gradients([summed, projection], system=system)
        if summed is not None and _peak(summed) >= keep_above:
            rotated.append(summed)

    return [*bypass, *rotated]
