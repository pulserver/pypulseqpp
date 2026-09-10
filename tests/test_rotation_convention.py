"""Physical gradients use R @ logical; trajectory and continuity checks must agree."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pypulseqpp as pp
from pypulseqpp import safety

#: General rotations: no axis aligned with a gradient channel, so R and R.T
#: give genuinely different vectors rather than a sign flip.
ROTATIONS = [
    Rotation.from_euler("xyz", [0.3, -0.7, 1.1]),
    Rotation.from_euler("xyz", [-1.2, 0.45, 0.9]),
    Rotation.from_euler("xyz", [2.0, 1.3, -0.6]),
]

AMPLITUDE = 1.0e5


def _constant(vector, duration=2e-4, system=None):
    """One extended trapezoid per axis, holding ``vector`` for ``duration``."""
    vector = np.asarray(vector, dtype=float)
    return [
        pp.make_extended_trapezoid(
            channel=axis,
            times=np.array([0.0, duration]),
            amplitudes=np.array([vector[index], vector[index]]),
            system=system,
        )
        for index, axis in enumerate(("x", "y", "z"))
    ]


def _ramp(start, stop, duration=1e-4, system=None):
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    return [
        pp.make_extended_trapezoid(
            channel=axis,
            times=np.array([0.0, duration]),
            amplitudes=np.array([start[index], stop[index]]),
            system=system,
        )
        for index, axis in enumerate(("x", "y", "z"))
    ]


@pytest.mark.parametrize("rotation", ROTATIONS, ids=range(len(ROTATIONS)))
def test_a_rotated_gradient_plays_along_the_rotated_direction(rotation):
    """``R @ logical`` is what the trajectory reports, as ``mr.rotate3D`` defines."""
    system = pp.Opts()
    logical = np.array([1.0, 0.0, 0.0])
    expected = rotation.as_matrix() @ logical

    gx = pp.make_trapezoid(
        channel="x",
        amplitude=AMPLITUDE,
        flat_time=4e-4,
        rise_time=1e-4,
        fall_time=1e-4,
        system=system,
    )
    adc = pp.make_adc(num_samples=64, duration=4e-4, delay=1e-4, system=system)
    seq = pp.Sequence(system)
    seq.add_block(gx, adc, pp.make_rotation(rotation))
    seq.add_block(pp.make_delay(1e-3))

    k = np.asarray(seq.calculate_kspace()[0])[:, -1]
    np.testing.assert_allclose(k / np.linalg.norm(k), expected, atol=1e-4)


@pytest.mark.parametrize("rotation", ROTATIONS, ids=range(len(ROTATIONS)))
def test_blocks_meeting_in_the_physical_frame_are_continuous(rotation):
    """Two blocks holding one physical vector do not step, whatever frame states it.

    The second block states the same physical gradient in its own rotated
    frame, so the amplifiers see no change at the boundary. A checker that
    resolved the rotation the wrong way round would see the difference
    between ``R @ v`` and ``R.T @ v`` and refuse a sequence that is in fact
    continuous -- which is how a rotation-encoded readout like ZTE, where
    every view hands its gradient to the next, gets rejected.
    """
    system = pp.Opts()
    matrix = rotation.as_matrix()
    physical = np.array([AMPLITUDE, 0.0, 0.0])
    logical = matrix.T @ physical
    event = pp.make_rotation(rotation)

    seq = pp.Sequence(system)
    seq.add_block(*_ramp(np.zeros(3), physical, system=system))
    seq.add_block(*_constant(physical, system=system))
    seq.add_block(*_constant(logical, system=system), event)
    seq.add_block(*_ramp(logical, np.zeros(3), system=system), event)

    # Whether two blocks meet is `check_grad_continuity`'s question; how fast
    # anything slews inside one is `check_max_slew`'s, and they were split.
    is_ok, report = safety.check_grad_continuity(seq, system)

    assert report.discontinuities == []
    assert is_ok
