"""Moving the field of view of a sequence that is already designed.

A shift is a phase, and the phase is ``dr . k``: how far the volume moved,
against where the trajectory stands. Everything here holds that one identity,
because it is what a shift *means* -- the phase every pulse is transmitted
with and every sample is acquired with has to come out as ``dr . k(t)``,
however the sequence chooses to split it between a frequency, a phase and a
profile.

That split is worth having. Under a gradient that does not move across an
event the whole shift is two numbers on the event's own row, so a Cartesian
readout carries no profile at all; only where the gradient moves is there
anything left over.

The shift is written in the logical frame, which is the frame the gradients
were designed in. ``dr . k`` does not change when both are turned, so it
needs to know nothing about the rotation the scanner applies.
"""

import math

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=200,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
        adc_raster_time=100e-9,
        block_duration_raster=10e-6,
    )


def trapezoid_k(gradient, when):
    """Where a trapezoid has taken the trajectory by ``when``, in 1/m."""
    rise, flat, fall = gradient.rise_time, gradient.flat_time, gradient.fall_time
    amplitude = gradient.amplitude
    when = np.asarray(when, dtype=float)
    swept = np.where(
        when <= rise,
        0.5 * amplitude * when * when / max(rise, 1e-30),
        0.5 * amplitude * rise
        + amplitude * np.clip(when - rise, 0.0, flat)
        + np.where(
            when > rise + flat,
            amplitude * (when - rise - flat)
            - 0.5 * amplitude * (when - rise - flat) ** 2 / max(fall, 1e-30),
            0.0,
        ),
    )
    return swept


def acquired_phase(adc):
    """The phase each sample is taken with, in turns, however it is stored."""
    when = adc.delay + adc.dwell * (np.arange(int(adc.num_samples)) + 0.5)
    profile = getattr(adc, "phase_modulation", None)
    residual = np.zeros(int(adc.num_samples))
    if profile is not None:
        held = np.asarray(profile).reshape(-1)
        residual[: held.size] = held[: residual.size]
    turns = (
        adc.phase_offset / (2 * np.pi)
        + adc.freq_offset * (when - adc.delay)
        + residual / (2 * np.pi)
    )
    return when, turns


def apart(one, other):
    """How far two phases in turns are, allowing for whole turns."""
    return float(
        np.abs(((np.asarray(one) - np.asarray(other)) + 0.5) % 1.0 - 0.5).max()
    )


# -- what a readout is acquired with ---------------------------------------


@pytest.mark.parametrize("shift", [0.011, -0.023, 0.0])
def test_a_readout_is_acquired_at_the_phase_the_shift_asks_for(system, shift):
    """Sampled across the ramps as well as the top: the gradient moves."""
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid(
        "x",
        amplitude=1.5e5,
        rise_time=6e-4,
        flat_time=1e-3,
        fall_time=6e-4,
        system=system,
    )
    sequence.add_block(gradient, pp.make_adc(64, duration=2.2e-3, system=system))

    _ext.apply_fov_shift(sequence._native, shift=(shift, 0.0, 0.0))

    when, acquired = acquired_phase(sequence.get_block(1).adc)
    assert apart(acquired, shift * trapezoid_k(gradient, when)) < 1e-12


def test_a_cartesian_readout_costs_two_numbers(system):
    """Nothing moves across its window, so there is no residual to store."""
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid(
        "x",
        amplitude=1.5e5,
        rise_time=6e-4,
        flat_time=2e-3,
        fall_time=6e-4,
        system=system,
    )
    sequence.add_block(
        gradient,
        pp.make_adc(64, duration=2e-3, delay=gradient.rise_time, system=system),
    )

    _ext.apply_fov_shift(sequence._native, shift=(0.011, 0.0, 0.0))

    adc = sequence.get_block(1).adc
    profile = getattr(adc, "phase_modulation", None)
    assert profile is None or np.asarray(profile).size == 0
    when, acquired = acquired_phase(adc)
    assert apart(acquired, 0.011 * trapezoid_k(gradient, when)) < 1e-12


# -- what a pulse is transmitted with --------------------------------------


def test_a_pulse_under_a_steady_gradient_is_two_numbers(system):
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", flat_area=5000, flat_time=2e-3, system=system)
    pulse = pp.make_block_pulse(
        math.pi / 6,
        duration=1e-3,
        delay=gradient.rise_time,
        system=system,
        use="excitation",
    )
    sequence.add_block(pulse, gradient)
    before = sequence._native.num_shapes()

    _ext.apply_fov_shift(sequence._native, shift=(0.01, 0.0, 0.0))

    assert sequence._native.num_shapes() == before
    assert sequence.get_block(1).rf.freq_offset == pytest.approx(
        0.01 * gradient.amplitude, rel=1e-12
    )


def test_a_pulse_under_a_moving_gradient_gains_a_profile(system):
    """A pulse on a ramp cannot be moved by two numbers."""
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid(
        "x",
        amplitude=2.0e5,
        rise_time=2e-3,
        flat_time=1e-3,
        fall_time=2e-3,
        system=system,
    )
    pulse = pp.make_sinc_pulse(
        math.pi / 6,
        duration=1.4e-3,
        slice_thickness=5e-3,
        delay=2e-4,
        system=system,
        use="excitation",
    )
    sequence.add_block(pulse, gradient)
    before = sequence._native.num_shapes()

    _ext.apply_fov_shift(sequence._native, shift=(0.011, 0.0, 0.0))

    assert sequence._native.num_shapes() == before + 1


# -- the frame it is written in --------------------------------------------


def test_a_turned_block_is_shifted_the_same_way(system):
    """``dr . k`` does not change when both are turned, so this need not."""
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    gradient = pp.make_trapezoid("x", flat_area=5000, flat_time=2e-3, system=system)
    adc = pp.make_adc(64, duration=2e-3, delay=gradient.rise_time, system=system)

    straight = pp.Sequence(system)
    straight.add_block(gradient, adc)
    turned = pp.Sequence(system)
    turned.add_block(gradient, adc, pp.make_rotation(Rotation.from_euler("z", 0.7)))

    for sequence in (straight, turned):
        _ext.apply_fov_shift(sequence._native, shift=(0.01, 0.0, 0.0))

    assert straight.get_block(1).adc.freq_offset == pytest.approx(
        turned.get_block(1).adc.freq_offset, rel=1e-12
    )


# -- what is carried from block to block -----------------------------------


def test_two_identical_repetitions_read_at_the_same_phase(system):
    """Which is what says the phase is not reset where the trajectory is.

    A readout is measured against the phase its own excitation was given, so
    the two are counted from the same place. Restart the count between them --
    at the excitation, where the trajectory does restart -- and two
    repetitions of the same thing come out different.
    """
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", flat_area=5000, flat_time=2e-3, system=system)
    pulse = pp.make_block_pulse(
        math.pi / 6,
        duration=1e-3,
        delay=gradient.rise_time,
        system=system,
        use="excitation",
    )
    adc = pp.make_adc(64, duration=2e-3, delay=gradient.rise_time, system=system)
    for _ in range(2):
        sequence.add_block(pulse, gradient)
        sequence.add_block(gradient, adc)

    _ext.apply_fov_shift(sequence._native, shift=(0.013, 0.0, 0.0))

    relative = [
        (
            sequence.get_block(readout).adc.phase_offset
            - sequence.get_block(excitation).rf.phase_offset
        )
        / (2 * np.pi)
        for excitation, readout in ((1, 2), (3, 4))
    ]

    assert apart(relative[0], relative[1]) < 1e-12


def test_a_shift_can_be_applied_in_chunks(system):
    """A scan too large to hold at once is moved a piece at a time."""

    def built():
        sequence = pp.Sequence(system)
        gradient = pp.make_trapezoid("x", flat_area=5000, flat_time=2e-3, system=system)
        adc = pp.make_adc(64, duration=2e-3, delay=gradient.rise_time, system=system)
        for _ in range(6):
            sequence.add_block(gradient, adc)
        return sequence

    whole, in_pieces = built(), built()
    _ext.apply_fov_shift(whole._native, shift=(0.01, 0.0, 0.0))

    carry = (0.0, 0.0, 0.0)
    for first in (1, 3, 5):
        carry = _ext.apply_fov_shift(
            in_pieces._native,
            shift=(0.01, 0.0, 0.0),
            first=first,
            last=first + 1,
            carry=carry,
        )

    for index in range(1, 7):
        assert in_pieces.get_block(index).adc.phase_offset == pytest.approx(
            whole.get_block(index).adc.phase_offset, abs=1e-9
        )


def test_no_shift_changes_nothing(system):
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", flat_area=5000, flat_time=2e-3, system=system)
    adc = pp.make_adc(64, duration=2e-3, delay=gradient.rise_time, system=system)
    sequence.add_block(gradient, adc)
    was = sequence.get_block(1).adc

    _ext.apply_fov_shift(sequence._native, shift=(0.0, 0.0, 0.0))

    now = sequence.get_block(1).adc
    assert now.freq_offset == pytest.approx(was.freq_offset)
    assert now.phase_offset == pytest.approx(was.phase_offset)
