"""Per gradient event: the steepest slew rate and the integrals of the squared gradient and slew rate."""

import math

import numpy as np
import pytest

import pypulseqpp as pp

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def dense(seq, axis=0):
    """Peak slew, energy and slew energy of the drawn corners, on a nanosecond grid."""
    times, values = seq.waveforms_and_times()[0][axis]
    grid = np.arange(times[0], times[-1], 1e-9)
    g = np.interp(grid, times, values)
    slew = np.diff(g) / 1e-9
    return np.abs(slew).max(), np.trapezoid(g**2, grid), np.sum(slew**2) * 1e-9


def test_a_trapezoid_is_measured_in_closed_form(system):
    seq = pp.Sequence(system)
    gx = pp.make_trapezoid(
        "x",
        amplitude=1e5,
        rise_time=1e-4,
        flat_time=1e-3,
        fall_time=2e-4,
        system=system,
    )
    seq.add_block(gx)

    stats = seq.gradient_statistics()

    a, rise, flat, fall = gx.amplitude, gx.rise_time, gx.flat_time, gx.fall_time
    assert stats.peak_slew[0] == pytest.approx(a / rise)
    assert stats.energy[0] == pytest.approx(a**2 * (flat + (rise + fall) / 3))
    assert stats.slew_energy[0] == pytest.approx(a**2 * (1 / rise + 1 / fall))


def test_a_triangle_is_measured_in_closed_form(system):
    seq = pp.Sequence(system)
    gy = pp.make_extended_trapezoid(
        "y", amplitudes=[0, 2e4, 0], times=[0, 2e-4, 6e-4], system=system
    )
    seq.add_block(gy)

    stats = seq.gradient_statistics()

    assert stats.peak_slew[0] == pytest.approx(2e4 / 2e-4)
    assert stats.energy[0] == pytest.approx(2e4**2 * 6e-4 / 3)
    assert stats.slew_energy[0] == pytest.approx(2e4**2 * (1 / 2e-4 + 1 / 4e-4))


@pytest.mark.parametrize("oversampling", [False, True], ids=["raster", "half-raster"])
def test_an_arbitrary_gradient_is_measured_on_the_waveform_it_plays(
    system, oversampling
):
    raster = system.grad_raster_time
    count = 39 if oversampling else 20
    step = raster / 2 if oversampling else raster
    times = (np.arange(count) + (1.0 if oversampling else 0.5)) * step
    waveform = 3e4 * np.sin(2 * np.pi * times / (count * step)) ** 2
    seq = pp.Sequence(system)
    seq.add_block(
        pp.make_arbitrary_grad(
            "x", waveform, first=0, last=0, oversampling=oversampling, system=system
        )
    )

    stats = seq.gradient_statistics()

    peak, energy, slew_energy = dense(seq)
    assert stats.energy[0] > 0
    assert stats.peak_slew[0] == pytest.approx(peak, rel=1e-4)
    assert stats.energy[0] == pytest.approx(energy, rel=1e-4)
    assert stats.slew_energy[0] == pytest.approx(slew_energy, rel=1e-4)


def test_entry_i_is_gradient_id_i_plus_one_across_both_kinds(system):
    seq = pp.Sequence(system)
    trapezoid = pp.make_trapezoid(
        "x", amplitude=1e4, rise_time=1e-4, flat_time=1e-3, system=system
    )
    arbitrary = pp.make_extended_trapezoid(
        "y", amplitudes=[0, 3e4, 0], times=[0, 3e-4, 6e-4], system=system
    )
    steeper = pp.make_trapezoid(
        "z", amplitude=4e4, rise_time=1e-4, flat_time=1e-3, system=system
    )
    seq.add_block(trapezoid)
    seq.add_block(arbitrary)
    seq.add_block(steeper)
    ids = {
        "x": int(seq.block_events[1][2]),
        "y": int(seq.block_events[2][3]),
        "z": int(seq.block_events[3][4]),
    }

    peak = seq.gradient_statistics().peak_slew

    assert peak.shape == (3,)
    assert peak[ids["x"] - 1] == pytest.approx(1e8)
    assert peak[ids["y"] - 1] == pytest.approx(1e8)
    assert peak[ids["z"] - 1] == pytest.approx(4e8)


def test_a_blocks_rotation_does_not_enter(system):
    gx = pp.make_trapezoid(
        "x", amplitude=1e5, rise_time=1e-4, flat_time=1e-3, system=system
    )
    plain = pp.Sequence(system)
    plain.add_block(gx)
    turned = pp.Sequence(system)
    turned.add_block(gx, pp.make_rotation(Rotation.from_euler("z", 30, degrees=True)))

    a, b = plain.gradient_statistics(), turned.gradient_statistics()

    np.testing.assert_array_equal(a.peak_slew, b.peak_slew)
    np.testing.assert_array_equal(a.energy, b.energy)
    np.testing.assert_array_equal(a.slew_energy, b.slew_energy)


def test_a_sequence_without_gradients_has_no_entries():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3))

    stats = seq.gradient_statistics()

    assert (
        stats.peak_slew.shape == stats.energy.shape == stats.slew_energy.shape == (0,)
    )
