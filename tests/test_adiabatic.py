"""Adiabatic pulses.

Hyperbolic-secant and WURST sweeps are compared sample by sample with
PyPulseq's. BIR-4 and GOIA-WURST are checked in a Bloch simulation across a
range of transmit fields: BIR-4 tips by its flip angle, GOIA-WURST inverts its
slice and nothing else.
"""

from __future__ import annotations

import numpy as np
import pypulseq
import pytest

import pypulseqpp as pp
from pypulseqpp import _events

SYSTEM = pp.Opts.default
THICKNESS = 10e-3
GOIA_BANDWIDTH = 10e3


def played(rf, scale=1.0, offsets_hz=(0.0,)):
    """Magnetisation after the pulse at each off-resonance, from +z."""
    signal = np.asarray(rf.signal) * scale
    dwell = float(rf.t[1] - rf.t[0])
    return pp.sim_bloch(signal, np.asarray(offsets_hz, dtype=float)[:, None], dwell)


@pytest.mark.parametrize("pulse_type", ["hypsec", "wurst"])
def test_upstreams_sweeps_are_upstreams(pulse_type):
    """Upstream's factory builds them; the compiled event holds the samples to
    within rounding, hence ``rtol=1e-12`` against upstream."""
    ours = pp.make_adiabatic_pulse(pulse_type, duration=8e-3, system=SYSTEM)
    compiled = _events.make_adiabatic_pulse(pulse_type, duration=8e-3, system=SYSTEM)
    theirs = pypulseq.make_adiabatic_pulse(pulse_type, duration=8e-3, system=SYSTEM)
    assert np.array_equal(np.asarray(ours.signal), np.asarray(compiled.signal))
    assert np.allclose(np.asarray(ours.signal), theirs.signal, rtol=1e-12, atol=0)
    assert ours.delay == theirs.delay


def test_an_unknown_adiabatic_pulse_is_refused():
    with pytest.raises(ValueError, match="pulse_type must be one of"):
        pp.make_adiabatic_pulse("bir1")


# %% BIR-4


def bir4(flip, **kwargs):
    arguments = {"duration": 6e-3, "bandwidth": 20e3, "system": SYSTEM}
    return pp.make_adiabatic_pulse("bir4", flip_angle=flip, **(arguments | kwargs))


@pytest.mark.parametrize("flip", [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
def test_a_bir4_pulse_tips_by_its_flip_angle_whatever_the_b1(flip):
    rf = bir4(flip)
    for scale in (1.0, 1.5, 2.0, 3.0):
        assert played(rf, scale)[0, 2] == pytest.approx(np.cos(flip), abs=0.05)


def test_a_bir4_pulse_tolerates_off_resonance():
    rf = bir4(np.pi / 2)
    assert np.allclose(played(rf, 1.5, (-100.0, 0.0, 100.0))[:, 2], 0.0, atol=0.05)


def test_a_bir4_pulse_is_scaled_by_the_root_of_its_adiabaticity():
    gentle = np.abs(np.asarray(bir4(np.pi / 2, adiabaticity=4).signal)).max()
    strict = np.abs(np.asarray(bir4(np.pi / 2, adiabaticity=16).signal)).max()
    assert strict / gentle == pytest.approx(2.0)


def test_a_bir4_pulse_selects_no_slice():
    with pytest.raises(ValueError, match="no slice"):
        pp.make_adiabatic_pulse("bir4", return_gz=True, slice_thickness=5e-3)


# %% GOIA-WURST


def goia(**kwargs):
    arguments = {
        "duration": 5e-3,
        "bandwidth": GOIA_BANDWIDTH,
        "slice_thickness": THICKNESS,
        "return_gz": True,
        "system": SYSTEM,
    }
    return pp.make_adiabatic_pulse("goia_wurst", **(arguments | kwargs))


def gradient_at(gz, times):
    waveform = np.asarray(gz.waveform)
    centres = gz.delay + (np.arange(waveform.size) + 0.5) * SYSTEM.grad_raster_time
    return np.interp(times, centres, waveform, left=0.0, right=0.0)


def inversion(rf, gz, positions, scale=1.0):
    """Mz after the pulse across ``positions``, the gradient played with it."""
    dwell = float(rf.t[1] - rf.t[0])
    field = gradient_at(gz, rf.delay + np.asarray(rf.t))
    offsets = np.asarray(positions)[:, None] * field[None, :]
    return pp.sim_bloch(np.asarray(rf.signal) * scale, offsets, dwell)[:, 2]


def test_a_goia_wurst_pulse_inverts_its_slice_and_leaves_the_rest():
    rf, gz, _ = goia()
    inside = np.array([-0.3, 0.0, 0.3]) * THICKNESS
    outside = np.array([-1.0, -0.8, 0.8, 1.0]) * THICKNESS
    assert np.all(inversion(rf, gz, inside) < -0.9)
    assert np.all(inversion(rf, gz, outside) > 0.9)


def test_a_goia_wurst_inversion_does_not_follow_b1():
    rf, gz, _ = goia()
    inside = np.array([-0.3, 0.0, 0.3]) * THICKNESS
    for scale in (1.0, 1.5, 2.5):
        assert np.all(inversion(rf, gz, inside, scale) < -0.9)


def test_the_goia_gradient_dips_mid_pulse_by_its_modulation():
    rf, gz, _ = goia(gradient_modulation=0.8)
    peak = GOIA_BANDWIDTH / THICKNESS
    during = gradient_at(gz, rf.delay + np.asarray(rf.t))
    assert during.min() == pytest.approx(0.2 * peak, rel=0.05)
    assert during.max() == pytest.approx(peak, rel=0.02)


def test_the_goia_pulse_starts_where_its_gradient_reaches_the_plateau():
    rf, gz, _ = goia()
    ramp_end = gradient_at(gz, np.array([rf.delay]))[0]
    assert ramp_end == pytest.approx(GOIA_BANDWIDTH / THICKNESS, rel=0.1)


def test_a_goia_wurst_pulse_needs_its_gradient():
    with pytest.raises(ValueError, match="designed with its gradient"):
        pp.make_adiabatic_pulse("goia_wurst", duration=5e-3)
