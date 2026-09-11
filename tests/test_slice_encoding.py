"""gSlider, Hadamard and PINS pulses, held to their profiles in a Bloch simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import hadamard

import pypulseqpp as pp

SYSTEM = pp.Opts.default
DURATION = 4e-3
TBW = 12.0


def transverse(rf, offsets_hz, fields=None):
    """Mxy after the pulse, from +z, at each off-resonance (Hz)."""
    dwell = float(rf.t[1] - rf.t[0])
    offsets = np.asarray(offsets_hz, dtype=float)[:, None]
    if fields is not None:
        offsets = offsets * fields[None, :]
    m = pp.sim_bloch(np.asarray(rf.signal), offsets, dwell)
    return m[:, 0] + 1j * m[:, 1]


def subslice_centres(count):
    """Sub-slice centres in Hz: lowest frequency is the most negative position."""
    return (np.arange(count) + 0.5 - count / 2) / count * TBW / DURATION


# %% gSlider


@pytest.mark.parametrize("count, subslice", [(5, 0), (5, 1), (5, 2), (5, 4), (4, 1)])
def test_a_gslider_pulse_puts_its_phase_on_one_subslice(count, subslice):
    """Sub-slices barely wider than their transitions leak phase: an interior
    sub-slice's neighbours turn by up to a tenth of pi at TBW / count ~ 1.2 dinf."""
    arguments = {"duration": DURATION, "time_bw_product": TBW, "system": SYSTEM}
    encoded = pp.make_gslider_pulse(np.pi / 2, count, subslice, **arguments)
    plain = pp.make_gslider_pulse(
        np.pi / 2, count, subslice, subslice_phase=0.0, **arguments
    )
    centres = subslice_centres(count)
    reference = transverse(plain, centres)
    turned = np.angle(transverse(encoded, centres) / reference)
    others = np.delete(turned, subslice)
    assert np.all(np.abs(reference) > 0.9)
    assert abs(abs(turned[subslice]) - np.pi) < 0.1 * np.pi
    assert np.all(np.abs(others) < 0.12 * np.pi)


def test_a_gslider_pulse_leaves_the_outside_of_the_slab_alone():
    rf = pp.make_gslider_pulse(
        np.pi / 2, 5, 1, duration=DURATION, time_bw_product=TBW, system=SYSTEM
    )
    outside = np.array([-1.5, 1.5]) * TBW / DURATION
    assert np.all(np.abs(transverse(rf, outside)) < 0.05)


def test_a_subslice_outside_the_slab_is_refused():
    with pytest.raises(ValueError, match="subslice must lie"):
        pp.make_gslider_pulse(np.pi / 2, 5, 5, system=SYSTEM)


def test_subslices_narrower_than_their_transitions_are_refused():
    with pytest.raises(ValueError, match="narrower than their transitions"):
        pp.make_gslider_pulse(np.pi / 2, 8, 0, time_bw_product=8.0, system=SYSTEM)


def test_a_slab_too_wide_for_its_samples_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        pp.make_gslider_pulse(
            np.pi / 2, 5, 0, duration=40e-6, time_bw_product=TBW, system=SYSTEM
        )


# %% Hadamard


@pytest.mark.parametrize("row", range(4))
def test_a_hadamard_pulse_signs_the_subslices_by_its_row(row):
    arguments = {"duration": DURATION, "time_bw_product": TBW, "system": SYSTEM}
    encoded = pp.make_hadamard_pulse(np.pi / 2, 4, row, **arguments)
    plain = pp.make_hadamard_pulse(np.pi / 2, 4, 0, **arguments)
    centres = subslice_centres(4)
    ratio = transverse(encoded, centres) / transverse(plain, centres)
    assert np.allclose(ratio, hadamard(4)[row], atol=0.15)


def test_a_hadamard_order_that_is_not_a_power_of_two_is_refused():
    with pytest.raises(ValueError, match="power of two"):
        pp.make_hadamard_pulse(np.pi / 2, 3, 1, system=SYSTEM)


# %% PINS

THICKNESS = 2e-3
SEPARATION = 20e-3


def pins(**kwargs):
    return pp.make_pins_pulse(
        np.deg2rad(30), THICKNESS, SEPARATION, system=SYSTEM, **kwargs
    )


def excited(rf, gz, positions):
    """Mxy across ``positions`` (m), the blips played with the pulse."""
    times = rf.delay + np.asarray(rf.t)
    fields = np.interp(times, gz.delay + np.asarray(gz.tt), np.asarray(gz.waveform))
    return transverse(rf, positions, fields)


def test_a_pins_pulse_excites_a_slice_every_separation():
    rf, gz, _ = pins()
    on = np.array([-SEPARATION, 0.0, SEPARATION, 2 * SEPARATION])
    between = np.array([-0.5, 0.5, 1.5]) * SEPARATION
    assert np.allclose(np.abs(excited(rf, gz, on)), np.sin(np.deg2rad(30)), atol=0.05)
    assert np.all(np.abs(excited(rf, gz, between)) < 0.05)


def test_a_pins_slice_is_as_thick_as_asked():
    rf, gz, _ = pins()
    inside = np.array([-0.25, 0.25]) * THICKNESS
    outside = np.array([-1.0, 1.0]) * THICKNESS
    assert np.all(np.abs(excited(rf, gz, inside)) > 0.45)
    assert np.all(np.abs(excited(rf, gz, outside)) < 0.05)


def test_pins_subpulses_stay_within_the_b1_limit():
    rf, _, _ = pins(max_b1=10e-6)
    assert np.abs(np.asarray(rf.signal)).max() <= 10e-6 * SYSTEM.gamma * (1 + 1e-9)


def test_the_pins_rephaser_takes_back_the_blips_after_the_centre():
    """A linear-phase envelope is centred on the middle blip."""
    *_, gzr = pins()
    count = round(4.0 * SEPARATION / THICKNESS)
    assert gzr.area == pytest.approx(-(count - 1) / 2 / SEPARATION, rel=1e-6)


def test_pins_needs_enough_subpulses():
    with pytest.raises(ValueError, match="at least 8 subpulses"):
        pp.make_pins_pulse(np.pi / 2, 5e-3, 6e-3, system=SYSTEM)
