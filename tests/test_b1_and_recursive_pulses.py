"""B1-selective, Bloch-Siegert and recursive SLR pulses, held to what they do in a Bloch simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import hadamard

import pypulseqpp as pp

SYSTEM = pp.Opts(rf_raster_time=4e-6)
AMPLITUDE = 50e-6


def transverse_at(rf, scales):
    """Mx + i My after the pulse, from +z, where B1 is ``scales`` of nominal."""
    signal = np.asarray(rf.signal)
    dwell = float(rf.t[1] - rf.t[0])
    out = []
    for scale in scales:
        m = pp.sim_bloch(scale * signal, np.zeros((1, 1)), dwell)[0]
        out.append(m[0] + 1j * m[1])
    return np.asarray(out)


# %% B1-selective


def test_a_b1_selective_pulse_excites_only_its_band_of_b1():
    rf = pp.make_b1_selective_pulse(
        np.pi / 4, AMPLITUDE, 1.0, 0.4, pulse_type="ex", system=SYSTEM
    )
    inside = np.abs(transverse_at(rf, [0.97, 1.0, 1.03]))
    outside = np.abs(transverse_at(rf, [0.2, 0.4, 1.6, 1.9]))
    assert np.allclose(inside, np.sin(np.pi / 4), atol=0.05)
    assert np.all(outside < 0.05)


def test_a_b1_selective_pulse_lasts_twice_its_time_bandwidth_over_the_width():
    rf = pp.make_b1_selective_pulse(np.pi / 4, AMPLITUDE, 1.0, 0.4, system=SYSTEM)
    width_hz = 0.4 * AMPLITUDE * SYSTEM.gamma
    assert float(rf.shape_dur) == pytest.approx(2 * 4.0 / width_hz, rel=0.02)


def test_a_passband_reaching_zero_b1_is_refused():
    with pytest.raises(ValueError, match="above zero B1"):
        pp.make_b1_selective_pulse(np.pi / 4, AMPLITUDE, 0.5, 1.2, system=SYSTEM)


def band_centres(count, centre, width):
    return centre + (np.arange(count) + 0.5 - count / 2) * width / count


@pytest.mark.parametrize("subslice", [0, 2])
def test_the_b1_gslider_pulse_puts_its_phase_on_one_sub_band(subslice):
    arguments = {"passband_center": 1.0, "passband_width": 0.8, "system": SYSTEM}
    encoded = pp.make_b1_gslider_pulse(np.pi / 2, AMPLITUDE, 4, subslice, **arguments)
    plain = pp.make_b1_gslider_pulse(
        np.pi / 2, AMPLITUDE, 4, subslice, subslice_phase=0.0, **arguments
    )
    centres = band_centres(4, 1.0, 0.8)
    turned = np.angle(transverse_at(encoded, centres) / transverse_at(plain, centres))
    assert abs(abs(turned[subslice]) - np.pi) < 0.15 * np.pi
    assert np.all(np.abs(np.delete(turned, subslice)) < 0.15 * np.pi)


@pytest.mark.parametrize("row", [1, 2, 3])
def test_the_b1_hadamard_pulse_signs_its_sub_bands_by_its_row(row):
    arguments = {"passband_center": 1.0, "passband_width": 1.0, "system": SYSTEM}
    encoded = pp.make_b1_hadamard_pulse(np.pi / 2, AMPLITUDE, 4, row, **arguments)
    plain = pp.make_b1_hadamard_pulse(np.pi / 2, AMPLITUDE, 4, 0, **arguments)
    centres = band_centres(4, 1.0, 1.0)
    ratio = transverse_at(encoded, centres) / transverse_at(plain, centres)
    # Each sub-band carries its row's sign, to within the transitions' leakage.
    assert np.all(np.abs(np.angle(ratio * hadamard(4)[row])) < 0.2 * np.pi)


# %% Bloch-Siegert


def phase_left(rf, scale):
    """Phase added to transverse magnetisation where B1 is ``scale`` of nominal."""
    signal = np.asarray(rf.signal)
    dwell = float(rf.t[1] - rf.t[0])
    m = pp.sim_bloch(
        scale * signal, np.zeros((1, 1)), dwell, initial=np.array([1.0, 0.0, 0.0])
    )[0]
    return float(np.angle(m[0] + 1j * m[1]))


def test_a_bloch_siegert_phase_grows_with_the_square_of_b1():
    rf = pp.make_bloch_siegert_pulse(10e-6, 4e-3, system=SYSTEM)
    assert phase_left(rf, 1.0) / phase_left(rf, 0.5) == pytest.approx(4.0, rel=0.05)


def test_a_bloch_siegert_pulse_leaves_the_magnetisation_along_z():
    rf = pp.make_bloch_siegert_pulse(10e-6, 4e-3, system=SYSTEM)
    signal = np.asarray(rf.signal)
    m = pp.sim_bloch(signal, np.zeros((1, 1)), float(rf.t[1] - rf.t[0]))[0]
    assert m[2] > 0.99


def test_the_other_side_of_resonance_reverses_the_phase():
    plus = pp.make_bloch_siegert_pulse(10e-6, 4e-3, system=SYSTEM)
    minus = pp.make_bloch_siegert_pulse(10e-6, 4e-3, frequency_sign=-1, system=SYSTEM)
    assert phase_left(minus, 1.0) == pytest.approx(-phase_left(plus, 1.0), rel=1e-6)


def test_a_sweep_that_would_reach_resonance_is_refused():
    with pytest.raises(ValueError, match="reaches resonance"):
        pp.make_bloch_siegert_pulse(10e-6, 4e-3, k=1.0, system=SYSTEM)


# %% recursive SLR


def train_profile(pulses, offsets_hz):
    """|Mxy| each pulse leaves, transverse magnetisation spoiled between them."""
    dwell = float(pulses[0].t[1] - pulses[0].t[0])
    state = np.tile([0.0, 0.0, 1.0], (len(offsets_hz), 1))
    left = []
    for rf in pulses:
        m = pp.sim_bloch(
            np.asarray(rf.signal), np.asarray(offsets_hz)[:, None], dwell, initial=state
        )
        left.append(np.hypot(m[:, 0], m[:, 1]))
        state = np.column_stack([np.zeros(len(m)), np.zeros(len(m)), m[:, 2]])
    return np.asarray(left)


def test_every_pulse_of_a_recursive_train_excites_the_same_magnetisation():
    """Hyperpolarised spins: each segment reads what the last left, evenly."""
    pulses = pp.make_recursive_slr_pulses(3, duration=4e-3, system=SYSTEM)
    left = train_profile(pulses, [0.0])[:, 0]
    assert np.allclose(left, 1 / np.sqrt(3), rtol=0.05)


def test_a_recursive_train_excites_nothing_outside_its_slice():
    pulses = pp.make_recursive_slr_pulses(3, duration=4e-3, system=SYSTEM)
    bandwidth = 4.0 / (4e-3 / 1.75)
    assert np.all(train_profile(pulses, [3 * bandwidth, -3 * bandwidth]) < 0.05)


def test_a_spin_echo_train_comes_with_a_refocusing_pulse_that_inverts():
    _, refocusing = pp.make_recursive_slr_pulses(
        3, duration=4e-3, spin_echo=True, system=SYSTEM
    )
    m = pp.sim_bloch(
        np.asarray(refocusing.signal),
        np.zeros((1, 1)),
        float(refocusing.t[1] - refocusing.t[0]),
    )[0]
    assert m[2] < -0.95


def test_a_recursive_train_needs_a_pulse():
    with pytest.raises(ValueError, match="n_segments"):
        pp.make_recursive_slr_pulses(0, system=SYSTEM)
