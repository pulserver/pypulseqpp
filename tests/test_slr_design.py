"""SLR filter designs, root-flipped pulses, and the phases of multiband ones.

Each filter is held to the slice it selects in a Bloch simulation. Root flipping trades the linear phase of an SLR profile for a lower peak B1
and keeps the profile's magnitude; that is held against a Bloch simulation
rather than against the transform that built the pulse. The compiled search
is held to an exhaustive NumPy one over the same flip patterns, at sample
counts where the two pad their spectra identically.
"""

from __future__ import annotations

import numpy as np
import pytest
from pypulseqpp._ext import slr as kernels

import pypulseqpp as pp
from pypulseqpp import _slr
from pypulseqpp._band_phases import band_phases

#: Seconds per sample the dimensionless pulses are simulated at.
DWELL = 4e-6


def search_inputs(n, tbw, pulse_type, filter_type="min"):
    """What `design_slr` hands the search: Leja-ordered roots, the passband's
    flippable ones, and the peak |beta| every candidate is scaled to."""
    scale, d1, d2 = _slr._calc_ripples(pulse_type, 0.01, 0.001)
    if filter_type == "min":
        beta = _slr._minimum_phase(n, tbw, d1, d2)[::-1]
    else:
        beta = _slr._least_squares(n, tbw, d1, d2)
    roots = _slr._leja(np.roots(scale * np.asarray(beta, dtype=complex)))
    candidates = (np.abs(1.0 - np.abs(roots)) > 0.004) & (
        np.abs(np.angle(roots)) < tbw / n * np.pi
    )
    target = float(np.sin(_slr.NOMINAL_FLIP[pulse_type] / 2 + np.arctan(2 * d1) / 2))
    return roots, candidates, target


def unflipped(roots, target, n):
    return _slr._flipped_pulse(roots, np.zeros(roots.size, dtype=bool), target, n)


def root_flipped(n, tbw, pulse_type, filter_type="min"):
    return _slr.design_slr(
        n,
        tbw,
        pulse_type=pulse_type,
        filter_type=filter_type,
        stopband_ripple=0.001,
        root_flip=True,
    )


def profile(rf, n, tbw):
    """Magnetisation after the pulse, from +z, across three bandwidths."""
    offsets = np.linspace(-3.0, 3.0, 241) * tbw / (n * DWELL)
    return pp.bloch(np.asarray(rf) / (2 * np.pi * DWELL), offsets[:, None], DWELL)


# %% filters


@pytest.mark.parametrize("filter_type", ["ls", "pm", "min", "max", "ms"])
def test_every_filter_design_selects_the_slice_it_was_designed_for(filter_type):
    n, tbw = 128, 4
    rf = _slr.design_slr(n, tbw, pulse_type="ex", filter_type=filter_type)
    bandwidth = tbw / (n * DWELL)
    offsets = np.array([0.0, 3.0 * bandwidth, -3.0 * bandwidth])[:, None]
    magnetisation = pp.bloch(rf / (2 * np.pi * DWELL), offsets, DWELL)
    assert np.hypot(*magnetisation[0, :2]) > 0.95
    assert np.all(magnetisation[1:, 2] > 0.95)


def centroid(filter_type):
    """Where the pulse's energy sits, as a fraction of its duration."""
    energy = np.abs(_slr.design_slr(128, 4, pulse_type="ex", filter_type=filter_type))
    energy = energy**2
    return (np.arange(energy.size) + 0.5) @ energy / energy.sum() / energy.size


def test_minimum_and_maximum_phase_pulses_load_opposite_ends():
    """What makes either worth choosing over a linear-phase pulse, whose energy
    sits in the middle: the echo forms near one end of the pulse."""
    assert centroid("ls") == pytest.approx(0.5, abs=1e-3)
    assert centroid("min") > 0.6
    assert centroid("max") < 0.4
    assert centroid("min") + centroid("max") == pytest.approx(1.0, abs=1e-3)


# %% root flipping


CASES = [(128, 12, "ex", "min"), (256, 8, "se", "ls"), (256, 8, "inv", "min")]


@pytest.mark.parametrize("n, tbw, pulse_type, filter_type", CASES)
def test_root_flipping_keeps_the_magnitude_of_the_slice_profile(
    n, tbw, pulse_type, filter_type
):
    """Flipping leaves |beta| exactly as it was on the SLR model's own grid.
    A Bloch simulation turns about the RF and the off-resonance together each
    step where that model turns about one and then the other, and two pulses
    with different phase histories pick up that difference differently where
    the profile is steepest -- in the transition band, by a few thousandths."""
    roots, _, target = search_inputs(n, tbw, pulse_type, filter_type)
    flipped = profile(root_flipped(n, tbw, pulse_type, filter_type), n, tbw)
    plain = profile(unflipped(roots, target, n), n, tbw)
    assert np.allclose(flipped[:, 2], plain[:, 2], atol=5e-3)
    assert np.allclose(
        np.hypot(flipped[:, 0], flipped[:, 1]),
        np.hypot(plain[:, 0], plain[:, 1]),
        atol=5e-3,
    )


@pytest.mark.parametrize("n, tbw, pulse_type, filter_type", CASES)
def test_root_flipping_never_raises_the_peak(n, tbw, pulse_type, filter_type):
    """The unflipped pulse is one of the patterns searched."""
    roots, _, target = search_inputs(n, tbw, pulse_type, filter_type)
    peak = np.abs(root_flipped(n, tbw, pulse_type, filter_type)).max()
    assert peak <= np.abs(unflipped(roots, target, n)).max() * (1 + 1e-12)


def test_root_flipping_lowers_the_peak_of_a_high_bandwidth_excitation():
    roots, _, target = search_inputs(128, 12, "ex")
    peak = np.abs(root_flipped(128, 12, "ex")).max()
    assert peak < 0.65 * np.abs(unflipped(roots, target, 128)).max()


@pytest.mark.parametrize(
    "n, tbw, pulse_type", [(128, 4, "se"), (256, 8, "inv"), (128, 8, "ex")]
)
def test_the_compiled_search_picks_the_pattern_an_exhaustive_numpy_search_does(
    n, tbw, pulse_type
):
    roots, candidates, target = search_inputs(n, tbw, pulse_type)
    index = np.flatnonzero(candidates)

    def pattern(number):
        flips = np.zeros(roots.size, dtype=bool)
        flips[index] = [(number >> j) & 1 for j in range(index.size)]
        return flips

    peaks = [
        np.abs(_slr._flipped_pulse(roots, pattern(number), target, n)).max()
        for number in range(2**index.size)
    ]
    best = int(np.argmin(peaks))
    flips, peak = kernels.root_flip_search(roots, candidates, target, n)
    assert np.array_equal(flips, pattern(best))
    assert peak == pytest.approx(peaks[best], rel=1e-9)


def test_the_search_does_not_depend_on_how_many_threads_share_it():
    roots, candidates, target = search_inputs(128, 12, "ex")
    alone = kernels.root_flip_search(roots, candidates, target, 128, 1)
    shared = kernels.root_flip_search(roots, candidates, target, 128, 5)
    assert np.array_equal(alone[0], shared[0])
    assert alone[1] == shared[1]


def test_leja_ordering_permutes_the_roots_starting_from_the_largest():
    roots = np.roots(np.random.default_rng(0).standard_normal(20))
    ordered = _slr._leja(roots)
    assert np.allclose(np.sort_complex(ordered), np.sort_complex(roots))
    assert abs(ordered[0]) == pytest.approx(np.abs(roots).max())


def test_a_small_tip_pulse_has_no_nominal_flip_to_root_flip_for():
    with pytest.raises(ValueError, match="nominal flip"):
        pp.make_slr_pulse(np.deg2rad(10), root_flip=True)


def test_root_flipping_and_cancelling_the_alpha_phase_are_refused_together():
    with pytest.raises(ValueError, match="alpha phase"):
        pp.make_slr_pulse(
            np.pi, pulse_type="se", root_flip=True, cancel_alpha_phase=True
        )


def test_a_root_flipped_pulse_plays_at_the_amplitude_it_was_designed_at():
    """Its area is no measure of its flip, so it is not scaled by area."""
    system = pp.Opts()
    dwell = system.rf_raster_time
    flip = np.deg2rad(150)
    rf = pp.make_slr_pulse(
        flip, duration=1e-3, pulse_type="se", root_flip=True, system=system
    )
    design = _slr.design_slr(round(1e-3 / dwell), 4, pulse_type="se", root_flip=True)
    assert np.allclose(
        np.asarray(rf.signal) * 2 * np.pi * dwell, design * flip / np.pi, rtol=1e-6
    )


def test_a_root_flipped_pulse_on_a_fine_raster_keeps_the_coarse_design_profile():
    """Finding beta's roots costs the cube of its length, so a pulse with more
    samples than `ROOT_FLIP_SAMPLES` is designed at that many and resampled."""
    coarse_n = _slr.ROOT_FLIP_SAMPLES
    fine_n = 4 * coarse_n
    duration = coarse_n * DWELL
    coarse = _slr.design_slr(coarse_n, 8, pulse_type="inv", root_flip=True)
    fine = _slr.design_slr(fine_n, 8, pulse_type="inv", root_flip=True)
    offsets = (np.linspace(-3.0, 3.0, 241) * 8 / duration)[:, None]
    fine_dwell = duration / fine_n
    at_fine = pp.bloch(fine / (2 * np.pi * fine_dwell), offsets, fine_dwell)
    at_coarse = pp.bloch(coarse / (2 * np.pi * DWELL), offsets, DWELL)
    assert np.allclose(at_fine[:, 2], at_coarse[:, 2], atol=2e-2)


# %% band phases


def band_sum_peak(phases):
    count = phases.size
    t = np.linspace(0.0, 1.0, 4096, endpoint=False)
    k = np.arange(count) - (count - 1) / 2
    return np.abs(
        np.exp(1j * (2 * np.pi * k[:, None] * t + phases[:, None])).sum(0)
    ).max()


@pytest.mark.parametrize(
    "schedule, counts",
    [("wong", range(3, 17)), ("malik", range(4, 13)), ("quadratic", range(3, 17))],
)
def test_a_phase_schedule_lowers_the_peak_of_the_band_sum(schedule, counts):
    """Bands in phase peak at the band count; a schedule spreads them."""
    for count in counts:
        assert band_sum_peak(band_phases(count, schedule)) < count


@pytest.mark.parametrize("count", range(4, 13))
def test_malik_phases_make_the_modulation_of_symmetric_bands_real(count):
    t = np.linspace(0.0, 1.0, 512, endpoint=False)
    k = np.arange(count) - (count - 1) / 2
    phases = band_phases(count, "malik")
    modulation = np.exp(1j * (2 * np.pi * k[:, None] * t + phases[:, None])).sum(0)
    assert np.allclose(modulation.imag, 0.0, atol=1e-12)


def test_the_quadratic_schedule_rises_with_the_square_of_the_distance_from_the_centre():
    assert np.allclose(band_phases(5, "quadratic"), 3.4 / 5 * np.array([4, 1, 0, 1, 4]))


def test_make_sms_pulse_takes_a_phase_schedule_by_name():
    base = pp.make_sinc_pulse(np.deg2rad(30), duration=2e-3)
    _, _, weights = pp.make_sms_pulse(base, 5, 1000.0, phases="wong")
    assert np.allclose(
        np.exp(1j * np.angle(weights)), np.exp(1j * band_phases(5, "wong"))
    )


def test_a_phase_table_is_refused_outside_the_band_counts_it_covers():
    base = pp.make_sinc_pulse(np.deg2rad(30), duration=2e-3)
    with pytest.raises(ValueError, match="tabulated for 4 to 12"):
        pp.make_sms_pulse(base, 3, 1000.0, phases="malik")
