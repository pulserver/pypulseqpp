"""Mechanical-resonance check: windowed spectra of the physical axes against bands."""

import math

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext, safety
from pypulseqpp.safety import ForbiddenBand
from pypulseqpp.safety._resonance import _mkl_runtime


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
    )


def sinusoid(system, channel, amplitude, frequency, duration):
    """A sinusoidal gradient of `amplitude` mT/m, sampled at raster centres."""
    dt = system.grad_raster_time
    n = round(duration / dt)
    hz = amplitude * 1e-3 * system.gamma
    wave = hz * np.sin(2 * math.pi * frequency * (np.arange(n) + 0.5) * dt)
    return pp.make_arbitrary_grad(
        channel,
        wave,
        first=0.0,
        last=hz * math.sin(2 * math.pi * frequency * n * dt),
        system=system,
    )


def played(system, *blocks):
    sequence = pp.Sequence(system)
    for block in blocks:
        sequence.add_block(*block)
    return sequence


def windows_the_long_way(sequence, dt, bands, window, stride, oversampling):
    """Per band, the largest amplitude and the windows reading it, from the expanded waveforms.

    Windows holding the same samples read the same amplitude to rounding, so
    every window within 1e-9 of the largest is one the check may name.
    """
    total = round(sequence.duration()[0] / dt)
    t = (np.arange(total) + 0.5) * dt
    samples = [
        np.interp(t, channel[0], channel[1], left=0.0, right=0.0)
        if channel.shape[1]
        else np.zeros(total)
        for channel in sequence.waveforms()
    ]
    width = round(window / dt)
    step = round(stride / dt)
    length = oversampling * width
    count = 1 if total <= width else -(-(total - width) // step) + 1
    padded = [
        np.concatenate([x, np.zeros(max(0, (count - 1) * step + width - total))])
        for x in samples
    ]
    taper = 0.5 * (1 - np.cos(2 * np.pi * np.arange(width) / width))
    bin_width = 1.0 / (length * dt)

    found = []
    for axis, f_min, f_max in bands:
        lo = math.ceil(f_min / bin_width - 1e-9)
        hi = math.floor(f_max / bin_width + 1e-9)
        amplitudes = np.empty(count)
        for k in range(count):
            x = padded[axis][k * step : k * step + width]
            spectrum = np.abs(np.fft.rfft((x - x.mean()) * taper, length))
            amplitudes[k] = 2 * spectrum[lo : hi + 1].max() / taper.sum()
        best = amplitudes.max()
        found.append((best, set(np.flatnonzero(amplitudes >= best * (1 - 1e-9)))))
    return found


# -- the amplitude is a physical one ------------------------------------------


def test_a_sustained_sinusoid_reads_its_own_amplitude(system):
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)])

    _, report = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])

    assert report.bands[0].peak == pytest.approx(5.0, rel=0.02)
    assert report.bands[0].frequency == pytest.approx(600.0, abs=report.frequency_step)


def test_the_reading_does_not_depend_on_the_raster(system):
    fine = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=5e-6,
    )
    band = [("x", 550.0, 650.0, 0.0)]

    _, coarse_report = safety.check_mech_resonance(
        played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)]), band
    )
    _, fine_report = safety.check_mech_resonance(
        played(fine, [sinusoid(fine, "x", 5.0, 600.0, 0.2)]), band
    )

    assert fine_report.bands[0].peak == pytest.approx(
        coarse_report.bands[0].peak, rel=0.01
    )


def test_the_sequence_is_sampled_on_the_raster_it_was_designed_with(system):
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)])
    other = pp.Opts(grad_raster_time=4e-6)

    _, own = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])
    _, weighed = safety.check_mech_resonance(
        sequence, [("x", 550.0, 650.0, 0.0)], system=other
    )

    assert weighed.frequency_step == own.frequency_step
    assert own.frequency_step == pytest.approx(1 / (3 * 40e-3))


# -- the verdict -------------------------------------------------------------


def test_a_drive_outside_every_band_passes_whatever_its_amplitude(system):
    sequence = played(system, [sinusoid(system, "x", 30.0, 300.0, 0.2)])

    ok, report = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])

    assert ok
    assert report.bands[0].peak < 0.01 * 30.0


def test_a_stated_tolerance_is_the_threshold_even_below_the_floor(system):
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)])

    refused, stated = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 3.0)])
    allowed, silent = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])

    assert not refused
    assert stated.bands[0].threshold == 3.0
    assert stated.bands[0].violations > 0
    assert allowed
    assert silent.bands[0].threshold == 10.0


def test_the_floor_is_what_a_band_without_tolerance_is_held_to(system):
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)])

    ok, _ = safety.check_mech_resonance(
        sequence, [("x", 550.0, 650.0, 0.0)], min_threshold=4.0
    )

    assert not ok


def test_the_worst_window_is_kept_when_nothing_violates(system):
    sequence = played(
        system,
        [pp.make_delay(0.2)],
        [sinusoid(system, "x", 2.0, 600.0, 0.03)],
        [pp.make_delay(0.2)],
    )

    ok, report = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])

    worst = report.bands[0]
    assert ok
    assert worst.violations == 0
    assert worst.window_start == pytest.approx(worst.window * report.stride)
    centre = worst.window_start + 0.5 * report.window_width
    assert abs(centre - 0.215) <= 0.5 * report.stride + 1e-9


def test_the_default_stride_is_half_a_window(system):
    sequence = played(system, [sinusoid(system, "x", 1.0, 600.0, 0.2)])

    _, report = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])

    assert report.window_width == pytest.approx(40e-3)
    assert report.stride == pytest.approx(20e-3)


def test_nothing_to_guard_passes():
    ok, report = safety.check_mech_resonance(pp.Sequence(pp.Opts()), [])

    assert ok
    assert report.bands == []


# -- axes and orientation ----------------------------------------------------


def test_a_band_guards_only_the_axis_it_names(system):
    sequence = played(system, [sinusoid(system, "y", 20.0, 600.0, 0.2)])

    ok, report = safety.check_mech_resonance(sequence, [("x", 550.0, 650.0, 0.0)])

    assert ok
    assert [reading.axis for reading in report.bands[0].axes] == ["x"]
    assert report.bands[0].peak < 1e-6


def test_a_band_without_an_axis_guards_all_three(system):
    sequence = played(system, [sinusoid(system, "y", 20.0, 600.0, 0.2)])

    ok, report = safety.check_mech_resonance(sequence, [(None, 550.0, 650.0, 0.0)])

    band = report.bands[0]
    assert not ok
    assert band.axis is None
    assert band.peak_axis == "y"
    assert [reading.axis for reading in band.axes] == ["x", "y", "z"]
    assert [reading.violations > 0 for reading in band.axes] == [False, True, False]


def test_each_band_keeps_its_own_worst_window(system):
    sequence = played(
        system,
        [sinusoid(system, "x", 3.0, 600.0, 0.06)],
        [pp.make_delay(0.2)],
        [sinusoid(system, "x", 3.0, 900.0, 0.06)],
    )

    _, report = safety.check_mech_resonance(
        sequence, [("x", 550.0, 650.0, 0.0), ("x", 850.0, 950.0, 0.0)]
    )

    # The bursts play over 0-0.06 s and 0.26-0.32 s; each band's worst window
    # overlaps its own.
    early, late = report.bands
    assert early.window_start < 0.06
    assert late.window_start < 0.32
    assert late.window_start + report.window_width > 0.26


def test_a_window_violating_on_two_axes_counts_once(system):
    both = [
        sinusoid(system, "x", 20.0, 600.0, 0.2),
        sinusoid(system, "y", 20.0, 600.0, 0.2),
    ]
    sequence = played(system, both)

    _, report = safety.check_mech_resonance(sequence, [(None, 550.0, 650.0, 0.0)])

    x, y, _ = report.bands[0].axes
    assert x.violations == y.violations > 0
    assert report.bands[0].violations == x.violations


def test_a_turned_block_drives_the_axis_it_plays_on(system):
    sequence = played(
        system, [sinusoid(system, "x", 20.0, 600.0, 0.2), pp.make_rotation(math.pi / 2)]
    )

    _, report = safety.check_mech_resonance(sequence, [(None, 550.0, 650.0, 0.0)])

    x, y, _ = report.bands[0].axes
    assert y.peak == pytest.approx(20.0, rel=0.02)
    assert x.peak < 1e-6 * y.peak


def test_the_prescription_is_turned_after_the_blocks_own_rotation(system):
    # The block turns x onto y; the prescription turns y onto z. Composed the
    # other way round, the drive would end on y.
    sequence = played(
        system, [sinusoid(system, "x", 20.0, 600.0, 0.2), pp.make_rotation(math.pi / 2)]
    )
    prescription = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    _, report = safety.check_mech_resonance(
        sequence, [(None, 550.0, 650.0, 0.0)], rotation=prescription
    )

    x, y, z = report.bands[0].axes
    assert z.peak == pytest.approx(20.0, rel=0.02)
    assert max(x.peak, y.peak) < 1e-6 * z.peak


def test_a_prescription_must_be_a_rotation(system):
    sequence = played(system, [sinusoid(system, "x", 1.0, 600.0, 0.2)])

    with pytest.raises(ValueError, match="orthonormal"):
        safety.check_mech_resonance(sequence, [], rotation=2 * np.eye(3))


# -- the native path is the plain one ----------------------------------------


def test_the_native_windows_are_the_plain_windows(system):
    gx = sinusoid(system, "x", 4.0, 620.0, 0.05)
    trap = pp.make_trapezoid("y", area=2000, duration=2e-3, system=system)
    readout = pp.make_trapezoid(
        "x", amplitude=15e-3 * system.gamma, flat_time=0.8e-3, system=system
    )
    blocks = [[gx], [pp.make_delay(7e-3)], [trap, pp.make_rotation(0.3, 0.2)]]
    for sign in (1, -1) * 20:
        blocks.append([pp.scale_grad(readout, sign)])
    blocks.append([sinusoid(system, "z", 3.0, 1100.0, 0.02), pp.make_rotation(0.7)])
    sequence = played(system, *blocks)
    bands = [
        ("x", 550.0, 650.0, 0.0),
        ("y", 900.0, 1300.0, 0.0),
        ("z", 580.0, 590.0, 0.0),
    ]

    _, report = safety.check_mech_resonance(sequence, bands, stride=13e-3)

    expected = windows_the_long_way(
        sequence,
        system.grad_raster_time,
        [(0, 550.0, 650.0), (1, 900.0, 1300.0), (2, 580.0, 590.0)],
        report.window_width,
        report.stride,
        3,
    )
    to_hz = 1e-3 * system.gamma
    for entry, (peak, windows) in zip(report.bands, expected, strict=True):
        assert entry.peak * to_hz == pytest.approx(peak, rel=1e-9)
        assert entry.window in windows


def test_repeated_blocks_cut_anywhere_by_window_edges_read_as_the_plain_windows(system):
    readout = pp.make_trapezoid(
        "x", amplitude=12e-3 * system.gamma, flat_time=1.3e-3, system=system
    )
    short = pp.make_arbitrary_grad(
        "z",
        system.gamma * 0.4e-3 * np.array([1.0, 3.0, 4.0, 4.0, 2.0, -1.0, -2.0, 0.5]),
        first=0.0,
        last=0.0,
        system=system,
    )
    blocks = []
    for i in range(90):
        phase = pp.make_trapezoid(
            "y", area=40.0 * ((i % 9) - 4), duration=0.9e-3, system=system
        )
        turn = [pp.make_rotation(0.05 * (i % 4))] if i % 3 == 0 else []
        blocks.append([phase, short, *turn])
        blocks.append([pp.scale_grad(readout, 1 if i % 2 else -1), *turn])
        if i % 7 == 0:
            blocks.append([pp.make_delay(1.7e-3)])
    sequence = played(system, *blocks)
    bands = [
        ("x", 300.0, 420.0, 0.0),
        ("y", 150.0, 700.0, 0.0),
        ("z", 900.0, 1500.0, 0.0),
    ]

    _, report = safety.check_mech_resonance(sequence, bands, stride=7e-3)

    expected = windows_the_long_way(
        sequence,
        system.grad_raster_time,
        [(0, 300.0, 420.0), (1, 150.0, 700.0), (2, 900.0, 1500.0)],
        report.window_width,
        report.stride,
        3,
    )
    to_hz = 1e-3 * system.gamma
    for entry, (peak, windows) in zip(report.bands, expected, strict=True):
        assert entry.peak * to_hz == pytest.approx(peak, rel=1e-9)
        assert entry.window in windows


def test_a_scan_summed_over_many_passes_and_threads_reads_as_the_plain_windows(system):
    readout = pp.make_trapezoid(
        "x", amplitude=12e-3 * system.gamma, flat_time=1.3e-3, system=system
    )
    blocks = []
    for i in range(110):
        phase = pp.make_trapezoid(
            "y", area=40.0 * ((i % 9) - 4), duration=0.9e-3, system=system
        )
        turn = [pp.make_rotation(0.05 * (i % 4))] if i % 3 == 0 else []
        blocks.append([phase, *turn])
        blocks.append([pp.scale_grad(readout, 1 if i % 2 else -1), *turn])
        if i % 7 == 0:
            blocks.append([sinusoid(system, "z", 2.0, 1000.0, 1e-3)])
    sequence = played(system, *blocks)
    bands = [("x", 300.0, 900.0, 0.0), ("z", 500.0, 1500.0, 0.0)]

    # Windows of 2 ms every 50 us: several runs of 1024 intervals, each
    # summed by one thread, over more than one pass.
    _, report = safety.check_mech_resonance(
        sequence, bands, window_width=2e-3, stride=50e-6, frequency_oversampling=2
    )

    expected = windows_the_long_way(
        sequence,
        system.grad_raster_time,
        [(0, 300.0, 900.0), (2, 500.0, 1500.0)],
        report.window_width,
        report.stride,
        2,
    )
    assert report.windows > 4096
    to_hz = 1e-3 * system.gamma
    for entry, (peak, windows) in zip(report.bands, expected, strict=True):
        assert entry.peak * to_hz == pytest.approx(peak, rel=1e-9)
        assert entry.window in windows


@pytest.mark.parametrize("raster", [10e-6, 4e-6])
def test_a_block_after_200000_delays_reads_the_spectrum_it_has_alone(raster):
    """The delays, 1.23 ms each, are durations no double holds exactly."""
    system = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=raster,
        block_duration_raster=10e-6,
    )
    block = [
        sinusoid(system, "x", 5.0, 600.0, 10e-3),
        pp.make_trapezoid("y", area=1000, duration=3e-3, system=system),
    ]
    delays = 200_000
    deep = pp.Sequence(system)
    delay = pp.make_delay(1.23e-3)
    for _ in range(delays):
        deep.add_block(delay)
    deep.add_block(*block)
    # The last window holds 20 ms of silence, then the block.
    lead = played(system, [pp.make_delay(20e-3)], block)
    last = round(delays * 1.23e-3 / 20e-3) - 1

    first = safety.mech_resonance_spectrum(lead, window=0)
    later = safety.mech_resonance_spectrum(deep, window=last)

    assert later.window_start == pytest.approx(delays * 1.23e-3 - 20e-3)
    assert first.amplitude.max() > 1.0
    np.testing.assert_array_equal(later.amplitude, first.amplitude)


@pytest.mark.skipif(not _mkl_runtime(), reason="no MKL runtime installed")
def test_mkl_and_the_compiled_fft_agree(system):
    sequence = played(
        system,
        [sinusoid(system, "x", 4.0, 620.0, 0.1)],
        [pp.make_trapezoid("x", area=3000, duration=2e-3, system=system)],
    )
    arguments = {
        "bands": [(-1, 500.0, 700.0, 1e3, 0.0)],
        "window": 40e-3,
        "stride": 20e-3,
        "oversampling": 3,
        "rotation": np.eye(3).tolist(),
    }

    mkl = _ext.mech_resonance(sequence._native, mkl_runtime=_mkl_runtime(), **arguments)
    pocket = _ext.mech_resonance(sequence._native, mkl_runtime="", **arguments)

    assert (mkl["backend"], pocket["backend"]) == ("mkl", "pocketfft")
    for ours, theirs in zip(mkl["readings"], pocket["readings"], strict=True):
        assert ours["peak"] == pytest.approx(theirs["peak"], rel=1e-9)
        assert ours["window"] == theirs["window"]


def test_an_unloadable_mkl_falls_back_to_the_compiled_fft(system, tmp_path):
    sequence = played(system, [sinusoid(system, "x", 1.0, 600.0, 0.1)])

    found = _ext.mech_resonance(
        sequence._native,
        bands=[(0, 550.0, 650.0, 1.0, 0.0)],
        window=40e-3,
        stride=20e-3,
        oversampling=3,
        rotation=np.eye(3).tolist(),
        mkl_runtime=str(tmp_path / "libmkl_rt.so"),
    )

    assert found["backend"] == "pocketfft"


# -- vendor tables -----------------------------------------------------------


def test_a_table_other_than_an_asc_file_is_refused(tmp_path):
    table = tmp_path / "bands.dat"
    table.write_text("1\n500 700 0.2\n")

    with pytest.raises(ValueError, match="asc"):
        safety.read_forbidden_bands(table)


def test_an_asc_resonance_guards_every_axis_with_no_tolerance(tmp_path):
    table = tmp_path / "gradient.asc"
    table.write_text(
        "aflGCAcousticResonanceFrequency[0] = 590.0\n"
        "aflGCAcousticResonanceBandwidth[0] = 100.0\n"
    )

    assert safety.read_forbidden_bands(table) == [
        ForbiddenBand(None, 540.0, 640.0, 0.0)
    ]


def test_a_table_path_can_stand_for_the_bands(system, tmp_path):
    table = tmp_path / "gradient.asc"
    table.write_text(
        "aflGCAcousticResonanceFrequency[0] = 590.0\n"
        "aflGCAcousticResonanceBandwidth[0] = 100.0\n"
    )
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)])

    _, from_path = safety.check_mech_resonance(sequence, table)
    _, from_bands = safety.check_mech_resonance(
        sequence, safety.read_forbidden_bands(table)
    )

    assert from_path.bands == from_bands.bands


# -- the spectrum a diagram draws -------------------------------------------


def test_the_spectrum_reads_the_amplitude_the_check_reports(system):
    """A diagram of a band and the verdict on it come from the same pass."""
    sequence = played(system, [sinusoid(system, "x", 12.0, 700.0, 60e-3)])
    band = ForbiddenBand(axis=None, f_min=650.0, f_max=750.0, tolerance=5.0)

    _, report = safety.check_mech_resonance(sequence, [band], window_width=20e-3)
    worst = report.bands[0]
    spectrum = safety.mech_resonance_spectrum(
        sequence, window=worst.window, window_width=20e-3
    )

    bin_of_peak = round(worst.frequency / spectrum.frequency_step)
    assert spectrum.amplitude.shape[0] == 3
    assert spectrum.frequency[bin_of_peak] == pytest.approx(worst.frequency)
    assert spectrum.amplitude[:, bin_of_peak].max() == pytest.approx(worst.peak)
    assert spectrum.window_start == pytest.approx(worst.window_start)


def test_a_spectrum_is_read_on_every_axis_whether_a_band_guards_it_or_not(system):
    sequence = played(system, [sinusoid(system, "z", 9.0, 400.0, 60e-3)])

    spectrum = safety.mech_resonance_spectrum(sequence, window=0, window_width=20e-3)

    loudest = spectrum.amplitude.max(axis=1)
    assert loudest[2] > loudest[0]
    assert spectrum.axes == ("x", "y", "z")


def test_a_window_the_sequence_does_not_reach_is_refused(system):
    sequence = played(system, [sinusoid(system, "x", 12.0, 700.0, 60e-3)])

    with pytest.raises(ValueError, match="no window"):
        safety.mech_resonance_spectrum(sequence, window=99, window_width=20e-3)


# -- a second, longer read of the windows that exceed --------------------------


def burst(system, channel, amplitude, frequency, duration, silence):
    """A sinusoid of `duration` s between two stretches of `silence` s."""
    return [
        [pp.make_delay(silence)],
        [sinusoid(system, channel, amplitude, frequency, duration)],
        [pp.make_delay(silence)],
    ]


def test_a_steady_line_beside_a_band_is_leakage_on_the_longer_read(system):
    sequence = played(system, [sinusoid(system, "x", 20.0, 1080.0, 0.4)])
    band = [("x", 1100.0, 1200.0, 0.0)]

    once_ok, once = safety.check_mech_resonance(
        sequence, band, window_width=0.02, reread=False
    )
    ok, report = safety.check_mech_resonance(sequence, band, window_width=0.02)

    assert not once_ok
    assert ok
    assert report.bands[0].violations == 0
    assert report.bands[0].leakage == once.bands[0].violations


def test_a_short_train_in_a_band_stays_a_violation_on_the_longer_read(system):
    sequence = played(system, *burst(system, "x", 25.0, 1150.0, 0.012, 0.1))
    band = [("x", 1100.0, 1200.0, 0.0)]

    _, once = safety.check_mech_resonance(
        sequence, band, window_width=0.02, reread=False
    )
    ok, report = safety.check_mech_resonance(sequence, band, window_width=0.02)

    assert once.bands[0].violations > 0
    assert not ok
    assert report.bands[0].violations == once.bands[0].violations
    assert report.bands[0].leakage == 0


def two_tones(system, strong, weak):
    """`strong` mT/m at 300 Hz and `weak` mT/m at 1150 Hz on x, 0.2 s."""
    dt = system.grad_raster_time
    n = round(0.2 / dt)
    t = (np.arange(n) + 0.5) * dt
    hz = 1e-3 * system.gamma
    wave = hz * (
        strong * np.sin(2 * math.pi * 300.0 * t)
        + weak * np.sin(2 * math.pi * 1150.0 * t)
    )
    return played(
        system,
        [pp.make_arbitrary_grad("x", wave, first=0.0, last=wave[-1], system=system)],
    )


def test_a_stated_tolerance_limits_the_gradient_not_the_line(system):
    band = [("x", 1100.0, 1200.0, 16.0)]
    strong = two_tones(system, 30.0, 8.0)
    weak = two_tones(system, 5.0, 8.0)

    on_line, _ = safety.check_mech_resonance(strong, band, window_width=0.02)
    on_gradient, report = safety.check_mech_resonance(
        strong, band, window_width=0.02, tolerance_limits="gradient"
    )
    within, _ = safety.check_mech_resonance(
        weak, band, window_width=0.02, tolerance_limits="gradient"
    )

    assert on_line
    assert not on_gradient
    assert report.bands[0].threshold == 6.0
    assert report.bands[0].gradient > 30.0
    assert within


def test_a_tolerance_is_said_to_limit_a_line_or_a_gradient(system):
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.1)])

    with pytest.raises(ValueError, match="tolerance_limits"):
        safety.check_mech_resonance(
            sequence, [("x", 550.0, 650.0, 5.0)], tolerance_limits="plateau"
        )


@pytest.mark.parametrize("amplitude", [5.0, 15.0])
def test_a_wide_band_reads_the_same_verdict_capped_as_read_whole(system, amplitude):
    sequence = played(system, [sinusoid(system, "y", amplitude, 600.0, 0.2)])
    dt = system.grad_raster_time
    nyquist = 0.5 / dt

    ok, report = safety.check_mech_resonance(
        sequence, [("y", 300.0, nyquist, 0.0)], window_width=0.02, reread=False
    )
    ((best, _),) = windows_the_long_way(
        sequence, dt, [(1, 300.0, nyquist)], 0.02, 0.01, 3
    )
    best /= 1e-3 * system.gamma

    assert ok == (best <= 10.0)
    assert report.bands[0].peak == pytest.approx(best, rel=1e-6)


def test_a_violated_band_names_a_window_that_violates(system):
    """A stated tolerance on the gradient: the strongest line is where the gradient is weak."""
    dt = system.grad_raster_time
    hz = 1e-3 * system.gamma
    t = (np.arange(round(0.1 / dt)) + 0.5) * dt
    line = np.sin(2 * math.pi * 1150.0 * t)
    strong = hz * 9.0 * line
    weak = hz * (7.0 * line + 25.0 * np.sin(2 * math.pi * 100.0 * t))
    sequence = played(
        system,
        *(
            [pp.make_arbitrary_grad("x", wave, first=0.0, last=wave[-1], system=system)]
            for wave in (strong, weak)
        ),
    )

    ok, report = safety.check_mech_resonance(
        sequence,
        [("x", 1100.0, 1200.0, 16.0)],
        window_width=0.02,
        tolerance_limits="gradient",
    )

    assert not ok
    assert report.bands[0].gradient > 16.0
    assert report.bands[0].window_start + 0.02 > 0.1
