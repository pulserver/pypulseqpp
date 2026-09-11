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
    """Per band, the largest amplitude and its window, from the expanded waveforms."""
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
        best, where = 0.0, 0
        for k in range(count):
            x = padded[axis][k * step : k * step + width]
            spectrum = np.abs(np.fft.rfft((x - x.mean()) * taper, length))
            amplitude = 2 * spectrum[lo : hi + 1].max() / taper.sum()
            if amplitude > best:
                best, where = amplitude, k
        found.append((best, where))
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
    for entry, (peak, window) in zip(report.bands, expected, strict=True):
        assert entry.peak * to_hz == pytest.approx(peak, rel=1e-9)
        assert entry.window == window


@pytest.mark.skipif(not _mkl_runtime(), reason="no MKL runtime installed")
def test_mkl_and_the_compiled_fft_agree(system):
    sequence = played(
        system,
        [sinusoid(system, "x", 4.0, 620.0, 0.1)],
        [pp.make_trapezoid("x", area=3000, duration=2e-3, system=system)],
    )
    arguments = {
        "bands": [(-1, 500.0, 700.0, 1e3)],
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
        bands=[(0, 550.0, 650.0, 1.0)],
        window=40e-3,
        stride=20e-3,
        oversampling=3,
        rotation=np.eye(3).tolist(),
        mkl_runtime=str(tmp_path / "libmkl_rt.so"),
    )

    assert found["backend"] == "pocketfft"


# -- vendor tables -----------------------------------------------------------


def test_an_esp_table_reads_as_bands_at_half_the_echo_spacing_rate(tmp_path):
    table = tmp_path / "epiesp.dat"
    table.write_text("# x\n1\n500 700 0.2\n1\n480 720 0.15\n0\n")

    bands = safety.read_forbidden_bands(table)

    assert [band.axis for band in bands] == ["x", "y"]
    assert bands[0].f_min == pytest.approx(5e5 / 700)
    assert bands[0].f_max == pytest.approx(1000.0)
    assert bands[0].tolerance == pytest.approx(2.0)
    assert bands[1].tolerance == pytest.approx(1.5)


@pytest.mark.parametrize(
    "text",
    ["1\n500 700 0.2\n", "1\n700 500 0.2\n0\n0\n", "one\n0\n0\n", "11\n0\n0\n"],
    ids=["truncated", "inverted", "unreadable", "implausible"],
)
def test_a_corrupt_esp_table_is_refused(tmp_path, text):
    table = tmp_path / "epiesp.dat"
    table.write_text(text)

    with pytest.raises(ValueError, match="ESP table"):
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
    table = tmp_path / "epiesp.dat"
    table.write_text("1\n770 850 0.3\n0\n0\n")
    sequence = played(system, [sinusoid(system, "x", 5.0, 600.0, 0.2)])

    _, from_path = safety.check_mech_resonance(sequence, table)
    _, from_bands = safety.check_mech_resonance(
        sequence, safety.read_forbidden_bands(table)
    )

    assert from_path.bands == from_bands.bands
