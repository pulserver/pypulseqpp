"""Sequence report parity and independently calculated encoding statistics."""

import math

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext as _cxx


def gradient_echo(lines: int = 16) -> pp.Sequence:
    """A Cartesian gradient echo: one readout at as many phase encodes."""
    system = pp.Opts()
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid("x", flat_area=1000, flat_time=4e-3, system=system)
    adc = pp.make_adc(
        num_samples=32, duration=readout.flat_time, delay=readout.rise_time
    )
    prewinder = pp.make_trapezoid("x", area=-readout.area / 2, system=system)
    for line in range(lines):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3, use="excitation"))
        seq.add_block(
            prewinder,
            pp.make_trapezoid(
                "y", area=(line - lines / 2) * 20, duration=1e-3, system=system
            ),
        )
        seq.add_block(readout, adc)
        seq.add_block(pp.make_delay(5e-3))
    return seq


def radial(spokes: int = 17) -> pp.Sequence:
    """The same readout swept around the origin: one spoke per excitation."""
    system = pp.Opts()
    seq = pp.Sequence(system)
    adc = pp.make_adc(num_samples=32, duration=4e-3, delay=0.0)
    for spoke in range(spokes):
        angle = math.pi * spoke / spokes
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3, use="excitation"))
        seq.add_block(
            pp.make_trapezoid(
                "x", flat_area=1000 * math.cos(angle), flat_time=4e-3, system=system
            ),
            pp.make_trapezoid(
                "y", flat_area=1000 * math.sin(angle), flat_time=4e-3, system=system
            ),
            adc,
        )
        seq.add_block(pp.make_delay(5e-3))
    return seq


def pulses_only() -> pp.Sequence:
    """Two excitations and nothing that acquires."""
    seq = pp.Sequence(pp.Opts())
    for _ in range(2):
        seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"))
        seq.add_block(pp.make_delay(10e-3))
    return seq


def plain_coverage(k_traj_adc, threshold):
    """The binning of the sampled trajectory, written out sample by sample."""
    keys = np.round(k_traj_adc / threshold).astype(np.int32)
    axes, samples = keys.shape

    visits = {}
    order = []
    for i in range(samples):
        column = tuple(keys[:, i])
        if column not in visits:
            visits[column] = 0
            order.append(i)
        visits[column] += 1
    counts = np.array(list(visits.values()), dtype=float)

    positions = []
    for axis in range(axes):
        seen = {}
        for i in order:
            key = int(keys[axis, i])
            if key in seen or key + 1 in seen or key - 1 in seen:
                continue
            seen[key] = len(seen)
        positions.append(len(seen))

    return {
        "unique_positions": np.array(positions, dtype=float),
        "repeats_min": counts.min(),
        "repeats_max": counts.max(),
        "repeats_median": float(np.median(counts)),
        "is_cartesian": np.prod(positions) == len(order),
    }


@pytest.fixture(params=[gradient_echo, radial], ids=lambda build: build.__name__)
def encoded(request):
    """A sequence that acquires, built either Cartesian or radial."""
    return request.param()


def test_binning_the_trajectory_matches_the_plain_pass(encoded):
    k_traj_adc = encoded.calculate_kspace()[0]
    extent = np.max(np.abs(k_traj_adc), axis=1)
    threshold = np.max(extent) / 4e6
    k_traj_adc = np.ascontiguousarray(k_traj_adc[extent >= threshold])

    found = _cxx.kspace_coverage(k_traj_adc, threshold)
    plain = plain_coverage(k_traj_adc, threshold)

    assert list(found["unique_positions"]) == list(plain["unique_positions"])
    assert found["repeats_min"] == plain["repeats_min"]
    assert found["repeats_max"] == plain["repeats_max"]
    assert found["repeats_median"] == plain["repeats_median"]
    assert found["is_cartesian"] == plain["is_cartesian"]


def test_a_cartesian_scan_is_reported_as_one():
    data = gradient_echo(lines=16).test_report_dict()

    assert data["is_cartesian"]
    assert data["dimensions"] == 2
    assert list(data["unique_k_positions"]) == [32, 16]


def test_a_radial_scan_is_not_reported_as_cartesian():
    data = radial().test_report_dict()

    assert not data["is_cartesian"]
    assert data["dimensions"] == 2


def test_a_pulse_played_at_one_flip_angle_is_reported_once():
    data = gradient_echo(lines=16).test_report_dict()

    assert data["flip_angles_deg"] == pytest.approx([22.5])


def test_a_pulse_played_many_times_is_integrated_once():
    seq = pp.Sequence(pp.Opts())
    pulse = pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation")
    for _ in range(100):
        seq.add_block(pulse)
        seq.add_block(pp.make_delay(5e-3))

    assert seq._native.num_rf() == 100
    assert seq._native.num_rf_definitions() == 1
    assert seq.test_report_dict()["flip_angles_deg"] == pytest.approx([90])


def test_a_pulse_swept_over_many_flip_angles_is_one_envelope():
    """One pulse at many amplitudes: one integral, and a multiply per shot."""
    seq = pp.Sequence(pp.Opts())
    asked = np.linspace(10, 180, 32)
    for angle in asked:
        seq.add_block(
            pp.make_block_pulse(math.radians(angle), duration=1e-3, use="excitation")
        )
        seq.add_block(pp.make_delay(5e-3))

    assert seq.test_report_dict()["flip_angles_deg"] == pytest.approx(asked)

    # A pulse registered again brings shapes of its own, so the sweep is as
    # many definitions as shots until duplicates are collapsed. The angles are
    # the same either way, to the six digits the file records an amplitude at.
    seq.remove_duplicates(in_place=True)

    assert seq._native.num_rf() == len(asked)
    assert seq._native.num_rf_definitions() == 1
    assert seq.test_report_dict()["flip_angles_deg"] == pytest.approx(asked, rel=1e-5)


def test_the_flip_angle_is_what_the_pulse_was_asked_for():
    seq = pp.Sequence(pp.Opts())
    for flip in (math.pi / 6, math.pi / 2, math.pi):
        seq.add_block(pp.make_sinc_pulse(flip, duration=2e-3, use="excitation"))
        seq.add_block(pp.make_delay(10e-3))

    assert seq.test_report_dict()["flip_angles_deg"] == pytest.approx(
        [30, 90, 180], rel=1e-3
    )


def test_the_repetition_time_is_one_repetition_long():
    seq = gradient_echo(lines=16)
    blocks_per_shot = len(seq) // 16

    shot = np.asarray(seq._native.block_durations())[:blocks_per_shot].sum()

    assert seq.test_report_dict()["TR"] == pytest.approx(shot)


def test_a_sequence_that_acquires_nothing_still_reports():
    data = pulses_only().test_report_dict()

    assert math.isnan(data["TE"])
    assert data["TR"] == pytest.approx(11e-3)
    assert data["flip_angles_deg"] == pytest.approx([90])
    assert list(data["unique_k_positions"]) == [1]
    assert "dimensions" not in data
    assert data["max_gradient"]["absolute_Hz_m"] == 0


def test_the_report_counts_every_library_the_sequence_fills():
    seq = gradient_echo(lines=4)
    seq.add_block(pp.make_delay(1e-3), pp.make_digital_output_pulse("osc0"))

    libraries = seq.test_report_dict()["libraries"]

    assert libraries["Trigger"] == 1
    assert libraries["Rotation"] == 0
    assert libraries["ADC"] == seq._native.num_adc()
    assert libraries["Shape"] == seq._native.num_shapes()


def test_a_failed_timing_check_is_reported_with_its_findings():
    seq = pp.Sequence(pp.Opts(adc_samples_divisor=4))
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"))
    seq.add_block(
        pp.make_trapezoid("x", flat_area=1000, flat_time=4e-3),
        pp.make_adc(num_samples=50, duration=4e-3),
    )

    report = seq.test_report()

    assert "Event timing check failed with 1 errors" in report
    assert "adc.num_samples" in report


def as_core(theirs):
    """The toolbox's sequence, held by the compiled core."""
    import convert

    ours = pp.Sequence(theirs.system)
    ours._native = convert.to_core(theirs)
    return ours


def assert_same_report(expected, found):
    """Hold two reports equal, entry by entry.

    Every entry is a number or a container of numbers, save the timing
    findings: the toolbox reports each as a formatted line and this reports it
    as the fields the line is formatted from, so what is held is that both
    found the same problems.
    """
    assert set(expected) <= set(found)
    for name, value in expected.items():
        if name == "timing_error_report":
            assert len(value) == len(found[name]), name
        elif isinstance(value, dict):
            for entry, number in value.items():
                assert np.allclose(
                    np.asarray(number, dtype=float),
                    np.asarray(found[name][entry], dtype=float),
                    equal_nan=True,
                ), f"{name}.{entry}"
        else:
            assert np.allclose(
                np.asarray(value, dtype=float),
                np.asarray(found[name], dtype=float),
                equal_nan=True,
            ), name


#: Where the toolbox's report is decided by its own integration error.
#:
#: It holds an axis at zero in front of what it plays by putting a knot a
#: picosecond ahead of the axis's first corner, and loses the corner to it --
#: so a ramp is a picosecond too long and encloses amplitude times half a
#: picosecond too much, every time an axis starts anywhere but the beginning.
#: See `tests/test_kspace.py`.
#:
#: Small enough to be invisible until something is balanced on it, and a
#: report is. `inversion_recovery_train` has four repetitions whose k-space
#: centres agree to twelve figures; which of them is nearest the origin
#: decides the echo, and the spacing after it is what the report calls TR.
#: The leak biases the first, so the toolbox settles on the second and calls
#: TR 0.557 -- the gap between the second repetition and the third. Without
#: it the four agree and the first wins, which is the gap of 0.537.
#:
#: The train's spacings are 0.537, 0.557 and 0.607 seconds: it has no one TR,
#: and what the report can say is the spacing after the echo it found. So the
#: value held here is this package's, and the toolbox's is recorded beside it
#: as the thing it is -- an answer chosen by an artefact.
DECIDED_BY_THE_LEAK = {
    "inversion_recovery_train": {"TR": 0.537},
}

#: The same, in a rendered report.
#:
#: `gre_with_noise_scan` resolves exactly 9.375 mm along its second axis, and
#: that is the number this package computes -- to the bit. Printed to two
#: places it is a tie, and Python rounds a tie to even: 9.38. The toolbox's
#: extent carries the leak, so its resolution is 9.37499999..., a hair under
#: the tie and not a tie at all: 9.37.
RENDERED_DIFFERENTLY = {
    "gre_with_noise_scan": (
        "Spatial resolution: 9.37 mm",
        "Spatial resolution: 9.38 mm",
    ),
}


def test_the_report_is_the_toolboxs(build_reference, reference_name):
    theirs = build_reference()
    ours = as_core(theirs)

    try:
        expected = theirs.test_report_dict()
    except ValueError:
        pytest.skip("the toolbox looks for an echo where nothing was acquired")

    expected = {**expected, **DECIDED_BY_THE_LEAK.get(reference_name, {})}
    assert_same_report(expected, ours.test_report_dict())


def test_the_report_reads_as_the_toolbox_writes_it(build_reference, reference_name):
    theirs = build_reference()
    ours = as_core(theirs)

    try:
        expected = theirs.test_report()
    except (NameError, ValueError):
        pytest.skip("the toolbox cannot render a report for this sequence")

    if reference_name in RENDERED_DIFFERENTLY:
        theirs_reads, ours_reads = RENDERED_DIFFERENTLY[reference_name]
        assert theirs_reads in expected
        expected = expected.replace(theirs_reads, ours_reads)

    assert ours.test_report() == expected
