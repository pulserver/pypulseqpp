"""The MPRAGE example sequences: Cartesian, stack of stars and stack of spirals."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

CARTESIAN = "mprage3D_sequence"
STARS = "mprage_stack_of_stars3D_sequence"
SPIRALS = "mprage_stack_of_spirals3D_sequence"

#: A prescription small enough to build in a moment, per sequence.
SMALL = {
    CARTESIAN: {"n_x": 32, "n_y": 16, "n_z": 8, "ti": 100e-3, "tr": 300e-3},
    STARS: {"n": 32, "n_z": 4, "ti": 100e-3, "tr": 500e-3},
    SPIRALS: {"n": 32, "n_z": 4, "n_shots": 4, "ti": 100e-3, "tr": 300e-3},
}

APPS = {
    CARTESIAN: "Mprage3DApp",
    STARS: "MprageStackOfStars3DApp",
    SPIRALS: "MprageStackOfSpirals3DApp",
}


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app(name, **kwargs):
    cls = getattr(module(name), APPS[name])
    return cls(pp.Opts(), **{**SMALL[name], "n_dummy": 0, **kwargs})


def labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def pulse_times(seq, use):
    """Centre time (s) of every RF pulse of ``use``, in play order."""
    t, times = 0.0, []
    for index in range(1, len(seq.block_events) + 1):
        block = seq.get_block(index)
        rf = getattr(block, "rf", None)
        if rf is not None and rf.use == use:
            times.append(t + rf.delay + rf.center)
        t += block.block_duration
    return np.asarray(times)


def shots_of(built):
    """How many readout excitations each shot plays."""
    if hasattr(built, "n_readouts"):
        return built.n_readouts
    return len(getattr(built, "spokes", getattr(built, "arms", [])))


def partitions_of(built):
    if hasattr(built, "shots"):
        return [z for z, _ in built.shots]
    return built.partitions


@pytest.mark.parametrize("name", SMALL)
def test_a_small_mprage_passes_its_timing_check_and_repeats_one_shot(name):
    built = app(name, n_dummy=1)
    seq = built.design()
    size, start = seq._detect_tr()

    assert seq.check_timing()[0]
    assert start == 1
    # One shot is the repeating unit, the dummy included.
    assert size * (1 + len(partitions_of(built))) == len(seq.block_events)


@pytest.mark.parametrize("name", SMALL)
def test_every_shot_reads_one_partition_in_order(name):
    built = app(name)
    par, eco = labels(built.design(), "PAR", "ECO")
    per_shot = [int(n) for n in np.bincount(par)[partitions_of(built)]]

    assert sorted(set(par)) == partitions_of(built)
    assert list(par) == sorted(par)
    assert all(np.diff(eco)[np.diff(par) == 0] == 1)
    assert max(per_shot) <= shots_of(built)


@pytest.mark.parametrize("name", SMALL)
def test_every_shot_plays_the_same_number_of_excitations(name):
    built = app(name, n_dummy=2)
    seq = built.design()
    inversions = pulse_times(seq, "inversion")
    excitations = pulse_times(seq, "excitation")

    assert len(inversions) == 2 + len(partitions_of(built))
    assert len(excitations) == len(inversions) * shots_of(built)


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("ti", [100e-3, 150e-3])
def test_the_first_excitation_of_a_shot_is_ti_after_its_inversion(name, ti):
    built = app(name, ti=ti)
    seq = built.design()
    inversions = pulse_times(seq, "inversion")
    excitations = pulse_times(seq, "excitation")[:: shots_of(built)]
    raster = built.system.block_duration_raster

    assert excitations - inversions == pytest.approx(ti, abs=raster)


@pytest.mark.parametrize("name", SMALL)
def test_the_first_views_central_adc_sample_is_ti_plus_te_after_inversion(name):
    built = app(name)
    seq = built.design()
    sample_times, _ = seq.adc_times()
    first_centre = sample_times[built.ro.center_sample]
    inversion = pulse_times(seq, "inversion")[0]

    assert first_centre - inversion == pytest.approx(
        built.ti + built.ro.echo_time, abs=built.ro.adc.dwell / 2 + 1e-9
    )


@pytest.mark.parametrize("name", SMALL)
def test_inversions_are_one_tr_apart(name):
    built = app(name, tr=800e-3, n_dummy=1)
    inversions = pulse_times(built.design(), "inversion")

    assert np.diff(inversions) == pytest.approx(800e-3, abs=1e-9)
    assert built.duration == pytest.approx(800e-3 * len(inversions))


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize(
    "timing",
    [{"ti": 100.0033e-3, "tr": 800.0047e-3}, {"ti": None, "tr": None}],
    ids=["off the raster", "shortest"],
)
def test_the_resolved_ti_and_tr_are_the_intervals_the_inversions_play(name, timing):
    built = app(name, n_dummy=1, **timing)
    seq = built.design()
    inversions = pulse_times(seq, "inversion")
    excitations = pulse_times(seq, "excitation")[:: shots_of(built)]
    resolved = built.resolved

    assert excitations - inversions == pytest.approx(resolved["ti"], abs=1e-9)
    assert np.diff(inversions) == pytest.approx(resolved["tr"], abs=1e-9)
    assert np.atleast_1d(seq.definitions["TI"])[0] == pytest.approx(resolved["ti"])
    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(resolved["tr"])


@pytest.mark.parametrize("name", SMALL)
def test_navigators_ride_in_the_recovery_without_moving_the_inversions(name):
    built = app(name, tr=1.5, navigator=True)
    seq = built.design()

    assert built.n_navigators > 0
    assert np.diff(pulse_times(seq, "inversion")) == pytest.approx(1.5, abs=1e-9)
    assert seq.check_timing()[0]


@pytest.mark.parametrize("name", SMALL)
def test_dummy_shots_acquire_nothing(name):
    plain, dummied = app(name), app(name, n_dummy=2)

    assert len(labels(dummied.design(), "PAR")[0]) == len(
        labels(plain.design(), "PAR")[0]
    )


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize(
    ("prescription", "match"),
    [
        ({"ti": 1e-3}, "TI"),
        ({"tr": 50e-3}, "TR"),
        ({"excitation": "adiabatic"}, "excitation"),
    ],
    ids=["TI", "TR", "excitation"],
)
def test_an_mprage_that_cannot_be_played_is_refused(name, prescription, match):
    with pytest.raises(ValueError, match=match):
        app(name, **prescription)


# -- Cartesian ----------------------------------------------------------------


@pytest.mark.parametrize("ordering", ["radial", "shuffling"])
def test_every_sampled_view_is_read_once_inside_the_ellipse(ordering):
    built = app(CARTESIAN, ordering=ordering, ry=2, rz=2, n_acs_y=4, n_acs_z=2)
    lin, par = labels(built.design(), "LIN", "PAR")
    views = list(zip(lin, par, strict=True))
    outside = [
        (y, z)
        for y, z in views
        if ((y - 8) / 16) ** 2 + ((z - 4) / 8) ** 2 > 0.25
        and (y, z) not in built.calibration
    ]

    assert len(views) == len(set(views))
    assert set(views) == {(y, z) for z, lines in built.shots for y in lines}
    assert outside == []
    assert built.calibration <= set(views)


def test_a_radial_shot_reads_its_lines_centre_out():
    built = app(CARTESIAN)
    lin, par = labels(built.design(), "LIN", "PAR")

    for z in set(par):
        distance = np.abs(lin[par == z] - 8)
        assert list(distance) == sorted(distance)
        assert distance[0] == 0 or 8 not in lin[par == z]


def test_a_shuffled_shot_reads_the_same_lines_in_another_order():
    radial = app(CARTESIAN)
    shuffled = app(CARTESIAN, ordering="shuffling")

    assert [sorted(lines) for _, lines in radial.shots] == [
        sorted(lines) for _, lines in shuffled.shots
    ]
    assert [lines for _, lines in radial.shots] != [
        lines for _, lines in shuffled.shots
    ]


def test_a_partition_with_fewer_lines_pads_its_train_without_acquiring():
    built = app(CARTESIAN)
    lengths = {z: len(lines) for z, lines in built.shots}
    seq = built.design()
    (par,) = labels(seq, "PAR")

    assert len(set(lengths.values())) > 1
    assert {
        z: int(n) for z, n in zip(*np.unique(par, return_counts=True), strict=True)
    } == lengths
    assert len(pulse_times(seq, "excitation")) == len(lengths) * built.n_readouts


def test_an_elliptical_calibration_region_is_the_inscribed_ellipse():
    rectangle = app(CARTESIAN, ry=2, rz=2, n_acs_y=8, n_acs_z=6)
    ellipse = app(CARTESIAN, ry=2, rz=2, n_acs_y=8, n_acs_z=6, elliptical_acs=True)

    assert len(rectangle.calibration) == 8 * 6
    assert ellipse.calibration < rectangle.calibration
    assert (8, 4) in ellipse.calibration and (4, 1) not in ellipse.calibration


def test_the_wave_free_reference_shots_lead_and_are_marked_ref():
    built = app(
        CARTESIAN, wave_amplitude=8e-3, wave_cycles=2, ry=2, n_acs_y=4, n_acs_z=2
    )
    seq = built.design()
    ref, ima = labels(seq, "REF", "IMA")
    n_reference = sum(len(lines) for _, lines in built.reference)
    reads = [
        block
        for block in (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
        if block.adc is not None
    ]

    assert seq.check_timing()[0]
    assert list(ref) == [1] * n_reference + [0] * (len(ref) - n_reference)
    assert set(ima) == {0}
    assert all(block.gy.amplitude == 0 for block in reads[:n_reference])
    assert all(block.gy.amplitude != 0 for block in reads[n_reference:])


# -- stacks ---------------------------------------------------------------------


@pytest.mark.parametrize("name", [STARS, SPIRALS])
def test_every_angle_is_read_once_at_every_partition(name):
    built = app(name, rz=2, n_z=8, n_acs_z=2)
    lin, par, eco, ima = labels(built.design(), "LIN", "PAR", "ECO", "IMA")
    angles = getattr(built, "spokes", getattr(built, "arms", None))

    assert list(zip(lin, par, strict=True)) == [
        (a, z) for z in built.partitions for a in angles
    ]
    assert list(eco) == list(range(len(angles))) * len(built.partitions)
    assert list(ima) == [int(z in built.calibration) for z in par]


@pytest.mark.parametrize("n", [4, 8, 13, 201])
def test_the_golden_order_plays_every_angle_once_spread_from_the_start(n):
    order = module(STARS).golden_order(n)

    assert sorted(order) == list(range(n))
    assert order[0] == 0
    if n > 4:
        # The first quarter of a train already covers every quarter of the turn.
        assert len({4 * a // n for a in order[: n // 4 + 2]}) == 4


@pytest.mark.parametrize("name", [STARS, SPIRALS])
def test_a_partition_shift_turns_every_angle_by_the_partition(name):
    built = app(name, partition_angle_shift="golden")

    first, second = built.rotation(0, 0), built.rotation(0, 1)
    assert first is not second
    assert built.rotation(0, 0) is first


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        (CARTESIAN, "--ordering", "Line order within a partition"),
        (STARS, "--ti", "Inversion time (s), from the inversion pulse's centre"),
        (SPIRALS, "--n-shots", "Interleaves that sample the centre of each plane"),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module(name).main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert flag in printed
    assert help_text in printed
