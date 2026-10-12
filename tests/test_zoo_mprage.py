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

#: The definition each stack writes its number of angles in.
ANGLES = {STARS: "NumSpokes", SPIRALS: "NumArms"}

RASTER = pp.Opts().block_duration_raster


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def built(name, **kwargs):
    """The sequence ``name`` designs from its small prescription, without dummy shots."""
    return module(name).main(pp.Opts(), **{**SMALL[name], "n_dummy": 0, **kwargs})


def definition(seq, key):
    """The first value of a numeric definition."""
    return np.atleast_1d(seq.definitions[key])[0]


def labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def blocks(seq):
    return [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]


def pulse_times(seq, use):
    """Centre time (s) of every RF pulse of ``use``, in play order."""
    t, times = 0.0, []
    for block in blocks(seq):
        rf = getattr(block, "rf", None)
        if rf is not None and rf.use == use:
            times.append(t + rf.delay + rf.center)
        t += block.block_duration
    return np.asarray(times)


def shots_of(seq):
    """How many readout excitations each shot plays."""
    return int(definition(seq, "EchoTrainLength"))


def support(ry=1, rz=1, n_acs_y=24, n_acs_z=16, ordering="radial", **_):
    """The views of a Cartesian prescription: all, and the calibration region, from the sampling helper the sequence calls."""
    calibrating, imaging = pp.make_cartesian_plane_sampling(
        (SMALL[CARTESIAN]["n_y"], SMALL[CARTESIAN]["n_z"]),
        (ry, rz),
        (n_acs_y, n_acs_z),
        elliptical=True,
        sampling="poisson" if ordering == "shuffling" else "lattice",
        seed=module(CARTESIAN).SHUFFLE_SEED,
        train_length=train_length(ry),
    )
    return sorted({*calibrating, *imaging}), set(calibrating)


def train_length(ry=1):
    """Views per Cartesian train: as many as one line holds at this undersampling."""
    return -(-SMALL[CARTESIAN]["n_y"] // ry)


def shot_count(name, **kwargs):
    """The shots a prescription plays: a train of views each, or a partition each."""
    if name == CARTESIAN:
        return len(support(**kwargs)[0]) // train_length(kwargs.get("ry", 1))
    return len(partitions(name, **kwargs))


def partitions(name, **kwargs):
    """The partitions a stack prescription acquires, from the sampling helper the sequence calls."""
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        kwargs.get("n_z", SMALL[name]["n_z"]),
        kwargs.get("rz", 1),
        kwargs.get("n_acs_z", 16),
        partial_fourier=1.0,
    )
    return sorted({*calibrating, *imaging})


@pytest.mark.parametrize("name", SMALL)
def test_a_small_mprage_passes_its_timing_check_and_repeats_one_shot(name):
    seq = built(name, n_dummy=1)
    size, start = seq.repetition()

    assert seq.check_timing()[0]
    assert start == 1
    # One shot is the repeating unit, the dummy included.
    assert size * (1 + shot_count(name)) == len(seq.block_events)


@pytest.mark.parametrize("name", [STARS, SPIRALS])
def test_every_shot_reads_one_partition_in_order(name):
    seq = built(name)
    (par,) = labels(seq, "PAR")
    acquired = partitions(name)
    per_shot = [int(n) for n in np.bincount(par)[acquired]]

    assert sorted(set(par)) == acquired
    assert list(par) == sorted(par)
    assert max(per_shot) <= shots_of(seq)


@pytest.mark.parametrize("name", SMALL)
def test_a_train_read_into_one_image_leaves_the_echo_counter_alone(name):
    (eco,) = labels(built(name), "ECO")

    assert set(eco) == {0}


def test_a_shuffled_train_counts_its_place_as_the_time_it_resolves():
    seq = built(CARTESIAN, ordering="shuffling")
    (eco,) = labels(seq, "ECO")

    assert list(eco) == list(np.tile(np.arange(train_length()), shot_count(CARTESIAN)))


@pytest.mark.parametrize("name", SMALL)
def test_every_shot_plays_the_same_number_of_excitations(name):
    seq = built(name, n_dummy=2)
    inversions = pulse_times(seq, "inversion")
    excitations = pulse_times(seq, "excitation")

    assert len(inversions) == 2 + shot_count(name)
    assert len(excitations) == len(inversions) * shots_of(seq)


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("ti", [100e-3, 150e-3])
def test_the_first_excitation_of_a_shot_is_ti_after_its_inversion(name, ti):
    seq = built(name, ti=ti)
    inversions = pulse_times(seq, "inversion")
    excitations = pulse_times(seq, "excitation")[:: shots_of(seq)]

    assert excitations - inversions == pytest.approx(ti, abs=RASTER)


@pytest.mark.parametrize("name", SMALL)
def test_the_first_views_central_adc_sample_is_ti_plus_te_after_inversion(name):
    seq = built(name)
    sample_times, _ = seq.adc_times()
    first_centre = sample_times[int(definition(seq, "kSpaceCenterSample"))]
    inversion = pulse_times(seq, "inversion")[0]
    dwell = next(block.adc.dwell for block in blocks(seq) if block.adc is not None)

    assert first_centre - inversion == pytest.approx(
        definition(seq, "TI") + definition(seq, "TE"), abs=dwell / 2 + 1e-9
    )


@pytest.mark.parametrize("name", SMALL)
def test_inversions_are_one_tr_apart(name):
    seq = built(name, tr=800e-3, n_dummy=1)
    inversions = pulse_times(seq, "inversion")

    assert np.diff(inversions) == pytest.approx(800e-3, abs=1e-9)
    assert seq.duration()[0] == pytest.approx(800e-3 * len(inversions))


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize(
    "timing",
    [{"ti": 100.0033e-3, "tr": 800.0047e-3}, {"ti": None, "tr": None}],
    ids=["off the raster", "shortest"],
)
def test_the_ti_and_tr_written_are_the_intervals_the_inversions_play(name, timing):
    seq = built(name, n_dummy=1, **timing)
    inversions = pulse_times(seq, "inversion")
    excitations = pulse_times(seq, "excitation")[:: shots_of(seq)]
    ti, tr = definition(seq, "TI"), definition(seq, "TR")

    assert excitations - inversions == pytest.approx(ti, abs=1e-9)
    assert np.diff(inversions) == pytest.approx(tr, abs=1e-9)
    for requested, written in zip(timing.values(), (ti, tr), strict=True):
        if requested is not None:
            assert written == pytest.approx(requested, abs=RASTER)


@pytest.mark.parametrize("name", SMALL)
def test_navigators_ride_in_the_recovery_without_moving_the_inversions(name):
    seq = built(name, tr=1.5, navigator=True)
    inversions = pulse_times(seq, "inversion")

    # The navigators excite too, beyond the readout's own excitations.
    assert len(pulse_times(seq, "excitation")) > len(inversions) * shots_of(seq)
    assert np.diff(inversions) == pytest.approx(1.5, abs=1e-9)
    assert seq.check_timing()[0]


@pytest.mark.parametrize("name", SMALL)
def test_dummy_shots_acquire_nothing(name):
    plain, dummied = built(name), built(name, n_dummy=2)

    assert len(labels(dummied, "PAR")[0]) == len(labels(plain, "PAR")[0])


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
        built(name, **prescription)


# -- Cartesian ----------------------------------------------------------------


@pytest.mark.parametrize("ordering", ["radial", "shuffling"])
def test_every_sampled_view_is_read_once_inside_the_ellipse(ordering):
    prescription = {
        "ordering": ordering,
        "ry": 2,
        "rz": 2,
        "n_acs_y": 4,
        "n_acs_z": 2,
    }
    lin, par = labels(built(CARTESIAN, **prescription), "LIN", "PAR")
    sampled, calibration = support(**prescription)
    views = list(zip(lin, par, strict=True))
    outside = [
        (y, z)
        for y, z in views
        if ((y - 8) / 16) ** 2 + ((z - 4) / 8) ** 2 > 0.25 and (y, z) not in calibration
    ]

    assert len(views) == len(set(views))
    assert set(views) == set(sampled)
    assert outside == []
    assert calibration <= set(views)


def cartesian_trains(seq):
    """The views each Cartesian shot acquires, in the order it acquires them."""
    lin, par = labels(seq, "LIN", "PAR")
    views = list(zip(lin.tolist(), par.tolist(), strict=True))
    return [views[i : i + shots_of(seq)] for i in range(0, len(views), shots_of(seq))]


@pytest.mark.parametrize(
    "prescription",
    [
        {},
        {"ordering": "shuffling", "ry": 2, "rz": 2, "n_acs_y": 4, "n_acs_z": 2},
        {"ry": 3, "n_acs_y": 4, "n_acs_z": 2},
    ],
    ids=["radial", "shuffling", "ry=3"],
)
def test_every_shot_acquires_a_full_train_of_views(prescription):
    seq = built(CARTESIAN, **prescription)
    length = train_length(prescription.get("ry", 1))

    assert shots_of(seq) == length
    assert {len(train) for train in cartesian_trains(seq)} == {length}
    assert (
        len(pulse_times(seq, "excitation"))
        == shot_count(CARTESIAN, **prescription) * length
    )


def test_a_radial_shot_reads_its_views_centre_out():
    trains = cartesian_trains(built(CARTESIAN))

    def radius(view):
        return ((view[0] - 8) / 16) ** 2 + ((view[1] - 4) / 8) ** 2

    assert (8, 4) in [train[0] for train in trains]
    for train in trains:
        distance = [radius(view) for view in train]
        assert distance == sorted(distance)


def test_a_shuffled_shot_reads_the_sampled_views_in_another_order():
    radial = cartesian_trains(built(CARTESIAN))
    shuffled = cartesian_trains(built(CARTESIAN, ordering="shuffling", ry=1, rz=1))
    views = [view for train in shuffled for view in train]

    assert sorted(views) == support(ordering="shuffling")[0]
    assert shuffled != radial


def test_an_elliptical_calibration_region_is_the_inscribed_ellipse():
    def calibration(**kwargs):
        seq = built(CARTESIAN, ry=2, rz=2, n_acs_y=8, n_acs_z=6, **kwargs)
        lin, par, ima = labels(seq, "LIN", "PAR", "IMA")
        return {(y, z) for y, z, marked in zip(lin, par, ima, strict=True) if marked}

    rectangle, ellipse = calibration(), calibration(elliptical_acs=True)

    assert len(rectangle) == 8 * 6
    assert ellipse < rectangle
    assert (8, 4) in ellipse and (4, 1) not in ellipse


def test_the_wave_free_reference_shots_lead_and_are_marked_ref():
    prescription = {"ry": 2, "n_acs_y": 4, "n_acs_z": 2}
    seq = built(CARTESIAN, wave_amplitude=8e-3, wave_cycles=2, **prescription)
    ref, ima, lin, par = labels(seq, "REF", "IMA", "LIN", "PAR")
    _, calibration = support(**prescription)
    n_reference = len(calibration)
    reads = [block for block in blocks(seq) if block.adc is not None]

    assert seq.check_timing()[0]
    assert list(ref) == [1] * n_reference + [0] * (len(ref) - n_reference)
    assert set(zip(lin[:n_reference], par[:n_reference], strict=True)) == calibration
    assert set(ima) == {0}
    assert all(block.gy.amplitude == 0 for block in reads[:n_reference])
    assert all(block.gy.amplitude != 0 for block in reads[n_reference:])


# -- stacks ---------------------------------------------------------------------


@pytest.mark.parametrize("name", [STARS, SPIRALS])
def test_every_angle_is_read_once_at_every_partition(name):
    seq = built(name, rz=2, n_z=8, n_acs_z=2)
    lin, par, ima = labels(seq, "LIN", "PAR", "IMA")
    calibrating, imaging = pp.make_cartesian_axis_sampling(8, 2, 2, partial_fourier=1.0)
    acquired = sorted({*calibrating, *imaging})
    angles = module(name).golden_order(int(definition(seq, ANGLES[name])))

    assert list(zip(lin, par, strict=True)) == [
        (a, z) for z in acquired for a in angles
    ]
    assert list(ima) == [int(z in calibrating) for z in par]


@pytest.mark.parametrize("n", [4, 8, 13, 201])
def test_the_golden_order_plays_every_angle_once_spread_from_the_start(n):
    order = module(STARS).golden_order(n)

    assert sorted(order) == list(range(n))
    assert order[0] == 0
    if n > 4:
        # The first quarter of a train already covers every quarter of the turn.
        assert len({4 * a // n for a in order[: n // 4 + 2]}) == 4


@pytest.mark.parametrize("name", [STARS, SPIRALS])
@pytest.mark.parametrize("shift", ["none", "golden", "tiny_golden"])
def test_a_partition_shift_turns_every_angle_by_the_partition(name, shift):
    seq = built(name, partition_angle_shift=shift)
    lin, par = labels(seq, "LIN", "PAR")
    span = np.pi if name == STARS else 2 * np.pi
    n_angles = int(definition(seq, ANGLES[name]))
    expected = (
        span * lin / n_angles + par * module(name).PARTITION_SHIFTS[shift] * span
    ) % span
    quaternions = np.asarray(seq.libraries().rotations)
    row = np.asarray(seq.block_rotations())[
        [i for i, block in enumerate(blocks(seq)) if block.adc is not None]
    ]
    played = 2 * np.arctan2(quaternions[row - 1, 3], quaternions[row - 1, 0])

    assert np.all(row > 0)
    assert ((played - expected + span / 2) % span - span / 2) == pytest.approx(
        0.0, abs=1e-9
    )
    # One rotation per distinct angle.
    assert len(np.unique(np.round(quaternions, 9), axis=0)) == len(
        np.unique(np.round(expected % span, 9))
    )


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        (CARTESIAN, "--ordering", "Train order, and the support"),
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
