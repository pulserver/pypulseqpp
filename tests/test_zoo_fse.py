"""The 3D fast spin-echo example sequence: views, echo times, refocusing trains and shots."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

fse3d = sequences.fse3D_sequence

SMALL = {"n_x": 32, "n_y": 16, "n_z": 8, "etl": 8, "te": 20e-3, "tr": 300e-3}
N_Y, N_Z, ETL = SMALL["n_y"], SMALL["n_z"], SMALL["etl"]
RASTER = pp.Opts().block_duration_raster

#: Individually parameterized trains: short and fast at the periphery.
INDIVIDUAL = {"tr_periphery": 150e-3, "etl_periphery": 5}


def built(**kwargs):
    """The sequence the function designs from the small prescription."""
    return fse3d.main(pp.Opts(), **{**SMALL, **kwargs})


def definition(seq, key):
    """The first value of a numeric definition."""
    return np.atleast_1d(seq.definitions[key])[0]


def labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def blocks(seq):
    return [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]


def pulses(seq):
    """``(time, use, amplitude)`` of every RF pulse, in play order."""
    t, found = 0.0, []
    for block in blocks(seq):
        rf = getattr(block, "rf", None)
        if rf is not None:
            found.append((t + rf.delay + rf.center, rf.use, abs(rf.signal).max()))
        t += block.block_duration
    return found


def echo_times(seq):
    """Time (s) of each acquisition's echo sample after the excitation that opened its train."""
    centre = int(definition(seq, "kSpaceCenterSample"))
    t, excited, found = 0.0, 0.0, []
    for block in blocks(seq):
        rf = getattr(block, "rf", None)
        if rf is not None and rf.use == "excitation":
            excited = t + rf.delay + rf.center
        if block.adc is not None:
            found.append(t + block.adc.delay + centre * block.adc.dwell - excited)
        t += block.block_duration
    return np.asarray(found)


def trains(seq):
    """The acquisitions of each train as ``(line, partition, ECO, echo)``, in play order.

    ``echo`` counts the refocusing pulses played since the train's excitation,
    from 0. A train that acquires nothing is an empty list. The sequence has
    no navigators.
    """
    found = iter(zip(*labels(seq, "LIN", "PAR", "ECO"), strict=True))
    played, echo = [], -1
    for block in blocks(seq):
        rf = getattr(block, "rf", None)
        if rf is not None and rf.use == "excitation":
            played.append([])
            echo = -1
        if rf is not None and rf.use == "refocusing":
            echo += 1
        if block.adc is not None:
            played[-1].append((*(int(v) for v in next(found)), echo))
    return played


def refocusing_flips(seq):
    """Each train's refocusing angles (degrees) from the pulses played, ``(trains, ETL)``.

    A 180 degree pulse has the amplitude the pulses of a constant train play.
    """
    full = max(a for _, use, a in pulses(built()) if use == "refocusing")
    played = [a for _, use, a in pulses(seq) if use == "refocusing"]
    return 180.0 * np.reshape(played, (-1, ETL)) / full


def sampling(ry=1, rz=1, n_acs_y=24, n_acs_z=16, ordering="radial", **_):
    """The calibration and imaging views the prescription samples, from the helper the sequence calls."""
    return pp.make_cartesian_plane_sampling(
        (N_Y, N_Z),
        (ry, rz),
        (n_acs_y, n_acs_z),
        elliptical=True,
        sampling="poisson" if ordering == "shuffling" else "lattice",
        seed=fse3d.SHUFFLE_SEED,
    )


def radius(line, partition):
    return np.hypot((line - N_Y // 2) / N_Y, (partition - N_Z // 2) / N_Z)


def along_cubic(n, centre, periphery):
    """``n`` values moving from ``centre`` to ``periphery`` along ``3 u**2 - 2 u**3``."""
    place = np.arange(n) / max(n - 1, 1)
    return centre + (periphery - centre) * (3 * place**2 - 2 * place**3)


@pytest.mark.parametrize(
    "prescription",
    [{}, {"excitation": "nonselective", "te": None}, INDIVIDUAL],
    ids=["slab", "nonselective", "individual"],
)
def test_a_small_fse_passes_its_timing_check_and_repeats_one_train(prescription):
    seq = built(n_dummy=1, **prescription)
    size, start = seq.repetition()

    assert seq.check_timing()[0]
    assert start == 1
    assert size * (1 + int(definition(seq, "NumShots"))) == len(seq.block_events)


def test_none_uses_the_shortest_fse_repetition_time():
    seq = built(tr=None)
    shortest = definition(seq, "TR")
    excitations = [t for t, use, _ in pulses(seq) if use == "excitation"]

    assert seq.check_timing()[0]
    assert np.diff(excitations) == pytest.approx(shortest, abs=1e-9)
    with pytest.raises(ValueError, match="shorter than"):
        built(tr=shortest - RASTER)
    assert built(tr=shortest).duration()[0] == pytest.approx(seq.duration()[0])


@pytest.mark.parametrize(
    "prescription",
    [{}, {"ordering": "shuffling"}, INDIVIDUAL],
    ids=["radial", "shuffling", "individual"],
)
def test_a_tr_off_the_raster_is_the_tr_every_train_plays(prescription):
    requested = 300.0047e-3
    seq = built(tr=requested, n_dummy=1, **prescription)
    spacing = np.diff([t for t, use, _ in pulses(seq) if use == "excitation"])
    n = int(definition(seq, "NumShots"))
    periphery = prescription.get("tr_periphery", requested)
    times = along_cubic(n, requested, periphery)
    tr = definition(seq, "TR")

    assert tr == pytest.approx(requested, abs=RASTER)
    assert tr / RASTER == pytest.approx(round(tr / RASTER))
    # The dummy train plays the first shot's TR, at the centre of k-space.
    assert spacing[:2] == pytest.approx(tr, abs=1e-9)
    assert spacing[1:] == pytest.approx(times[:-1], abs=RASTER)
    assert seq.duration()[0] == pytest.approx(tr + sum(times), abs=n * RASTER)
    if "tr_periphery" in prescription:
        assert definition(seq, "TRPeriphery") == pytest.approx(periphery, abs=RASTER)
    else:
        assert "TRPeriphery" not in seq.definitions


def test_a_periphery_left_to_the_design_is_the_centre_tr_and_train_length():
    left = built(tr=300.0047e-3)
    named = built(tr=300.0047e-3, tr_periphery=300.0047e-3, etl_periphery=ETL)

    assert named.definitions == left.definitions
    assert [b.block_duration for b in blocks(named)] == [
        b.block_duration for b in blocks(left)
    ]
    assert "TRPeriphery" not in left.definitions
    assert definition(left, "EchoTrainLength") == ETL


@pytest.mark.parametrize(
    "prescription",
    [
        {},
        {"ordering": "shuffling", "ry": 2, "rz": 2, "n_acs_y": 4, "n_acs_z": 2},
        INDIVIDUAL,
    ],
    ids=["radial", "shuffling", "individual"],
)
def test_every_view_is_read_once_at_its_place_in_its_train(prescription):
    seq = built(**prescription)
    calibrating, imaging = sampling(**prescription)
    acquisitions = [a for train in trains(seq) for a in train]
    views = [(line, partition) for line, partition, _, _ in acquisitions]

    assert all(echo == place for _, _, echo, place in acquisitions)
    assert len(set(views)) == len(views)
    assert set(views) == {*calibrating, *imaging}
    for train in trains(seq):
        places = [place for *_, place in train]
        assert places == sorted(set(places))


def test_only_views_inside_the_inscribed_ellipse_are_read():
    lin, par = labels(built(), "LIN", "PAR")

    assert np.all(radius(lin, par) <= 0.5)
    assert len(lin) < N_Y * N_Z


@pytest.mark.parametrize("te", [None, 20e-3, 40e-3])
def test_the_centre_of_k_space_is_read_at_the_te_written(te):
    seq = built(te=te)
    written, esp = definition(seq, "TE"), definition(seq, "EchoSpacing")
    lin, par, eco = labels(seq, "LIN", "PAR", "ECO")
    (centre,) = np.flatnonzero((lin == N_Y // 2) & (par == N_Z // 2))
    times = echo_times(seq)

    assert times[centre] == pytest.approx(written, abs=1e-9)
    # Every echo is its place in the train away from the TE echo, by one spacing each.
    assert times == pytest.approx(written + esp * (eco - eco[centre]), abs=1e-9)
    if te is None:
        assert eco[centre] == 0
    else:
        assert abs(written - te) <= esp / 2


def test_radial_trains_read_further_out_the_further_the_echo_is_from_te():
    seq = built(te=40e-3)
    lin, par, eco = labels(seq, "LIN", "PAR", "ECO")
    (centre,) = np.flatnonzero((lin == N_Y // 2) & (par == N_Z // 2))
    by_echo = {}
    for line, partition, echo in zip(lin, par, eco, strict=True):
        by_echo.setdefault(abs(echo - eco[centre]), []).append(radius(line, partition))
    distances = sorted(by_echo)
    widest = [max(by_echo[d]) for d in distances]
    narrowest = [min(by_echo[d]) for d in distances]

    assert all(np.diff(narrowest) >= -1e-12)
    assert all(np.diff(widest) >= -1e-12)


# -- refocusing trains -----------------------------------------------------------


def test_constant_trains_refocus_at_the_angle_asked_for():
    seq = built(refocusing_angle_deg=140.0)
    refocusing = [a for _, use, a in pulses(seq) if use == "refocusing"]
    full = [
        a
        for _, use, a in pulses(built(refocusing_angle_deg=180.0))
        if use == "refocusing"
    ]

    assert np.asarray(refocusing) == pytest.approx(np.asarray(full) * 140.0 / 180.0)
    assert seq.definitions["RefocusingFlipAngles"] == pytest.approx([140.0] * ETL)


@pytest.fixture
def torchsim():
    return pytest.importorskip("torchsim")


@pytest.mark.parametrize("te", [None, 20e-3, 40e-3])
def test_an_optimized_train_passes_the_prescribed_angle_at_te(torchsim, te):
    seq = built(te=te, flip_modulation="optimized", refocusing_angle_deg=120.0)
    lin, par, eco = labels(seq, "LIN", "PAR", "ECO")
    (centre,) = np.flatnonzero((lin == N_Y // 2) & (par == N_Z // 2))
    flips = refocusing_flips(seq)[0]

    assert flips[eco[centre]] == pytest.approx(120.0)
    assert flips.min() >= min(fse3d.DESIGN_MIN_DEG, 60.0) - 1e-9
    assert flips.max() <= 180.0 + 1e-9
    assert not np.allclose(flips, 120.0)
    assert seq.definitions["RefocusingFlipAngles"] == pytest.approx(flips)


def test_optimized_individual_trains_are_silent_past_their_length(torchsim):
    seq = built(flip_modulation="optimized", refocusing_angle_deg=120.0, **INDIVIDUAL)
    flips = refocusing_flips(seq)
    lengths = np.round(along_cubic(len(flips), ETL, INDIVIDUAL["etl_periphery"]))
    played = np.arange(ETL)[None, :] < lengths[:, None]

    assert np.all(flips[~played] == 0.0)
    assert np.all(flips[played] > 0.0)
    assert seq.check_timing()[0]


# -- individually parameterized trains ---------------------------------------------


def test_every_shot_moves_from_the_centre_to_the_periphery_along_a_cubic():
    seq = built(**INDIVIDUAL)
    n = int(definition(seq, "NumShots"))
    lengths = np.count_nonzero(refocusing_flips(seq), axis=1)
    excitations = [t for t, use, _ in pulses(seq) if use == "excitation"]

    assert lengths[0] == 8 and lengths[-1] == 5
    assert list(lengths) == list(np.round(along_cubic(n, 8, 5)))
    assert np.diff(excitations) == pytest.approx(
        along_cubic(n, 300e-3, 150e-3)[:-1], abs=RASTER / 2 + 1e-9
    )
    assert definition(seq, "TRPeriphery") == pytest.approx(
        150e-3, abs=RASTER / 2 + 1e-9
    )
    assert sum(lengths) >= sum(len(train) for train in trains(seq))
    shorter = fse3d.shot_parameters(sum(lengths[:-1]) - 1, 8, 5, 300e-3, 150e-3)[0]
    assert len(shorter) < n


def test_the_centre_of_k_space_is_read_by_the_first_shot_at_the_te_echo():
    seq = built(**INDIVIDUAL)
    first = trains(seq)[0]
    lin, par = labels(seq, "LIN", "PAR")
    (centre,) = np.flatnonzero((lin == N_Y // 2) & (par == N_Z // 2))

    assert (N_Y // 2, N_Z // 2) in [(line, partition) for line, partition, *_ in first]
    assert echo_times(seq)[centre] == pytest.approx(definition(seq, "TE"), abs=1e-9)


def test_every_shot_plays_the_longest_train_and_waits_out_its_own_tr():
    seq = built(**INDIVIDUAL)
    found = pulses(seq)
    excitations = [t for t, use, _ in found if use == "excitation"]
    refocusing = [a for _, use, a in found if use == "refocusing"]
    nominal = max(refocusing)
    n = int(definition(seq, "NumShots"))
    per_shot = np.reshape(refocusing, (n, ETL))
    lengths = np.round(along_cubic(n, 8, 5))

    assert np.diff(excitations) == pytest.approx(
        along_cubic(n, 300e-3, 150e-3)[:-1], abs=RASTER / 2 + 1e-9
    )
    for amplitudes, length in zip(per_shot, lengths, strict=True):
        assert amplitudes[: int(length)] == pytest.approx(nominal)
        assert np.all(amplitudes[int(length) :] == 0.0)


def test_individual_trains_read_nothing_past_their_length():
    seq = built(**INDIVIDUAL)
    lengths = np.round(along_cubic(int(definition(seq, "NumShots")), 8, 5))

    for train, length in zip(trains(seq), lengths, strict=True):
        assert all(echo < length for *_, echo in train)


@pytest.mark.parametrize(
    ("prescription", "match"),
    [
        ({"ordering": "shuffling", **INDIVIDUAL}, "radial"),
        ({"etl_periphery": 2, "te": 40e-3}, "ends before echo"),
        ({"tr": 20e-3}, "shorter than"),
        ({"tr_periphery": 20e-3}, "shorter than"),
        ({"flip_modulation": "variable"}, "flip_modulation"),
        ({"excitation": "spsp"}, "excitation"),
    ],
    ids=[
        "shuffled",
        "short train",
        "short TR",
        "short periphery TR",
        "modulation",
        "excitation",
    ],
)
def test_an_fse_that_cannot_be_played_is_refused(prescription, match):
    with pytest.raises(ValueError, match=match):
        built(**prescription)


# -- extras ---------------------------------------------------------------------


def test_the_wave_free_reference_trains_lead_and_are_marked_ref():
    prescription = {"ry": 2, "n_acs_y": 4, "n_acs_z": 2}
    seq = built(wave_amplitude=8e-3, wave_cycles=2, **prescription)
    ref, ima, lin, par = labels(seq, "REF", "IMA", "LIN", "PAR")
    calibrating, _ = sampling(**prescription)
    n_reference = len(calibrating)
    reads = [block for block in blocks(seq) if block.adc is not None]

    assert seq.check_timing()[0]
    assert list(ref) == [1] * n_reference + [0] * (len(ref) - n_reference)
    assert set(zip(lin[:n_reference], par[:n_reference], strict=True)) == set(
        calibrating
    )
    assert set(ima) == {0}
    assert all(block.gy.amplitude == 0 for block in reads[:n_reference])
    assert all(block.gy.amplitude != 0 for block in reads[n_reference:])


def test_navigators_ride_after_the_train_without_moving_the_excitations():
    seq = built(tr=800e-3, navigator=True)
    found = pulses(seq)
    # The navigators excite too, at their own amplitude.
    train = found[0][2]
    excitations = [t for t, use, a in found if use == "excitation" and a == train]
    navigating = [t for t, use, a in found if use == "excitation" and a != train]

    assert navigating
    assert np.diff(excitations) == pytest.approx(800e-3, abs=1e-9)


def test_dummy_trains_acquire_nothing():
    plain, dummied = built(), built(n_dummy=2)

    assert len(labels(dummied, "LIN")[0]) == len(labels(plain, "LIN")[0])


def test_a_flag_is_named_and_described_by_the_function_it_runs(capsys):
    with pytest.raises(SystemExit):
        cli.run(fse3d.main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert "--etl-periphery" in printed
    assert "Echo train length at the periphery of k-space." in printed
