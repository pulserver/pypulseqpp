"""The 3D fast spin-echo example sequence: views, echo times, refocusing trains and shots."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

fse3d = sequences.fse3D_sequence

SMALL = {"n_x": 32, "n_y": 16, "n_z": 8, "etl": 8, "te": 20e-3, "tr": 300e-3}

#: Individually parameterized trains: short and fast at the periphery.
INDIVIDUAL = {"tr_periphery": 150e-3, "etl_periphery": 5}


def app(**kwargs):
    return fse3d.Fse3DApp(pp.Opts(), **{**SMALL, **kwargs})


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


def radius(view, built):
    n_y, n_z = built.matrix[1:]
    return np.hypot((view[0] - n_y // 2) / n_y, (view[1] - n_z // 2) / n_z)


@pytest.mark.parametrize(
    "prescription",
    [{}, {"excitation": "nonselective", "te": None}, INDIVIDUAL],
    ids=["slab", "nonselective", "individual"],
)
def test_a_small_fse_passes_its_timing_check_and_repeats_one_train(prescription):
    built = app(n_dummy=1, **prescription)
    seq = built.design()
    size, start = seq.repetition()

    assert seq.check_timing()[0]
    assert start == 1
    assert size * (1 + len(built.trains)) == len(seq.block_events)


def test_none_uses_the_shortest_fse_repetition_time():
    built = app(tr=None)
    seq = built.design()

    expected = built.fse.duration + built.system.block_duration_raster
    assert built.repetition_time == pytest.approx(expected)
    assert built.times == pytest.approx([expected] * len(built.trains))
    assert seq.check_timing()[0]


@pytest.mark.parametrize(
    "prescription",
    [{}, {"ordering": "shuffling"}, INDIVIDUAL],
    ids=["radial", "shuffling", "individual"],
)
def test_a_tr_off_the_raster_resolves_to_the_tr_every_train_plays(prescription):
    built = app(tr=300.0047e-3, n_dummy=1, **prescription)
    seq = built.design()
    spacing = np.diff([t for t, use, _ in pulses(seq) if use == "excitation"])
    resolved = built.resolved

    # The dummy train plays the first shot's TR, at the centre of k-space.
    assert spacing[:2] == pytest.approx(resolved["tr"], abs=1e-9)
    assert spacing[1:] == pytest.approx(built.times[:-1], abs=1e-9)
    assert resolved["tr_periphery"] == pytest.approx(built.times[-1])
    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(resolved["tr"])
    assert built.scan_time() == pytest.approx(seq.duration()[0], abs=1e-9)


def test_a_periphery_left_to_the_design_resolves_to_the_centre_tr_and_train_length():
    resolved = app(tr=300.0047e-3).resolved

    assert resolved["tr_periphery"] == resolved["tr"]
    assert resolved["etl_periphery"] == resolved["etl"] == 8


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
    built = app(**prescription)
    lin, par, eco = labels(built.design(), "LIN", "PAR", "ECO")
    expected = [
        (*view, echo)
        for train in built.trains
        for echo, view in enumerate(train)
        if view is not None
    ]
    views = [(y, z) for y, z, _ in expected]

    assert list(zip(lin, par, eco, strict=True)) == expected
    assert len(set(views)) == len(views)


def test_only_views_inside_the_inscribed_ellipse_are_read():
    built = app()
    views = [view for train in built.trains for view in train if view is not None]

    assert all(radius(view, built) <= 0.5 for view in views)
    assert len(views) < 16 * 8


@pytest.mark.parametrize("te", [None, 20e-3, 40e-3])
def test_the_centre_of_k_space_is_read_at_the_te_written(te):
    built = app(te=te)
    written = np.atleast_1d(built.design().definitions["TE"])[0]
    centre = (8, 4)
    echo = next(
        e
        for s, train in enumerate(built.trains)
        for e, view in enumerate(train)
        if view == centre
    )

    assert echo == built.te_echo
    assert written == pytest.approx(built.fse.echo_times[echo])
    if te is not None:
        assert abs(written - te) <= built.fse.esp / 2


def test_radial_trains_read_further_out_the_further_the_echo_is_from_te():
    built = app(te=40e-3)
    by_echo = {}
    for train in built.trains:
        for echo, view in enumerate(train):
            if view is not None:
                by_echo.setdefault(abs(echo - built.te_echo), []).append(
                    radius(view, built)
                )
    distances = sorted(by_echo)
    widest = [max(by_echo[d]) for d in distances]
    narrowest = [min(by_echo[d]) for d in distances]

    assert all(np.diff(narrowest) >= -1e-12)
    assert all(np.diff(widest) >= -1e-12)


# -- refocusing trains -----------------------------------------------------------


def test_constant_trains_refocus_at_the_angle_asked_for():
    built = app(refocusing_angle_deg=140.0)
    refocusing = [
        amplitude for _, use, amplitude in pulses(built.design()) if use == "refocusing"
    ]
    nominal = app(refocusing_angle_deg=180.0)
    full = [a for _, use, a in pulses(nominal.design()) if use == "refocusing"]

    assert np.asarray(refocusing) == pytest.approx(np.asarray(full) * 140.0 / 180.0)
    assert built.design().definitions["RefocusingFlipAngles"] == pytest.approx(
        [140.0] * 8
    )


@pytest.fixture
def torchsim():
    return pytest.importorskip("torchsim")


@pytest.mark.parametrize("te", [None, 20e-3, 40e-3])
def test_an_optimized_train_passes_the_prescribed_angle_at_te(torchsim, te):
    built = app(te=te, flip_modulation="optimized", refocusing_angle_deg=120.0)
    flips = built.flips[0]

    assert flips[built.te_echo] == pytest.approx(120.0)
    assert flips.min() >= min(built.DESIGN_MIN_DEG, 60.0) - 1e-9
    assert flips.max() <= 180.0 + 1e-9
    assert not np.allclose(flips, 120.0)


def test_optimized_individual_trains_are_silent_past_their_length(torchsim):
    built = app(flip_modulation="optimized", refocusing_angle_deg=120.0, **INDIVIDUAL)
    played = np.arange(8)[None, :] < np.asarray(built.lengths)[:, None]

    assert np.all(built.flips[~played] == 0.0)
    assert np.all(built.flips[played] > 0.0)
    assert built.design().check_timing()[0]


# -- individually parameterized trains ---------------------------------------------


def test_every_shot_moves_from_the_centre_to_the_periphery_along_a_cubic():
    built = app(**INDIVIDUAL)
    n = len(built.lengths)
    place = np.arange(n) / (n - 1)
    step = 3 * place**2 - 2 * place**3

    assert built.lengths[0] == 8 and built.lengths[-1] == 5
    assert built.lengths == [round(8 - 3 * c) for c in step]
    raster = built.system.block_duration_raster
    assert built.times == pytest.approx(300e-3 - 150e-3 * step, abs=raster / 2)
    assert sum(built.lengths) >= sum(
        view is not None for train in built.trains for view in train
    )
    shorter = fse3d.shot_parameters(sum(built.lengths[:-1]) - 1, 8, 5, 300e-3, 150e-3)[
        0
    ]
    assert len(shorter) < n


def test_the_centre_of_k_space_is_read_by_the_first_shot():
    built = app(**INDIVIDUAL)

    assert built.trains[0][built.te_echo] == (8, 4)


def test_every_shot_plays_the_longest_train_and_waits_out_its_own_tr():
    built = app(**INDIVIDUAL)
    found = pulses(built.design())
    excitations = [t for t, use, _ in found if use == "excitation"]
    refocusing = [a for _, use, a in found if use == "refocusing"]
    nominal = max(refocusing)
    per_shot = np.reshape(refocusing, (len(built.trains), 8))

    assert np.diff(excitations) == pytest.approx(built.times[:-1], abs=1e-9)
    for amplitudes, length in zip(per_shot, built.lengths, strict=True):
        assert amplitudes[:length] == pytest.approx(nominal)
        assert np.all(amplitudes[length:] == 0.0)


def test_individual_trains_read_nothing_past_their_length():
    built = app(**INDIVIDUAL)

    for train, length in zip(built.trains, built.lengths, strict=True):
        assert all(view is None for view in train[length:])


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
        app(**prescription)


# -- extras ---------------------------------------------------------------------


def test_the_wave_free_reference_trains_lead_and_are_marked_ref():
    built = app(wave_amplitude=8e-3, wave_cycles=2, ry=2, n_acs_y=4, n_acs_z=2)
    seq = built.design()
    ref, ima = labels(seq, "REF", "IMA")
    n_reference = sum(v is not None for train in built.reference for v in train)
    reads = [block for block in blocks(seq) if block.adc is not None]

    assert seq.check_timing()[0]
    assert list(ref) == [1] * n_reference + [0] * (len(ref) - n_reference)
    assert set(ima) == {0}
    assert all(block.gy.amplitude == 0 for block in reads[:n_reference])
    assert all(block.gy.amplitude != 0 for block in reads[n_reference:])


def test_navigators_ride_after_the_train_without_moving_the_excitations():
    built = app(tr=800e-3, navigator=True)
    seq = built.design()
    found = pulses(seq)
    # The navigators excite too, at their own amplitude.
    train = found[0][2]
    excitations = [t for t, use, a in found if use == "excitation" and a == train]

    assert built.n_navigators > 0
    assert np.diff(excitations) == pytest.approx(800e-3, abs=1e-9)


def test_dummy_trains_acquire_nothing():
    plain, dummied = app(), app(n_dummy=2)

    assert len(labels(dummied.design(), "LIN")[0]) == len(
        labels(plain.design(), "LIN")[0]
    )


def test_a_flag_is_named_and_described_by_the_function_it_runs(capsys):
    with pytest.raises(SystemExit):
        cli.run(fse3d.main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert "--etl-periphery" in printed
    assert "Echo train length at the periphery of k-space." in printed
