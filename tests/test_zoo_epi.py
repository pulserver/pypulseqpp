"""The 2D and 3D EPI example sequences: linked prescans, sampling, timing and labels."""

from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

epi2d = sequences.epi2D_sequence
epi3d = sequences.epi3D_sequence

SMALL_2D = {"n_x": 32, "n_y": 16, "n_dummy": 0}
SMALL_3D = {"n_x": 32, "n_y": 16, "n_z": 8, "n_dummy": 0}


def app2d(**kwargs):
    return epi2d.Epi2DApp(pp.Opts(), **{**SMALL_2D, **kwargs})


def app3d(**kwargs):
    return epi3d.Epi3DApp(pp.Opts(), **{**SMALL_3D, **kwargs})


#: One prescription per case the shared tests run on.
CASES = {
    "2D": lambda **kw: app2d(**kw),
    "2D segmented": lambda **kw: app2d(n_y=32, ry=2, n_shots=3, **kw),
    "2D multiband": lambda **kw: app2d(n_slices=4, multiband=2, n_shots=2, **kw),
    "3D": lambda **kw: app3d(**kw),
    "3D skipped-CAIPI": lambda **kw: app3d(ry=2, rz=2, n_shots=2, **kw),
}


def labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def echoes(seq, n_samples):
    """``(ky, kz, echo time)`` of every acquired line, at its read-axis centre."""
    k, _, t_excitation, _, t_adc = seq.calculate_kspace()
    t_excitation = np.asarray(t_excitation)
    kx, t = k[0].reshape(-1, n_samples), np.asarray(t_adc).reshape(-1, n_samples)
    ky, kz = k[1].reshape(-1, n_samples), k[2].reshape(-1, n_samples)
    rows = []
    for r in range(len(kx)):
        i = int(np.argmin(np.abs(kx[r])))
        excited = t_excitation[t_excitation <= t[r, i]].max()
        rows.append((ky[r, i], kz[r, i], t[r, i] - excited))
    return np.array(rows)


def samples(app):
    return int(app.epi.adc.num_samples)


def partition_fov(app):
    """The field of view the partition (band) encode is counted over, and its size."""
    if hasattr(app, "shells"):
        return app.fov[2], app.matrix[2]
    band_spacing = len(app.centers) * (app.positions[1] - app.positions[0])
    return app.multiband * band_spacing, app.multiband


def shots_per_volume(app):
    if hasattr(app, "volume"):
        return len(app.volume)
    return app.n_shots * sum(map(len, app.packets))


def excitations(seq):
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return sum(block.rf is not None for block in blocks)


# -- the linked chain ------------------------------------------------------


@pytest.mark.parametrize(
    ("app", "files"),
    [
        (
            lambda: app2d(n_slices=2, ry=2, n_acs_y=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_reference.seq", "epi_2d_reference"),
                ("scan_main.seq", "epi_2d"),
            ],
        ),
        (
            lambda: app2d(n_dummy=1),
            [("scan.seq", "epi_2d_reference"), ("scan_main.seq", "epi_2d")],
        ),
        (
            lambda: app2d(n_slices=4, multiband=2, n_acs_y=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_reference.seq", "epi_2d_reference"),
                ("scan_main.seq", "epi_2d"),
            ],
        ),
        (
            lambda: app3d(ry=2, n_acs_y=4, n_acs_z=2),
            [
                ("scan.seq", "epi_3d_calibration"),
                ("scan_reference.seq", "epi_3d_reference"),
                ("scan_main.seq", "epi_3d"),
            ],
        ),
        (
            lambda: app3d(n_dummy=1),
            [("scan.seq", "epi_3d_reference"), ("scan_main.seq", "epi_3d")],
        ),
    ],
    ids=["2D undersampled", "2D", "2D multiband", "3D undersampled", "3D"],
)
def test_the_chain_is_written_in_play_order_each_file_naming_the_next(
    tmp_path, app, files
):
    app = app()
    paths = app.write(tmp_path / "scan.seq")

    assert [Path(p).name for p in paths] == [name for name, _ in files]
    for i, path in enumerate(paths):
        seq = pp.Sequence(system=app.system)
        seq.read(path)
        assert seq.definitions["Name"] == files[i][1]
        following = files[i + 1][0] if i + 1 < len(files) else None
        assert seq.definitions.get("NextSequence") == following
        is_ok, errors = seq.check_timing()
        assert is_ok, errors


@pytest.mark.parametrize("case", CASES)
def test_every_file_of_the_chain_repeats_from_its_first_block(tmp_path, case):
    app = CASES[case](n_dummy=2, n_frames=2, volume_output=True, n_acs_y=4)
    for path in app.write(tmp_path / "scan.seq"):
        seq = pp.Sequence(system=app.system)
        seq.read(path)
        assert seq._detect_tr()[1] == 1, path


# -- what every shot samples ------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_every_line_is_labelled_with_the_view_it_samples(case):
    app = CASES[case]()
    seq = app.design()
    rows = echoes(seq, samples(app))
    lin, par, nav = labels(seq, "LIN", "PAR", "NAV")
    imaging = nav == 0
    n_y = app.matrix[1]

    assert rows[imaging, 0] == pytest.approx((lin[imaging] - n_y // 2) / app.fov[1])
    if case != "2D" and case != "2D segmented":
        fov, n = partition_fov(app)
        assert rows[imaging, 1] == pytest.approx((par[imaging] - n // 2) / fov)


@pytest.mark.parametrize("case", CASES)
def test_every_shot_opens_with_its_navigator_at_the_centre_of_k_space(case):
    app = CASES[case]()
    seq = app.design()
    rows = echoes(seq, samples(app))
    nav, rev = labels(seq, "NAV", "REV")
    per_shot = app.NAVIGATOR_LINES + app.epi.etl
    shots = len(nav) // per_shot

    assert list(nav) == ([1] * app.NAVIGATOR_LINES + [0] * app.epi.etl) * shots
    assert list(rev) == [i % 2 for i in range(per_shot)] * shots
    assert rows[nav == 1, :2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("te", [None, 25e-3], ids=["shortest", "25 ms"])
def test_the_centre_line_is_echoed_at_the_echo_time_written(case, te):
    app = CASES[case](te=te)
    seq = app.design()
    rows = echoes(seq, samples(app))
    lin, nav = labels(seq, "LIN", "NAV")
    centre = (nav == 0) & (lin == app.matrix[1] // 2)
    written = np.atleast_1d(seq.definitions["TE"])[0]
    half_dwell = 0.5 * float(app.epi.adc.dwell)

    assert centre.any()
    assert rows[centre, 2] == pytest.approx(written, abs=half_dwell + 1e-9)
    if te is not None:
        assert written == pytest.approx(te, abs=app.system.block_duration_raster)


@pytest.mark.parametrize("case", ["2D segmented", "2D multiband", "3D skipped-CAIPI"])
def test_shifted_echoes_grow_evenly_across_the_lattice_lines(case):
    """Each shot is delayed by its share of the echo spacing."""
    app = CASES[case]()
    seq = app.design()
    rows = echoes(seq, samples(app))
    lin, nav = labels(seq, "LIN", "NAV")
    first = {}
    for line, (_, _, t) in zip(lin[nav == 0], rows[nav == 0], strict=True):
        first.setdefault(int(line), t)
    lines = sorted(first)
    steps = np.diff([first[line] for line in lines])
    raster = app.system.block_duration_raster

    assert app.n_shots > 1
    assert steps == pytest.approx(app.epi.esp / app.n_shots, abs=raster + 1e-9)


@pytest.mark.parametrize("case", CASES)
def test_the_reference_reverses_the_phase_encode_and_keeps_the_labels(case):
    app = CASES[case]()
    forward, reversed_ = app.design(), app.design("reference")
    n = samples(app)
    ahead, behind = echoes(forward, n), echoes(reversed_, n)
    lin, nav = labels(forward, "LIN", "NAV")
    lin_r, nav_r, set_r = labels(reversed_, "LIN", "NAV", "SET")

    assert list(lin_r) == list(lin)
    assert set(set_r) == {1}
    assert behind[nav_r == 0, 0] == pytest.approx(-ahead[nav == 0, 0])
    assert behind[:, 1] == pytest.approx(ahead[:, 1])


# -- volumes, dummies and the output ----------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_every_volume_is_marked_by_a_digital_output_dummies_included(case):
    app = CASES[case](n_frames=3, n_dummy=2, volume_output=True)
    seq = app.design()
    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    starts, t = [], 0.0
    for block in blocks:
        if getattr(block, "trig", None):
            assert block.rf is not None
            assert [trigger.channel for trigger in block.trig] == ["ext1"]
            starts.append(t)
        t += block.block_duration
    written = np.atleast_1d(seq.definitions["TR"])[0]

    assert len(starts) == 3 + 2
    assert np.diff(starts) == pytest.approx(written)


@pytest.mark.parametrize("case", CASES)
def test_a_time_series_labels_its_volumes_after_its_dummy_volumes(case):
    app = CASES[case](n_frames=3, n_dummy=2)
    seq = app.design()
    (rep,) = labels(seq, "REP")

    assert np.bincount(rep).tolist() == [len(rep) // 3] * 3
    assert excitations(seq) == (3 + 2) * shots_per_volume(app)


def test_a_single_2d_volume_plays_its_dummies_per_slice():
    app = app2d(n_slices=3, n_shots=2, n_dummy=2)

    assert excitations(app.design()) == 3 * (2 + 2)


def test_a_single_3d_volume_plays_its_dummies_as_shots():
    app = app3d(rz=2, n_dummy=3)

    assert excitations(app.design()) == 3 + len(app.volume)


@pytest.mark.parametrize("case", CASES)
def test_the_repetition_time_is_the_volume_asked_for(case):
    shortest = CASES[case]().repetition_time
    app = CASES[case](tr=2 * shortest)
    # Each shot's pad is rounded onto the block raster.
    tolerance = shots_per_volume(app) * app.system.block_duration_raster

    assert app.repetition_time == pytest.approx(2 * shortest, abs=tolerance)
    assert app.design().check_timing()[0]


# -- 2D slices and bands ----------------------------------------------------


def test_the_even_slices_of_a_packet_are_excited_before_the_odd_ones():
    assert app2d(n_slices=6).packets == [[0, 2, 4, 1, 3, 5]]
    assert app2d(n_slices=8, multiband=2).packets == [[0, 2, 1, 3]]


def test_a_single_volume_deals_slices_one_tr_cannot_hold_into_packets():
    app = app2d(n_slices=6, tr=2 * app2d().repetition_time)

    assert len(app.packets) == 3
    assert app.design().check_timing()[0]


def test_a_time_series_whose_tr_cannot_hold_every_slice_is_refused():
    with pytest.raises(ValueError, match="cannot hold"):
        app2d(n_slices=6, n_frames=2, tr=2 * app2d().repetition_time)


def test_the_slice_timing_is_each_slices_excitation_within_the_volume():
    app = app2d(n_slices=4, multiband=2)
    written = app.design().definitions["SliceTiming"]
    shot = app.shot_duration

    # Groups 0 and 1 are excited in that order; slice k belongs to group k % 2.
    assert written == pytest.approx([0.0, shot, 0.0, shot])


@pytest.mark.parametrize(("ry", "multiband"), [(1, 2), (2, 3), (1, 4)])
def test_a_multiband_train_walks_the_caipi_lattice_from_the_centre_band(ry, multiband):
    app = app2d(n_slices=2 * multiband, multiband=multiband, ry=ry, n_acs_y=4)
    lin, par, nav = labels(app.design(), "LIN", "PAR", "NAV")
    shift = epi2d.caipi_shift(ry, multiband)
    lin, par = lin[nav == 0], par[nav == 0]

    assert list(par) == list((multiband // 2 + shift * ((lin - 8) // ry)) % multiband)


def test_a_multiband_shot_is_centred_on_its_slice_group():
    app = app2d(n_slices=6, slice_spacing=1e-3, multiband=3)
    seq = app.design()
    offsets = sorted(
        {
            round(seq.get_block(i).rf.freq_offset, 6)
            for i in range(1, len(seq.block_events) + 1)
            if seq.get_block(i).rf is not None
        }
    )
    expected = sorted(round(app.selection_amplitude * c, 6) for c in app.centers)

    assert offsets == expected
    assert app.centers == pytest.approx(
        [np.mean(app.positions[g::2]) for g in range(2)]
    )


def test_an_undersampled_2d_scan_calibrates_every_slice_with_a_gradient_echo():
    app = app2d(n_slices=3, ry=2, n_acs_y=4)
    seq = app.design("calibration")
    lin, slc, ref = labels(seq, "LIN", "SLC", "REF")

    assert list(zip(slc, lin, strict=True)) == [
        (s, line) for s in (0, 2, 1) for line in range(6, 10)
    ]
    assert set(ref) == {1}


@pytest.mark.parametrize(
    "prescription",
    [
        {"partial_fourier_y": 0.4},
        {"n_slices": 3, "multiband": 2},
        {"n_shots": 0},
        {"te": 1e-3},
    ],
    ids=str,
)
def test_an_impossible_2d_prescription_is_refused(prescription):
    with pytest.raises(ValueError):
        app2d(**prescription)


# -- 3D shells --------------------------------------------------------------


@pytest.mark.parametrize(
    ("ry", "rz", "shift"), [(1, 1, 0), (1, 2, 1), (1, 3, 1), (1, 4, 2), (2, 2, 1)]
)
def test_the_caipi_shift_keeps_the_aliases_furthest_apart(ry, rz, shift):
    assert epi3d.caipi_shift(ry, rz) == shift


@pytest.mark.parametrize(
    ("ry", "rz", "n_shots"), [(1, 1, 1), (2, 2, 1), (2, 2, 2), (1, 3, 2), (2, 4, 3)]
)
def test_every_lattice_view_is_read_once_per_volume(ry, rz, n_shots):
    app = app3d(n_y=24, n_z=12, ry=ry, rz=rz, n_shots=n_shots)
    lin, par, nav = labels(app.design(), "LIN", "PAR", "NAV")
    views = list(zip(lin[nav == 0], par[nav == 0], strict=True))
    shift = app.shift
    lattice = {
        (y, z)
        for y in range(24)
        if (y - 12) % ry == 0
        for z in range(12)
        if (z - 6 - shift * ((y - 12) // ry)) % rz == 0
    }

    assert len(views) == len(set(views))
    assert set(views) <= lattice
    assert (12, 6) in views
    covered = {(y, z) for y, z in lattice if min(app.shells) <= z}
    assert len(covered - set(views)) < n_shots * 24 // ry


def test_a_shell_is_read_whole_before_the_next_and_the_centre_shell_first():
    app = app3d(n_z=12, rz=3)

    assert app.shells[0] <= 6 < app.shells[0] + 3
    assert sorted(app.shells) == [0, 3, 6, 9]


def test_shots_that_climb_a_whole_cycle_stay_in_their_partition():
    """Shot-selective CAIPI: the shift times the shot count is a multiple of rz."""
    app = app3d(ry=1, rz=2, n_shots=2)
    par, nav = labels(app.design(), "PAR", "NAV")
    etl = app.epi.etl
    shots = par[nav == 0].reshape(-1, etl)

    assert app.shift * app.n_shots % 2 == 0
    assert all(len(set(shot)) == 1 for shot in shots)


def test_one_shot_per_shell_is_blipped_caipi():
    app = app3d(ry=1, rz=2)
    par, nav = labels(app.design(), "PAR", "NAV")
    shots = par[nav == 0].reshape(-1, app.epi.etl)

    assert all(len(set(shot)) == 2 for shot in shots)


def test_partial_fourier_drops_the_leading_lines_and_shells():
    full = app3d(n_z=8, rz=2)
    partial = app3d(n_z=8, rz=2, partial_fourier_y=0.75, partial_fourier_z=0.5)

    assert len(partial.shells) == 2 and sorted(partial.shells) == [4, 6]
    assert partial.epi.etl == 12 and full.epi.etl == 16
    assert partial.echo_time < full.echo_time


def test_an_undersampled_3d_scan_calibrates_over_the_central_rectangle():
    app = app3d(ry=2, n_acs_y=4, n_acs_z=2)
    lin, par, ref = labels(app.design("calibration"), "LIN", "PAR", "REF")

    assert list(zip(lin, par, strict=True)) == [
        (y, z) for y in range(6, 10) for z in range(3, 5)
    ]
    assert set(ref) == {1}


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
def test_every_excitation_builds_a_3d_epi_that_passes_its_timing_check(excitation):
    seq = app3d(excitation=excitation, rz=2).design()

    assert seq.check_timing()[0]
    assert seq.definitions["Excitation"] == excitation


@pytest.mark.parametrize(
    "prescription",
    [
        {"partial_fourier_z": 0.4},
        {"excitation": "adiabatic"},
        {"rz": 0},
        {"rz": 16},
        {"tr": 1e-3},
    ],
    ids=str,
)
def test_an_impossible_3d_prescription_is_refused(prescription):
    with pytest.raises(ValueError):
        app3d(**prescription)


@pytest.mark.parametrize(
    ("module", "flag", "help_text"),
    [
        (epi2d, "--multiband", "Slices excited at once."),
        (epi3d, "--volume-output", "Play a digital output on"),
    ],
)
def test_an_epi_flag_is_named_and_described_by_its_docstring(
    capsys, module, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module.main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert flag in printed
    assert help_text in printed
