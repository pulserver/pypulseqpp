"""The 2D and 3D EPI zoo entries: linked prescans, encoding labels and prescription."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

epi2d = sequences.epi2D_sequence
epi3d = sequences.epi3D_sequence

SMALL_2D = {"n_x": 32, "n_y": 16, "n_dummy": 0}
SMALL_3D = {"n_x": 32, "n_y": 16, "n_z": 4, "n_dummy": 0}


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def played(seq, n_x, fov, grid):
    """Per axis after the readout, the grid index each line's echo was played at."""
    k = seq.calculate_kspace()[0][1:, n_x // 2 :: n_x]
    return [
        np.rint(k[axis] * fov[axis] + grid[axis] / 2).astype(int)
        for axis in range(len(fov))
    ]


def app2d(**kwargs):
    return epi2d.Epi2DApp(pp.Opts(), **{**SMALL_2D, **kwargs})


def app3d(**kwargs):
    return epi3d.Epi3DApp(pp.Opts(), **{**SMALL_3D, **kwargs})


def shots(seq, etl, *names):
    """Per acquiring shot, each label's values over its lines."""
    values = adc_labels(seq, *names)
    assert len(values[0]) % etl == 0
    return [
        [v[i : i + etl].tolist() for v in values] for i in range(0, len(values[0]), etl)
    ]


# -- the linked chain ------------------------------------------------------


@pytest.mark.parametrize(
    ("app", "files"),
    [
        (
            lambda: app2d(n_slices=2, acceleration=2, n_acs=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_navigator.seq", "epi_2d_navigator"),
                ("scan_main.seq", "epi_2d"),
            ],
        ),
        (
            lambda: app2d(n_dummy=1),
            [("scan.seq", "epi_2d_navigator"), ("scan_main.seq", "epi_2d")],
        ),
        (
            lambda: app2d(n_slices=4, n_bands=2, sms=True, n_acs=4),
            [("scan.seq", "sms_epi_2d_calibration"), ("scan_main.seq", "sms_epi_2d")],
        ),
        (
            lambda: app3d(acceleration=2, n_acs=4, n_acs_z=2),
            [
                ("scan.seq", "epi_3d_calibration"),
                ("scan_navigator.seq", "epi_3d_navigator"),
                ("scan_main.seq", "epi_3d"),
            ],
        ),
        (
            lambda: app3d(n_dummy=1),
            [("scan.seq", "epi_3d_navigator"), ("scan_main.seq", "epi_3d")],
        ),
    ],
    ids=["2D accelerated", "2D", "2D multiband", "3D undersampled", "3D"],
)
def test_the_chain_is_written_in_play_order_each_file_naming_the_next(
    tmp_path, app, files
):
    app = app()
    paths = app.write(tmp_path / "scan.seq")

    assert [p.rsplit("/", 1)[-1] for p in paths] == [name for name, _ in files]
    for i, path in enumerate(paths):
        seq = pp.Sequence(system=app.system)
        seq.read(path)
        assert seq.definitions["Name"] == files[i][1]
        following = files[i + 1][0] if i + 1 < len(files) else None
        assert seq.definitions.get("NextSequence") == following
        is_ok, errors = seq.check_timing()
        assert is_ok, errors


@pytest.mark.parametrize(
    "app",
    [
        lambda: app2d(n_slices=3, segments=2, n_dummy=2),
        lambda: app2d(n_slices=4, n_bands=2, sms=True, n_dummy=1),
        lambda: app3d(n_dummy=2),
    ],
    ids=["2D", "2D multiband", "3D"],
)
def test_the_main_sequence_repeats_from_its_first_block(app):
    _size, start = app().design()._detect_tr()

    assert start == 1


@pytest.mark.parametrize(
    "app",
    [
        lambda: app3d(n_y=24, n_z=8, acceleration=2, acceleration_z=2, segments=3),
        lambda: app2d(n_slices=4, n_bands=2, sms=True, n_acs=0, segments=2),
    ],
    ids=["3D caipi", "2D multiband"],
)
def test_a_written_segmented_caipi_scan_repeats_from_its_first_block(tmp_path, app):
    """Segments play trains of their own; the written file merges their equal events."""
    app = app()
    seq = pp.Sequence(system=app.system)
    seq.read(app.write(tmp_path / "scan.seq")[-1])

    assert len(set(map(id, app.trains))) == 2
    assert seq._detect_tr()[1] == 1


# -- 2D --------------------------------------------------------------------


def test_every_2d_imaging_line_carries_its_line_slice_repetition_and_polarity():
    app = app2d(n_slices=3, segments=2, acceleration=2, n_repetitions=2)
    found = shots(app.design(), app.epi.etl, "LIN", "SLC", "REP", "REV")
    offsets = app.epi.order[:, 0]

    assert [shot[:3] for shot in found] == [
        [(2 * segment + offsets).tolist(), [s] * len(offsets), [rep] * len(offsets)]
        for rep in range(2)
        for segment in range(2)
        for s in app.slices
    ]
    assert all(shot[3] == [i % 2 for i in range(app.epi.etl)] for shot in found)


def test_dummy_shots_are_played_once_ahead_of_the_2d_imaging():
    app = app2d(n_slices=2, n_dummy=2)
    seq = app.design()
    (once,) = adc_labels(seq, "ONCE")
    excitations = [i for i in seq.block_events if seq.get_block(i).rf is not None]
    first_adc = next(i for i in seq.block_events if seq.get_block(i).adc is not None)

    assert set(once) == {0}
    assert sum(i < first_adc for i in excitations) == 2 * 2 + 1


def test_an_accelerated_2d_scan_calibrates_each_slice_with_a_gradient_echo():
    app = app2d(n_slices=2, acceleration=2, n_acs=4)
    lin, slc, ref = adc_labels(app.design("calibration"), "LIN", "SLC", "REF")

    assert list(zip(lin, slc, strict=True)) == [
        (line, s) for s in app.calibration_slices for line in app.acs
    ]
    assert set(ref) == {1}
    assert "calibration" not in app2d(n_slices=2, acceleration=1).prescans()


def test_the_2d_navigator_is_blip_nulled_and_the_reference_train_reversed():
    app = app2d(n_slices=2)
    nav, ref, set_, rev, slc = adc_labels(
        app.design("navigator"), "NAV", "REF", "SET", "REV", "SLC"
    )

    assert rev[nav == 1].tolist() == [0, 1, 0] * 2
    assert set(ref[nav == 1]) == {1} and set(set_[nav == 1]) == {0}
    assert (set_ == 1).sum() == 2 * app.epi.etl
    assert slc[set_ == 1].tolist() == np.repeat(app.slices, app.epi.etl).tolist()

    def shot(kind):
        app.seq = pp.Sequence(app.system)
        app(0, None if kind == "navigator" else 0, kind)
        return [app.seq.get_block(i) for i in app.seq.block_events]

    image, reference, navigator = shot("image"), shot("reference"), shot("navigator")
    # Excitation, prewinder, then the lines: every phase-encode event negated.
    assert reference[1].gy.amplitude == pytest.approx(-image[1].gy.amplitude)
    line = reference[3].gy.waveform, image[3].gy.waveform
    assert np.allclose(line[0], -line[1]) and np.any(line[1])
    assert all(block.gy is None for block in navigator[1:-1])


@pytest.mark.parametrize("acceleration", [1, 2])
def test_every_2d_reference_line_is_labelled_with_the_line_it_encodes(acceleration):
    """The reference is placed by its labels, as the second SET of the scan."""
    app = app2d(acceleration=acceleration)
    seq = app.design("navigator")
    (set_, lin) = adc_labels(seq, "SET", "LIN")
    (ky,) = played(seq, 32, (0.22,), (16,))

    assert np.array_equal(lin[set_ == 1], ky[set_ == 1] % 16)
    assert len(set(lin[set_ == 1])) == app.epi.etl


def test_a_multiband_shot_encodes_its_group_and_band_phase():
    app = app2d(n_slices=4, n_bands=2, sms=True, n_acs=4)
    seq = app.design()
    found = shots(seq, app.epi.etl, "LIN", "PAR", "SLC", "SMS")

    assert [shot[2][0] for shot in found] == app.slices
    for shot in found:
        assert shot[0] == app.epi.order[:, 0].tolist()
        assert shot[1] == app.epi.order[:, 1].tolist()
        assert set(shot[3]) == {1}
    assert seq.definitions["MultibandFactor"] == [2.0]


def test_a_segmented_multiband_line_carries_the_band_phase_of_its_line():
    """The reconstruction models band j's phase on line ky as 2 pi ky j / n_bands."""
    n_bands, slice_step = 2, 5e-3
    app = app2d(n_slices=4, n_bands=n_bands, sms=True, n_acs=0, segments=2)
    seq = app.design()
    lin, par = adc_labels(seq, "LIN", "PAR")
    fov_z = n_bands * (4 // n_bands) * slice_step
    ky, kz = played(seq, 32, (0.22, fov_z), (16, 0))

    assert np.array_equal(par % n_bands, lin % n_bands)
    assert np.array_equal(kz % n_bands, par % n_bands)
    assert np.array_equal(ky, lin)


def test_multiband_imaging_starts_after_its_dummy_shots():
    app = app2d(n_slices=4, n_bands=2, sms=True, n_dummy=2)
    seq = app.design()
    excitations = [i for i in seq.block_events if seq.get_block(i).rf is not None]
    first_adc = next(i for i in seq.block_events if seq.get_block(i).adc is not None)

    # Two dummies for each of the two groups, then the first imaging shot.
    assert sum(i < first_adc for i in excitations) == 2 * 2 + 1


def test_the_multiband_calibration_navigates_then_calibrates_every_slice():
    app = app2d(n_slices=4, n_bands=2, sms=True, n_acs=4)
    nav, ref, slc, lin = adc_labels(
        app.design("calibration"), "NAV", "REF", "SLC", "LIN"
    )

    assert nav.tolist()[:3] == [1, 1, 1] and set(ref[:3]) == {0}
    assert list(zip(slc[3:], lin[3:], strict=True)) == [
        (s, line) for s in app.calibration_slices for line in app.acs
    ]
    assert set(ref[3:]) == {1} and set(nav[3:]) == {0}


def test_a_slice_count_the_multiband_factor_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="multiple of the multiband factor"):
        app2d(n_slices=3, n_bands=2, sms=True)


def test_fat_is_saturated_before_every_dummy_and_imaging_shot():
    kwargs = {**SMALL_2D, "n_slices": 2, "n_dummy": 1, "n_repetitions": 2}

    def pulses(seq):
        return sum(seq.get_block(i).rf is not None for i in seq.block_events)

    plain = pulses(epi2d(**kwargs))
    saturated = pulses(epi2d(**kwargs, fat_saturation=True))

    # Two slices: one dummy shot each, and one imaging shot per repetition.
    assert saturated - plain == 2 * 1 + 2 * 2


def test_the_2d_definitions_describe_the_train():
    app = app2d(n_slices=3)
    definitions = app.design().definitions

    assert definitions["Matrix"] == [32.0, 16.0, 3.0]
    assert definitions["EPIFactor"] == [float(app.epi.etl)]
    assert definitions["EchoSpacing"] == pytest.approx([app.epi.esp])
    assert definitions["NumGainCalibrationReadouts"] == [3.0]
    assert definitions["TE"][0] > app.epi.echo_time


def test_a_2d_echo_time_shorter_than_the_train_admits_is_refused():
    with pytest.raises(ValueError):
        app2d(te=1e-4)


# -- 3D --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "n_z": 8,
            "segments": 2,
            "acceleration": 2,
            "acceleration_z": 2,
            "partial_fourier": 0.75,
            "n_dummy": 1,
        },
        {"spsp": True, "te": 40e-3, "tr": 80e-3},
    ],
    ids=["segmented caipi partial-fourier", "water-selective"],
)
def test_a_3d_epi_passes_its_timing_check(kwargs):
    is_ok, errors = epi3d(**{**SMALL_3D, **kwargs}).check_timing()

    assert is_ok, errors


def test_every_3d_imaging_line_carries_its_line_and_partition():
    app = app3d(n_z=8, acceleration=2, acceleration_z=2, n_repetitions=2)
    found = shots(app.design(), app.epi.etl, "LIN", "PAR", "REP", "REV")
    order = app.epi.order

    assert [shot[:3] for shot in found] == [
        [order[:, 0].tolist(), (2 * shell + order[:, 1]).tolist(), [rep] * app.epi.etl]
        for rep in range(2)
        for shell in app.shells
    ]
    assert set(order[:, 1]) == {0, 1}  # the CAIPI sawtooth walks the shell
    assert all(shot[3] == [i % 2 for i in range(app.epi.etl)] for shot in found)


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_segmented_caipi_acquires_the_caipirinha_lattice(segments):
    n_y, n_z, shift = 24, 8, 1
    app = app3d(n_y=n_y, n_z=n_z, acceleration=2, acceleration_z=2, segments=segments)
    seq = app.design()
    lin, par = adc_labels(seq, "LIN", "PAR")
    ky, kz = played(seq, 32, (0.22, 0.096), (n_y, n_z))
    lattice = {
        (y, z)
        for y in range(0, n_y, 2)
        for z in range(n_z)
        if (z - (y // 2) * shift) % 2 == 0
    }

    assert np.array_equal(ky, lin) and np.array_equal(kz, par)
    assert set(zip(lin.tolist(), par.tolist(), strict=True)) == lattice
    assert len(lin) == len(lattice)


def test_the_first_shells_encoded_are_the_central_ones():
    app = app3d(n_z=8)
    distance = [abs(shell - 3.5) for shell in app.shells]

    assert sorted(app.shells) == list(range(8))
    assert distance == sorted(distance)


def test_partial_fourier_drops_the_leading_lines_and_shells_but_not_the_reference():
    app = app3d(n_z=8, partial_fourier=0.75, partial_fourier_z=0.5, n_acs=0, n_acs_z=0)
    lin, par = adc_labels(app.design(), "LIN", "PAR")
    (set_,) = adc_labels(app.design("navigator"), "SET")

    assert app.epi.etl < app.ref_epi.etl == 16
    assert lin.min() == app.first_y > 0
    assert sorted(set(par)) == [4, 5, 6, 7]
    assert (set_ == 1).sum() == app.ref_epi.etl


def test_an_undersampled_3d_scan_calibrates_over_the_central_rectangle():
    app = app3d(acceleration=2, n_acs=4, n_acs_z=2)
    lin, par, ref = adc_labels(app.design("calibration"), "LIN", "PAR", "REF")

    assert list(zip(lin, par, strict=True)) == [
        (y, z) for z in (1, 2) for y in (6, 7, 8, 9)
    ]
    assert set(ref) == {1}
    assert "calibration" not in app3d(n_acs=4, n_acs_z=2).prescans()


@pytest.mark.parametrize("acceleration_z", [1, 2])
def test_every_3d_reference_line_is_labelled_with_the_view_it_encodes(acceleration_z):
    app = app3d(n_z=8, acceleration_z=acceleration_z)
    seq = app.design("navigator")
    nav, set_, lin, par, rev = adc_labels(seq, "NAV", "SET", "LIN", "PAR", "REV")
    ky, kz = played(seq, 32, (0.22, 0.096), (16, 8))
    reference = set_ == 1

    assert rev[nav == 1].tolist() == [0, 1, 0]
    assert np.array_equal(lin[reference], ky[reference] % 16)
    assert np.array_equal(par[reference], kz[reference])


def test_a_3d_repetition_shorter_than_the_train_is_refused():
    with pytest.raises(ValueError):
        app3d(tr=1e-3)


# -- the command line ------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "flag", "help_text"),
    [
        (epi2d, "--segments", "Interleaved shots the train is split into."),
        (epi3d, "--acceleration-z", "Partition undersampling factor along z."),
    ],
    ids=["2D", "3D"],
)
def test_an_epi_flag_is_named_and_described_by_its_docstring(
    capsys, module, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module.main, ["--help"])

    printed = capsys.readouterr().out

    assert flag in printed
    assert help_text in printed
