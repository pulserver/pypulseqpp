"""The 2D and 3D EPI zoo entries: prescan, encoding labels and prescription."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

epi2d = importlib.import_module("pypulseqpp.sequences.sequence.epi2D_sequence")
epi3d = importlib.import_module("pypulseqpp.sequences.sequence.epi3D_sequence")

SMALL_2D = {"n_x": 32, "n_y": 16, "n_dummy": 0}
SMALL_3D = {"n_x": 32, "n_y": 16, "n_z": 4, "n_dummy": 0}


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def app2d(**kwargs):
    return epi2d.Epi2DApp(pp.Opts(), **{**SMALL_2D, **kwargs})


def app3d(**kwargs):
    return epi3d.Epi3DApp(pp.Opts(), **{**SMALL_3D, **kwargs})


def imaging_shots(seq, etl, *names):
    """Per imaging shot (``ONCE = 0``), each label's values over its lines."""
    once, *values = adc_labels(seq, "ONCE", *names)
    imaging = np.flatnonzero(once == 0)
    assert len(imaging) % etl == 0
    return [
        [v[imaging[i : i + etl]].tolist() for v in values]
        for i in range(0, len(imaging), etl)
    ]


# -- 2D --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_slices": 2, "segments": 2, "acceleration": 2, "n_acs": 4, "n_dummy": 1},
        {
            "n_slices": 4,
            "n_bands": 2,
            "sms": True,
            "n_acs": 4,
            "te": 30e-3,
            "tr": 60e-3,
        },
    ],
    ids=["single-shot", "segmented accelerated", "multiband"],
)
def test_a_small_2d_epi_passes_its_timing_check(kwargs):
    is_ok, errors = epi2d.main(**{**SMALL_2D, **kwargs}).check_timing()

    assert is_ok, errors


def test_every_2d_imaging_line_carries_its_line_slice_repetition_and_polarity():
    app = app2d(n_slices=3, segments=2, acceleration=2, n_acs=0, n_repetitions=2)
    shots = imaging_shots(app.design(), app.epi.etl, "LIN", "SLC", "REP", "REV")
    offsets = app.epi.order[:, 0]
    reversed_lines = [i % 2 for i in range(app.epi.etl)]

    expected = [
        [(2 * segment + offsets).tolist(), [s] * len(offsets), [rep] * len(offsets)]
        for rep in range(2)
        for segment in range(2)
        for s in app.slices
    ]
    assert [shot[:3] for shot in shots] == expected
    assert all(shot[3] == reversed_lines for shot in shots)


def test_the_2d_prescan_is_played_once_ahead_of_the_imaging():
    app = app2d(n_slices=2, acceleration=2, n_acs=4, n_dummy=2)
    once, nav, ref, set_ = adc_labels(app.design(), "ONCE", "NAV", "REF", "SET")

    n_prescan = 2 * 4 + 2 * (app.NAVIGATOR_LINES + app.epi.etl)
    assert once.tolist() == [1] * n_prescan + [0] * (len(once) - n_prescan)
    imaging = once == 0
    assert not nav[imaging].any() and not ref[imaging].any()
    assert not set_[imaging].any()


def test_an_accelerated_2d_scan_calibrates_each_slice_with_a_gradient_echo():
    app = app2d(n_slices=2, acceleration=2, n_acs=4)
    lin, slc, ref, nav = adc_labels(app.design(), "LIN", "SLC", "REF", "NAV")
    calibration = (ref == 1) & (nav == 0)

    assert list(zip(lin[calibration], slc[calibration], strict=True)) == [
        (line, s) for s in app.calibration_slices for line in app.acs
    ]
    assert app2d(n_slices=2, acceleration=1, n_acs=4).acs == []


def test_the_2d_navigator_is_blip_nulled_and_the_reference_train_reversed():
    app = app2d()
    nav, ref, set_, rev = adc_labels(app.design(), "NAV", "REF", "SET", "REV")

    assert rev[nav == 1].tolist() == [0, 1, 0]
    assert ref[nav == 1].tolist() == [1, 1, 1]
    assert (set_ == 1).sum() == app.epi.etl

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


def test_a_multiband_shot_encodes_its_group_and_band_phase():
    app = app2d(n_slices=4, n_bands=2, sms=True, n_acs=4)
    seq = app.design()
    shots = imaging_shots(seq, app.epi.etl, "LIN", "PAR", "SLC", "SMS")

    assert [shot[2][0] for shot in shots] == app.slices
    for shot in shots:
        assert shot[0] == app.epi.order[:, 0].tolist()
        assert shot[1] == app.epi.order[:, 1].tolist()
        assert set(shot[3]) == {1}
    assert set(app.epi.order[:, 1]) == {0, 1}
    assert seq.definitions["MultibandFactor"] == [2.0]
    assert seq.definitions["Name"] == "sms_epi_2d"


def test_a_slice_count_the_multiband_factor_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="multiple of the multiband factor"):
        app2d(n_slices=3, n_bands=2, sms=True)


def test_fat_is_saturated_before_every_dummy_and_imaging_shot():
    kwargs = {**SMALL_2D, "n_slices": 2, "n_dummy": 1, "n_repetitions": 2}

    def pulses(seq):
        return sum(seq.get_block(i).rf is not None for i in seq.block_events)

    plain = pulses(epi2d.main(**kwargs))
    saturated = pulses(epi2d.main(**kwargs, fat_saturation=True))

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
        {},
        {
            "n_z": 8,
            "segments": 2,
            "acceleration": 2,
            "acceleration_z": 2,
            "partial_fourier": 0.75,
            "n_acs": 4,
            "n_acs_z": 2,
            "n_dummy": 1,
        },
        {"spsp": True, "te": 40e-3, "tr": 80e-3},
    ],
    ids=["single-shot", "segmented caipi partial-fourier", "water-selective"],
)
def test_a_small_3d_epi_passes_its_timing_check(kwargs):
    is_ok, errors = epi3d.main(**{**SMALL_3D, **kwargs}).check_timing()

    assert is_ok, errors


def test_every_3d_imaging_line_carries_its_line_and_partition():
    app = app3d(n_z=8, acceleration=2, acceleration_z=2, n_acs=0, n_repetitions=2)
    shots = imaging_shots(app.design(), app.epi.etl, "LIN", "PAR", "REP", "REV")
    order = app.epi.order

    expected = [
        [order[:, 0].tolist(), (2 * shell + order[:, 1]).tolist(), [rep] * app.epi.etl]
        for rep in range(2)
        for shell in app.shells
    ]
    assert [shot[:3] for shot in shots] == expected
    assert set(order[:, 1]) == {0, 1}  # the CAIPI sawtooth walks the shell
    assert all(shot[3] == [i % 2 for i in range(app.epi.etl)] for shot in shots)


def test_the_first_shells_encoded_are_the_central_ones():
    app = app3d(n_z=8)
    distance = [abs(shell - 3.5) for shell in app.shells]

    assert sorted(app.shells) == list(range(8))
    assert distance == sorted(distance)


def test_partial_fourier_drops_the_leading_lines_and_shells_but_not_the_reference():
    app = app3d(n_z=8, partial_fourier=0.75, partial_fourier_z=0.5, n_acs=0, n_acs_z=0)
    seq = app.design()
    once, lin, par, set_ = adc_labels(seq, "ONCE", "LIN", "PAR", "SET")

    assert app.epi.etl < app.ref_epi.etl == 16
    assert lin[once == 0].min() == app.first_y > 0
    assert sorted(set(par[once == 0])) == [4, 5, 6, 7]
    assert (set_ == 1).sum() == app.ref_epi.etl


def test_an_undersampled_3d_scan_calibrates_over_the_central_rectangle():
    app = app3d(acceleration=2, n_acs=4, n_acs_z=2)
    lin, par, ref, nav = adc_labels(app.design(), "LIN", "PAR", "REF", "NAV")
    calibration = (ref == 1) & (nav == 0)

    assert list(zip(lin[calibration], par[calibration], strict=True)) == [
        (y, z) for z in (1, 2) for y in (6, 7, 8, 9)
    ]
    assert app3d(n_acs=4, n_acs_z=2).acs == []


def test_the_3d_reference_train_starts_at_the_centre_partition():
    app = app3d()
    nav, set_, lin, par, rev = adc_labels(
        app.design(), "NAV", "SET", "LIN", "PAR", "REV"
    )

    assert rev[nav == 1].tolist() == [0, 1, 0]
    assert lin[set_ == 1].tolist() == app.ref_epi.order[:, 0].tolist()
    assert set(par[set_ == 1]) == {2}


def test_the_3d_prescan_is_played_once_ahead_of_the_imaging():
    app = app3d(n_dummy=2)
    (once,) = adc_labels(app.design(), "ONCE")

    n_prescan = app.NAVIGATOR_LINES + app.ref_epi.etl
    assert once.tolist() == [1] * n_prescan + [0] * (len(once) - n_prescan)


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
