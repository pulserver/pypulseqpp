"""Inversion-prepared and balanced zoo sequences: MPRAGE, stack-of-spirals MPRAGE, 2D bSSFP."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per sequence.
SMALL = {
    "mprage3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "views_per_segment": 16,
        "ti": 100e-3,
        "tr_outer": 300e-3,
        "n_acs": 0,
        "n_acs_z": 0,
        "n_dummy": 0,
    },
    "mprage_stack_of_spirals3D_sequence": {
        "n_x": 32,
        "n_z": 4,
        "n_arms": 4,
        "ti": 100e-3,
        "tr_outer": 300e-3,
        "n_dummy": 0,
    },
    "bssfp2D_sequence": {
        "n_x": 64,
        "n_y": 16,
        "readout_bandwidth_hz": 50e3,
        "n_acs": 0,
        "n_dummy": 0,
    },
}

APPS = {
    "mprage3D_sequence": "Mprage3DApp",
    "mprage_stack_of_spirals3D_sequence": "MprageStackOfSpirals3DApp",
    "bssfp2D_sequence": "Bssfp2DApp",
}

RASTER = pp.Opts().block_duration_raster


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def build(name, **kwargs):
    return module(name).main(**{**SMALL[name], **kwargs})


def app(name, **kwargs):
    cls = getattr(module(name), APPS[name])
    return cls(pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def pulses(seq, use):
    return seq.rf_times(compat=False).of(use)


def same_phase(a, b):
    return np.allclose(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))), 0.0)


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_a_sequence_that_passes_its_timing_check(name):
    seq = build(name)

    is_ok, errors = seq.check_timing()
    assert is_ok, errors
    assert seq.definitions["Name"] == getattr(module(name), APPS[name]).NAME


# -- 3D MPRAGE -------------------------------------------------------------

MPRAGE = "mprage3D_sequence"


@pytest.mark.parametrize("ordering", ("linear", "centric", "radial", "shuffling"))
def test_every_mprage_view_is_acquired_once_at_its_place_in_the_segment(ordering):
    a = app(MPRAGE, ordering=ordering, acceleration=2, n_acs=4, n_acs_z=2, n_dummy=1)
    lin, par, eco = adc_labels(a.design(), "LIN", "PAR", "ECO")

    views = [v for segment in a.segments for v in segment if v is not None]
    echoes = [e for seg in a.segments for e, v in enumerate(seg) if v is not None]

    assert list(zip(lin, par, strict=True)) == views
    assert len(set(views)) == len(views)
    assert list(eco) == echoes


def test_the_centre_view_is_excited_ti_after_its_inversion():
    a = app(MPRAGE, n_dummy=1, ti=120e-3)
    seq = a.design()
    inverted = pulses(seq, "inversion").t
    excited = pulses(seq, "excitation").t.reshape(len(inverted), -1)
    centre = (a.matrix[1] // 2, a.matrix[2] // 2)

    assert [centre in segment for segment in a.segments].count(True) == 1
    segment = next(s for s in a.segments if centre in s)
    assert segment.index(centre) == a.n_center
    assert excited[:, a.n_center] - inverted == pytest.approx(120e-3, abs=RASTER / 2)


def test_each_mprage_shot_repeats_at_the_outer_repetition_time():
    seq = build(MPRAGE, n_dummy=1, tr_outer=400e-3, navigator=True)
    inverted = pulses(seq, "inversion").t

    assert np.diff(inverted) == pytest.approx(400e-3, abs=1e-9)
    assert seq.check_timing()[0]


def test_the_wave_free_calibration_shots_lead_and_are_marked_reference():
    a = app(MPRAGE, wave="both", wave_cycles=2, n_acs=4, n_acs_z=2)
    seq = a.design()
    ref, lin, par = adc_labels(seq, "REF", "LIN", "PAR")
    n_reference = len(a.calibration_views)

    assert seq.check_timing()[0]
    assert list(zip(lin, par, strict=True))[:n_reference] == a.calibration_views
    assert list(ref) == [1] * n_reference + [0] * (len(ref) - n_reference)
    # A part-filled calibration shot is padded, not shortened.
    assert np.diff(pulses(seq, "inversion").t) == pytest.approx(a.tr_outer, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"ti": 1e-3}, "TI"), ({"tr_outer": 50e-3}, "shorter than one segment")],
)
def test_an_mprage_that_does_not_fit_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build(MPRAGE, **kwargs)


# -- stack-of-spirals MPRAGE -----------------------------------------------

SPIRALS = "mprage_stack_of_spirals3D_sequence"


def test_every_arm_of_every_partition_is_acquired_under_its_own_train():
    a = app(SPIRALS, etl=2, n_dummy=1)
    lin, par, seg = adc_labels(a.design(), "LIN", "PAR", "SEG")
    n_arms, n_z = a.n_arms, a.matrix[2]

    expected = [
        (arm, partition, partition * (n_arms // 2) + arm // 2)
        for partition in range(n_z)
        for arm in range(n_arms)
    ]
    assert list(zip(lin, par, seg, strict=True)) == expected


def test_the_first_echo_of_a_spiral_train_is_ti_after_its_inversion():
    a = app(SPIRALS, n_dummy=1, ti=120e-3, tr_outer=400e-3)
    seq = a.design()
    inverted = pulses(seq, "inversion").t
    excited = pulses(seq, "excitation").t.reshape(len(inverted), -1)

    first_echo = excited[:, 0] + a.ro.echo_time
    assert first_echo - inverted == pytest.approx(120e-3, abs=RASTER / 2)
    assert np.diff(inverted) == pytest.approx(400e-3, abs=1e-9)


def test_one_rotation_event_is_made_per_distinct_shot_angle():
    plain = app(SPIRALS)
    staggered = app(SPIRALS, partition_angle_offset_deg=10.0)

    assert len(plain.rotations) == plain.n_arms
    assert len(staggered.rotations) == staggered.n_arms * staggered.matrix[2]


def test_explicit_arms_play_the_same_timing_as_rotated_ones():
    rotated = build(SPIRALS, partition_angle_offset_deg=10.0)
    explicit = build(SPIRALS, partition_angle_offset_deg=10.0, use_rotation_ext=False)

    assert explicit.check_timing()[0]
    assert np.array_equal(explicit.rf_times()[0], rotated.rf_times()[0])
    assert np.array_equal(explicit.adc_times()[0], rotated.adc_times()[0])


def test_an_echo_train_that_does_not_divide_the_arms_is_refused():
    with pytest.raises(ValueError, match="etl must divide"):
        build(SPIRALS, etl=3)


# -- 2D balanced SSFP ------------------------------------------------------

BSSFP = "bssfp2D_sequence"


def test_each_slice_is_acquired_whole_before_the_next():
    a = app(BSSFP, n_slices=3, slice_gap=1e-3, n_acs=4, acceleration=2, n_dummy=2)
    lin, slc, seg = adc_labels(a.design(), "LIN", "SLC", "SEG")

    expected = [(line, s) for s in range(3) for line in a.lines]
    assert list(zip(lin, slc, strict=True)) == expected
    assert list(seg) == [a.segment[line] for line, _ in expected]


def test_the_excitation_and_receiver_phase_alternate_after_an_opposite_half_flip():
    a = app(BSSFP, n_slices=3, slice_gap=1e-3, n_dummy=1)
    seq = a.design()
    shots = a.n_dummy + len(a.lines)
    train = [0.0] + [np.pi * ((shot + 1) % 2) for shot in range(shots)]
    received = [np.pi * ((a.n_dummy + i + 1) % 2) for i in range(len(a.lines))]

    excitation = pulses(seq, "excitation")
    assert np.any(excitation.freq_offset != 0.0)
    assert same_phase(excitation.phase_offset, train * 3)
    assert same_phase(seq.adc_times()[1][:, 1], received * 3)


def test_once_marks_each_slice_half_flip_and_closing_rewind():
    a = app(BSSFP, n_slices=2, n_dummy=1)
    seq = a.design()
    once = np.atleast_1d(seq.evaluate_labels(evolution="blocks")["ONCE"])
    half_flips = pulses(seq, "excitation").block[:: a.n_dummy + len(a.lines) + 1]

    chunks = np.split(once, 2)
    for chunk in chunks:
        assert chunk[0] == 1
        assert chunk[-1] == 2
        assert set(chunk[1:-1]) == {0}
    assert list(half_flips) == [1, len(chunks[0]) + 1]


def test_each_slice_sets_every_label_on_its_first_acquisition():
    """The module's ONCE change restarts the labels, as a ONCE through labels() would."""
    a = app(BSSFP, n_slices=3, n_dummy=0)
    seq = a.design()
    n_blocks = len(seq.block_events)
    acquiring = [
        seq.get_block(i) for i in range(1, n_blocks + 1) if seq.get_block(i).adc
    ]

    for first in acquiring[:: len(a.lines)]:
        assert sorted((e.label, e.type) for e in first.label) == [
            ("IMA", "labelset"),
            ("LIN", "labelset"),
            ("SEG", "labelset"),
            ("SLC", "labelset"),
        ]


def test_a_repetition_shorter_than_the_balanced_one_is_refused():
    with pytest.raises(ValueError):
        build(BSSFP, tr=1e-3)


# -- the command line ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        (MPRAGE, "--views-per-segment", "Views acquired per inversion."),
        (SPIRALS, "--tr-outer", "Inversion-to-inversion interval, in seconds."),
        (BSSFP, "--n-slices", "Number of slices, each acquired as its own"),
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
