"""The balanced example sequence: 2D bSSFP."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per sequence.
SMALL = {
    "bssfp2D_sequence": {
        "n_x": 64,
        "n_y": 16,
        "readout_bandwidth_hz": 50e3,
        "n_acs_y": 0,
        "n_dummy": 0,
    },
}

APPS = {
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


# -- 2D balanced SSFP ------------------------------------------------------

BSSFP = "bssfp2D_sequence"


def test_each_slice_is_acquired_whole_before_the_next():
    a = app(BSSFP, n_slices=3, slice_gap=1e-3, n_acs_y=4, acceleration=2, n_dummy=2)
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
