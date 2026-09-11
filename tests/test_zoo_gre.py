"""Gradient-echo zoo entries: 3D, and multi-echo 2D and 3D."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per zoo entry.
SMALL = {
    "gre3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "n_acs": 0,
        "n_acs_z": 0,
        "n_dummy": 0,
    },
    "gre_multiecho2D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_echoes": 3,
        "n_acs": 0,
        "n_dummy": 0,
    },
    "gre_multiecho3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "n_echoes": 3,
        "n_acs": 0,
        "n_acs_z": 0,
        "n_dummy": 0,
    },
}

APPS = {
    "gre3D_sequence": "Gre3DApp",
    "gre_multiecho2D_sequence": "GreMultiecho2DApp",
    "gre_multiecho3D_sequence": "GreMultiecho3DApp",
}


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app(name, **kwargs):
    cls = getattr(module(name), APPS[name])
    return cls(pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def acquisitions(seq):
    """The blocks that acquire, in play order."""
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return [block for block in blocks if block.adc is not None]


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_passes_its_timing_check(name):
    seq = module(name).main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("prescription", [{"te": 1e-6}, {"tr": 1e-3}], ids=str)
def test_an_echo_or_repetition_shorter_than_the_readout_is_refused(name, prescription):
    with pytest.raises(ValueError, match="shorter than"):
        module(name).main(**{**SMALL[name], **prescription})


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("gre3D_sequence", "--n-z", "Partition-encode steps."),
        ("gre_multiecho2D_sequence", "--n-echoes", "Echoes per excitation."),
        ("gre_multiecho3D_sequence", "--n-echoes", "Echoes per repetition."),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module(name).main, ["--help"])

    printed = capsys.readouterr().out

    assert flag in printed
    assert help_text in printed


# -- 3D: one (line, partition) pair per excitation ----------------------------


@pytest.mark.parametrize("name", ["gre3D_sequence", "gre_multiecho3D_sequence"])
def test_every_sampled_pair_is_acquired_in_order_with_its_calibration_flags(name):
    a = app(name, acceleration=2, acceleration_z=2, n_acs=4, n_acs_z=2, n_dummy=3)
    lin, par, ima, seg = adc_labels(a.design(), "LIN", "PAR", "IMA", "SEG")

    echoes = getattr(a, "n_echoes", 1)
    expected = [pair for pair in a.pairs for _ in range(echoes)]
    calibrating = [i < a.n_calibration for i, _ in enumerate(a.pairs)]
    calibrating = [c for c in calibrating for _ in range(echoes)]

    assert a.n_calibration > 0
    assert list(zip(lin, par, strict=True)) == expected
    assert list(ima) == [int(c) for c in calibrating]
    assert list(seg) == [1 - int(c) for c in calibrating]


def test_every_partition_is_encoded_at_the_step_its_label_names():
    a = app("gre3D_sequence", elliptical=False)
    seq = a.design()
    (par,) = adc_labels(seq, "PAR")
    n_z = a.matrix[2]

    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    steps = [
        blocks[i - 1].gz.amplitude / a.ro.gz_pre.amplitude
        for i, block in enumerate(blocks)
        if block.adc is not None
    ]

    assert sorted(set(par)) == list(range(n_z))
    assert steps == pytest.approx((par - n_z / 2) / (n_z / 2))


@pytest.mark.parametrize("name", ["gre3D_sequence", "gre_multiecho3D_sequence"])
def test_the_wave_free_reference_leads_and_is_marked_ref(name):
    a = app(name, wave="both", wave_cycles=2, n_acs=4, n_acs_z=2, n_dummy=2)
    seq = a.design()
    lin, par, ref, ima, seg = adc_labels(seq, "LIN", "PAR", "REF", "IMA", "SEG")

    echoes = getattr(a, "n_echoes", 1)
    n_reference = len(a.reference) * echoes
    views = [pair for pair in a.reference + a.pairs for _ in range(echoes)]

    assert seq.check_timing()[0]
    assert len(a.reference) == a.n_calibration > 0
    assert list(zip(lin, par, strict=True)) == views
    assert list(ref) == [1] * n_reference + [0] * (len(views) - n_reference)
    assert set(seg[n_reference:]) == {1}
    assert set(ima) == {0}


def test_the_reference_is_played_with_the_corkscrew_scaled_away():
    a = app("gre3D_sequence", wave="phase", wave_cycles=2, n_acs=4, n_acs_z=2)
    reads = acquisitions(a.design())
    n_reference = len(a.reference)

    assert all(b.gy.amplitude == 0 for b in reads[:n_reference])
    assert all(b.gy.amplitude != 0 for b in reads[n_reference:])


# -- multi-echo: every view read at each echo time ----------------------------


def test_each_acquisition_carries_the_line_slice_and_echo_it_reads():
    a = app("gre_multiecho2D_sequence", n_slices=3, acceleration=2, n_acs=4, n_dummy=2)
    lin, slc, eco, ima = adc_labels(a.design(), "LIN", "SLC", "ECO", "IMA")

    expected = [
        (line, s, echo)
        for group in a.passes
        for line in a.lines
        for s in group
        for echo in range(a.n_echoes)
    ]
    assert list(zip(lin, slc, eco, strict=True)) == expected
    assert list(ima) == [int(line in a.calibration) for line, _, _ in expected]


@pytest.mark.parametrize(
    "name", ["gre_multiecho2D_sequence", "gre_multiecho3D_sequence"]
)
@pytest.mark.parametrize("monopolar", [True, False], ids=["monopolar", "bipolar"])
def test_the_echo_times_written_are_evenly_spaced_from_the_first(name, monopolar):
    a = app(name, n_echoes=4, monopolar=monopolar)
    written = np.atleast_1d(a.design().definitions["TE"])

    assert len(written) == 4
    assert written[0] == pytest.approx(a.ro.echo_time)
    assert np.diff(written) == pytest.approx(np.diff(written)[0])


@pytest.mark.parametrize(
    "name", ["gre_multiecho2D_sequence", "gre_multiecho3D_sequence"]
)
def test_a_bipolar_train_reads_alternate_echoes_backwards(name):
    a = app(name, n_echoes=4, monopolar=False)
    signs = [np.sign(b.gx.amplitude) for b in acquisitions(a.design())]

    assert signs[:4] == [1, -1, 1, -1]
    assert len(set(map(tuple, np.reshape(signs, (-1, 4))))) == 1


@pytest.mark.parametrize(
    "name", ["gre_multiecho2D_sequence", "gre_multiecho3D_sequence"]
)
def test_a_bipolar_train_spaces_its_echoes_closer_than_a_monopolar_one(name):
    mono = np.atleast_1d(app(name, n_echoes=3).design().definitions["TE"])
    bi = np.atleast_1d(
        app(name, n_echoes=3, monopolar=False).design().definitions["TE"]
    )

    assert np.diff(bi)[0] < np.diff(mono)[0]


def test_every_slice_is_excited_at_the_repetition_time_written():
    """Including the smaller pass, which waits longer."""
    tr, lines = 30e-3, 16
    a = app("gre_multiecho2D_sequence", n_slices=7, tr=tr)
    seq = a.design()
    excited = np.asarray(seq.rf_times()[0])

    assert len({len(group) for group in a.passes}) == 2
    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(tr)
    at = 0
    for group in a.passes:
        spacing = np.diff(excited[at : at + len(group) * lines][:: len(group)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(group) * lines


def test_a_repetition_just_short_of_whole_shots_takes_more_passes_not_a_refusal():
    """A shot is the readout and its closing raster delay, not the readout alone."""
    shortest = app("gre_multiecho2D_sequence", n_slices=4, tr=None)
    a = app("gre_multiecho2D_sequence", n_slices=4, tr=2 * shortest.ro.duration)

    assert len(a.passes) == 4
    assert a.design().check_timing()[0]
