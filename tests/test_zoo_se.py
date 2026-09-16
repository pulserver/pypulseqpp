"""The spin-echo example sequences: 2D and 3D spin echo."""

import importlib
from itertools import pairwise

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

se2D = importlib.import_module("pypulseqpp.sequences.sequence.se2D_sequence")
se3D = importlib.import_module("pypulseqpp.sequences.sequence.se3D_sequence")

MODULES = {"se2D_sequence": se2D, "se3D_sequence": se3D}

#: A prescription small enough to build in a moment, per sequence.
SMALL = {
    "se2D_sequence": {"n_x": 32, "n_y": 16, "n_slices": 1, "n_acs_y": 0, "tr": None},
    "se3D_sequence": {
        "n_x": 32,
        "n_y": 8,
        "n_z": 4,
        "n_acs_y": 0,
        "n_acs_z": 0,
        "tr": None,
    },
}

APPS = {
    "se2D_sequence": se2D.Se2DApp,
    "se3D_sequence": se3D.Se3DApp,
}


def app(name, **kwargs):
    return APPS[name](pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def played(seq):
    """RF ``(centre time, use, frequency offset)`` and echo-sample times, in order."""
    center = int(np.atleast_1d(seq.definitions["kSpaceCenterSample"])[0])
    t, pulses, echoes = 0.0, [], []
    for index in range(1, len(seq.block_events) + 1):
        block = seq.get_block(index)
        rf, adc = getattr(block, "rf", None), getattr(block, "adc", None)
        if rf is not None:
            pulses.append((t + rf.delay + rf.center, rf.use, rf.freq_offset))
        if adc is not None:
            echoes.append(t + adc.delay + center * adc.dwell)
        t += block.block_duration
    return pulses, echoes


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_and_passes_its_timing_check(name):
    seq = MODULES[name].main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors
    assert seq.definitions["Name"] == APPS[name].NAME


# -- 2D spin echo ------------------------------------------------------------


def test_each_spin_echo_acquisition_carries_its_line_and_slice_in_play_order():
    se = app("se2D_sequence", n_slices=4, tr=40e-3, ry=2, n_acs_y=4)
    lin, slc, ima, seg = adc_labels(se.design(), "LIN", "SLC", "IMA", "SEG")

    assert len(se.packets) == 2
    expected = [(line, s) for packet in se.packets for line in se.lines for s in packet]
    assert list(zip(lin, slc, strict=True)) == expected
    assert list(ima) == [int(line in se.calibration) for line, _ in expected]
    assert list(seg) == [1 - int(line in se.calibration) for line, _ in expected]


def test_the_slices_of_a_spin_echo_packet_are_not_neighbours_and_keep_the_tr():
    se = app("se2D_sequence", n_slices=6, tr=40e-3)
    seq = se.design()

    assert len(se.packets) > 1
    for packet in se.packets:
        assert all(b - a >= len(se.packets) for a, b in pairwise(sorted(packet)))
    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(40e-3)


def test_every_refocusing_pulse_selects_the_slice_its_excitation_does():
    """Refocusing offset = excitation offset x the ratio of selection amplitudes.

    The refocusing pulse selects on the plateau between its crushers, its
    ``selection_amplitude``, not on the crusher peak its gradient's
    ``amplitude`` reports.
    """
    thickness = 4e-3
    sequence = app(
        "se2D_sequence", n_slices=3, slice_thickness=thickness, slice_spacing=1e-3
    )
    pulses, _ = played(sequence.design())
    excitation = sequence.exc.selection_amplitude
    refocusing = sequence.ref.selection_amplitude

    assert refocusing != pytest.approx(sequence.ref.gz.amplitude)

    excited, pairs = [], []
    for _, use, offset in pulses:
        if use == "excitation":
            excited.append(offset)
        else:
            pairs.append((offset, excited[-1]))

    assert pairs
    assert all(
        ref == pytest.approx(exc * refocusing / excitation, abs=1e-6)
        for ref, exc in pairs
    )
    assert sorted({round(f / excitation, 9) for f in excited}) == pytest.approx(
        sorted(sequence.positions)
    )
    assert any(exc != 0.0 for _, exc in pairs)


@pytest.mark.parametrize("name", ["se2D_sequence", "se3D_sequence"])
def test_a_spin_echo_forms_at_the_te_asked_for_with_the_180_midway(name):
    seq = app(name, te=20e-3).design()
    pulses, echoes = played(seq)
    excitation, refocusing = pulses[0][0], pulses[1][0]

    assert np.atleast_1d(seq.definitions["TE"])[0] == pytest.approx(20e-3, abs=1e-9)
    assert echoes[0] - excitation == pytest.approx(20e-3, abs=1e-9)
    assert refocusing - excitation == pytest.approx(10e-3, abs=1e-9)


@pytest.mark.parametrize("name", ["se2D_sequence", "se3D_sequence"])
@pytest.mark.parametrize(
    ("prescription", "match"),
    [({"te": 1e-3}, "TE"), ({"tr": 5e-3}, "TR")],
    ids=["te", "tr"],
)
def test_a_spin_echo_shorter_than_it_can_play_is_refused(name, prescription, match):
    with pytest.raises(ValueError, match=match):
        MODULES[name].main(**{**SMALL[name], **prescription})


# -- 3D spin echo ------------------------------------------------------------


def test_each_3d_acquisition_carries_its_view_calibration_rectangle_first():
    se = app("se3D_sequence", ry=2, n_acs_y=4, n_acs_z=2, tr=50e-3)
    seq = se.design()
    lin, par, ima, seg = adc_labels(seq, "LIN", "PAR", "IMA", "SEG")
    calibrating = [int(view in se.calibration) for view in se.views]

    assert se.views[: len(se.calibration)] == sorted(se.calibration)
    assert list(zip(lin, par, strict=True)) == se.views
    assert list(ima) == calibrating
    assert list(seg) == [1 - c for c in calibrating]
    assert len(played(seq)[0]) == 2 * len(se.views)


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
@pytest.mark.parametrize("name", ["se2D_sequence", "se3D_sequence"])
def test_the_180_sits_midway_for_a_te_off_the_raster(name, excitation):
    """The TE is rounded up onto the raster rather than the 180 moved off midway."""
    kind = {} if name == "se2D_sequence" else {"excitation": excitation}
    requested = app(name, **kind).echo_time + 4.013e-3
    built = app(name, **kind, te=requested)
    seq = built.design()
    pulses, echoes = played(seq)
    (excitation_time, _, _), (refocusing_time, use, _) = pulses[:2]
    written = np.atleast_1d(seq.definitions["TE"])[0]

    assert seq.check_timing()[0]
    assert use == "refocusing"
    assert echoes[0] - excitation_time == pytest.approx(written, abs=1e-9)
    assert refocusing_time - excitation_time == pytest.approx(written / 2, abs=1e-9)
    raster = built.system.block_duration_raster
    assert requested - 1e-9 <= written <= requested + 2 * raster


# -- the command line --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("se2D_sequence", "--partial-fourier-x", "Fraction of the echo acquired"),
        ("se3D_sequence", "--n-z", "Matrix size along the readout"),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(MODULES[name].main, ["--help"])

    printed = capsys.readouterr().out

    assert flag in printed
    assert help_text in printed
