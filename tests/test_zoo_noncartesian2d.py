"""The 2D non-Cartesian zoo entries: radial, spiral and PROPELLER."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per zoo entry.
SMALL = {
    "gre_radial2D_sequence": {"n_x": 32, "n_spokes": 8, "n_dummy": 0},
    "gre_spiral2D_sequence": {"n_x": 32, "n_arms": 4, "n_dummy": 0},
    "se_propeller2D_sequence": {
        "n_x": 32,
        "blade_width": 8,
        "n_blades": 4,
        "te": None,
        "tr": None,
    },
}

#: Every definition each entry writes.
DEFINITIONS = {
    "gre_radial2D_sequence": {"NumSpokes", "AngleScheme", "kSpaceCenterSample"},
    "gre_spiral2D_sequence": {"NumArms", "AngleScheme", "kSpaceCenterSample"},
    "se_propeller2D_sequence": {"BladeWidth", "NumBlades"},
}
COMMON = {
    "FOV",
    "Matrix",
    "Name",
    "TE",
    "TR",
    "Trajectory",
    "NumGainCalibrationReadouts",
}


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app(name, **kwargs):
    mod = module(name)
    cls = next(v for k, v in vars(mod).items() if k.endswith("App"))
    return cls(pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def blocks(seq):
    return [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]


def rotation_angle(block):
    """The in-plane angle a block's rotation extension turns by, in radians."""
    q = np.asarray(block.rotation.quaternion)
    return 2.0 * np.arctan2(q[3], q[0])


def same_angles(played, intended):
    """Equal modulo a full turn."""
    return np.allclose(
        np.exp(1j * np.asarray(played)), np.exp(1j * np.asarray(intended))
    )


def in_plane(block):
    return block.gx is not None or block.gy is not None


def plane_waveform(block):
    """``gx + i gy`` of a block's arbitrary in-plane gradients."""
    gx = np.asarray(block.gx.waveform, dtype=float)
    gy = np.zeros_like(gx) if block.gy is None else np.asarray(block.gy.waveform)
    return gx + 1j * gy


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_a_sequence_that_passes_its_timing_check(name):
    seq = module(name).main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors


@pytest.mark.parametrize("name", SMALL)
def test_every_definition_is_written(name):
    seq = module(name).main(**SMALL[name])

    assert COMMON | DEFINITIONS[name] <= set(seq.definitions)


# -- radial and spiral -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "count"),
    [("gre_radial2D_sequence", "n_spokes"), ("gre_spiral2D_sequence", "n_arms")],
)
def test_each_acquisition_carries_its_shot_and_slice_in_play_order(name, count):
    shots = 5
    built = app(name, **{count: shots}, n_slices=3, n_dummy=2, tr=None)
    lin, slc = adc_labels(built.design(), "LIN", "SLC")

    expected = [(i, s) for group in built.passes for i in range(shots) for s in group]
    assert list(zip(lin, slc, strict=True)) == expected


@pytest.mark.parametrize(
    ("name", "scheme"),
    [
        ("gre_radial2D_sequence", "golden"),
        ("gre_radial2D_sequence", "uniform"),
        ("gre_spiral2D_sequence", "golden"),
        ("gre_spiral2D_sequence", "uniform"),
    ],
)
def test_every_in_plane_block_of_a_shot_is_turned_by_its_angle(name, scheme):
    built = app(name, angle_scheme=scheme, n_dummy=2)
    seq = built.design()

    # Blocks between two excitations belong to one shot; the dummies play the
    # first shot's orientation.
    schedule = [0] * built.n_dummy + list(range(len(built.angles)))
    shots = iter(schedule)
    played, intended = [], []
    for block in blocks(seq):
        if block.rf is not None:
            angle = built.angles[next(shots)]
        elif in_plane(block):
            played.append(rotation_angle(block))
            intended.append(angle)

    assert next(shots, None) is None
    assert same_angles(played, intended)


@pytest.mark.parametrize("name", ["gre_radial2D_sequence", "gre_spiral2D_sequence"])
def test_an_explicit_shot_is_the_first_one_turned_by_its_angle(name):
    built = app(name, use_rotation_ext=False, angle_scheme="golden")
    readouts = [b for b in blocks(built.design()) if b.adc is not None]

    assert all(b.rotation is None for b in readouts)
    first = plane_waveform(readouts[0])
    turned = [np.angle(np.vdot(first, plane_waveform(b))) for b in readouts]
    assert same_angles(turned, built.angles - built.angles[0])


def test_each_echo_of_a_spiral_train_is_labelled_by_the_readout():
    built = app("gre_spiral2D_sequence", n_echoes=3)
    lin, eco = adc_labels(built.design(), "LIN", "ECO")

    assert list(eco) == [0, 1, 2] * len(built.angles)
    assert list(lin) == [i for i in range(len(built.angles)) for _ in range(3)]


def test_every_slice_of_a_pass_is_excited_at_the_repetition_time_asked_for():
    spokes, tr = 4, 12e-3
    built = app("gre_radial2D_sequence", n_spokes=spokes, n_slices=3, tr=tr)
    excited = np.asarray(built.design().rf_times()[0])

    assert len(built.passes) > 1
    at = 0
    for group in built.passes:
        spacing = np.diff(excited[at : at + len(group) * spokes][:: len(group)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(group) * spokes


@pytest.mark.parametrize(
    ("name", "prescription"),
    [
        ("gre_radial2D_sequence", {"te": 1e-6}),
        ("gre_radial2D_sequence", {"tr": 1e-4}),
        ("gre_spiral2D_sequence", {"te": 1e-6}),
        ("gre_spiral2D_sequence", {"tr": 1e-4}),
        ("se_propeller2D_sequence", {"te": 1e-3}),
        ("se_propeller2D_sequence", {"tr": 1e-3}),
    ],
)
def test_an_infeasible_prescription_is_refused(name, prescription):
    with pytest.raises(ValueError, match="shorter than"):
        module(name).main(**{**SMALL[name], **prescription})


# -- PROPELLER -------------------------------------------------------------


def test_each_line_of_a_blade_carries_its_line_blade_and_slice():
    built = app("se_propeller2D_sequence", n_slices=2, n_dummy=1)
    lin, slc, seg = adc_labels(built.design(), "LIN", "SLC", "SEG")

    width = built.blade.etl
    expected = [
        (line, s, b)
        for group in built.passes
        for b in range(built.blade.n_blades)
        for s in group
        for line in range(width)
    ]
    assert list(zip(lin, slc, seg, strict=True)) == expected


@pytest.mark.parametrize("scheme", ["uniform", "golden"])
def test_every_encoding_block_of_a_blade_is_turned_by_its_angle(scheme):
    built = app("se_propeller2D_sequence", angle_scheme=scheme, n_dummy=1)
    angles = built.blade.blade_angles

    shots = iter([0] * built.n_dummy + list(range(len(angles))))
    played, intended = [], []
    for block in blocks(built.design()):
        if block.rf is not None and block.rf.use == "excitation":
            angle = angles[next(shots)]
        elif in_plane(block):
            played.append(rotation_angle(block))
            intended.append(angle)

    assert next(shots, None) is None
    assert same_angles(played, intended)


def test_the_refocusing_pulse_is_not_turned():
    seq = module("se_propeller2D_sequence").main(**SMALL["se_propeller2D_sequence"])

    assert all(b.rotation is None for b in blocks(seq) if b.rf is not None)


# -- the command line ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("gre_radial2D_sequence", "--n-spokes", "Spokes to play."),
        ("gre_spiral2D_sequence", "--n-echoes", "Arms read per excitation"),
        ("se_propeller2D_sequence", "--blade-width", "Phase-encode lines per blade."),
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
