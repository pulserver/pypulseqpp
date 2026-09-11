"""3D non-Cartesian zoo entries: stack of stars, stack of spirals and ZTE."""

import importlib

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per entry.
SMALL = {
    "gre_stack_of_stars3D_sequence": {"n_x": 32, "n_z": 4, "n_spokes": 5, "n_dummy": 0},
    "gre_stack_of_spirals3D_sequence": {"n_x": 32, "n_z": 4, "n_arms": 4, "n_dummy": 0},
    "zte3D_sequence": {"n_x": 32, "n_views": 24, "n_shots": 2, "n_dummy": 0},
}
STACKS = ("gre_stack_of_stars3D_sequence", "gre_stack_of_spirals3D_sequence")
APPS = {
    "gre_stack_of_stars3D_sequence": "GreStackOfStars3DApp",
    "gre_stack_of_spirals3D_sequence": "GreStackOfSpirals3DApp",
    "zte3D_sequence": "Zte3DApp",
}


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app(name, **kwargs):
    return getattr(module(name), APPS[name])(pp.Opts(), **{**SMALL[name], **kwargs})


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def acquisitions(seq):
    """Every block that acquires, in play order."""
    blocks = (seq.get_block(index) for index in range(1, len(seq) + 1))
    return [block for block in blocks if block.adc is not None]


def z_angle(quaternion):
    """Angle of a scalar-first quaternion that turns about z, in radians."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    assert x == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx(0.0, abs=1e-12)
    return 2.0 * np.arctan2(z, w)


def matrix_of(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def in_plane(block):
    """The played x/y waveform of a block as one complex array."""
    x = np.asarray(block.gx.waveform) if block.gx is not None else 0.0
    y = np.asarray(block.gy.waveform) if block.gy is not None else 0.0
    return x + 1j * y


def wrapped(angles):
    return np.angle(np.exp(1j * np.asarray(angles)))


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_builds_a_sequence_that_passes_its_timing_check(name):
    seq = module(name).main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors
    assert seq.definitions["Name"] == getattr(module(name), APPS[name]).NAME


@pytest.mark.parametrize("name", STACKS)
def test_every_partition_of_a_stack_arm_is_acquired_before_the_next_arm(name):
    stack = app(name, n_dummy=3)
    seq = stack.design()
    lin, par, once = adc_labels(seq, "LIN", "PAR", "ONCE")
    n_z = stack.matrix[2]

    expected = [(arm, p) for arm in range(len(stack.angles)) for p in range(n_z)]
    assert list(zip(lin, par, strict=True)) == expected
    assert set(once) == {0}
    assert len(seq.rf_times()[0]) == 3 + len(expected)


@pytest.mark.parametrize(
    "use_rotation_ext", [True, False], ids=["extension", "waveform"]
)
@pytest.mark.parametrize("name", STACKS)
def test_every_stack_acquisition_is_turned_to_its_arm_and_partition_angle(
    name, use_rotation_ext
):
    offset = 10.0
    stack = app(
        name, partition_angle_offset_deg=offset, use_rotation_ext=use_rotation_ext
    )
    n_z = stack.matrix[2]
    expected = [
        stack.angles[arm] + np.deg2rad(offset) * p
        for arm in range(len(stack.angles))
        for p in range(n_z)
    ]

    played = acquisitions(stack.design())
    if use_rotation_ext:
        angles = [z_angle(block.rotation.quaternion) for block in played]
    else:
        # The waveform turned from the unrotated arm is the angle played.
        base = in_plane(acquisitions(app(name).design())[0])
        base = base / np.exp(1j * stack.angles[0])
        angles = [np.angle(np.sum(in_plane(b) * np.conj(base))) for b in played]
        assert all(block.rotation is None for block in played)

    assert wrapped(np.subtract(angles, expected)) == pytest.approx(0.0, abs=1e-6)


def test_a_stack_without_a_partition_offset_turns_every_partition_alike():
    stack = app("gre_stack_of_stars3D_sequence")
    played = acquisitions(stack.design())
    angles = np.reshape([z_angle(b.rotation.quaternion) for b in played], (-1, 4))

    assert wrapped(angles - stack.angles[:, None]) == pytest.approx(0.0, abs=1e-6)
    assert len(stack.rotations) == len(stack.angles)


def test_every_zte_view_of_every_shot_is_acquired_once_in_order():
    zte = app("zte3D_sequence", n_shots=3, n_dummy=2)
    lin, seg, once = adc_labels(zte.design(), "LIN", "SEG", "ONCE")
    n_views = len(zte.zte.directions)

    assert list(zip(lin, seg, strict=True)) == [
        (view, shot) for shot in range(3) for view in range(n_views)
    ]
    assert set(once) == {0}


def test_every_zte_acquisition_is_turned_by_its_shot_rotation():
    zte = app("zte3D_sequence", n_shots=3, n_dummy=1)
    seq = zte.design()
    (seg,) = adc_labels(seq, "SEG")

    for block, shot in zip(acquisitions(seq), seg, strict=True):
        assert matrix_of(block.rotation.quaternion) == pytest.approx(
            zte.zte.shot_rotations[shot], abs=1e-9
        )
    # A dummy shell is the first shot without the ADC.
    first = seq.get_block(1).rotation.quaternion
    assert matrix_of(first) == pytest.approx(zte.zte.shot_rotations[0], abs=1e-9)


@pytest.mark.parametrize(
    ("name", "infeasible"),
    [
        ("gre_stack_of_stars3D_sequence", {"te": 1e-6}),
        ("gre_stack_of_spirals3D_sequence", {"tr": 1e-4}),
        ("gre_stack_of_stars3D_sequence", {"angle_scheme": "random"}),
        ("zte3D_sequence", {"pulse_duration": 1e-3}),
    ],
    ids=["stars te", "spirals tr", "stars angles", "zte pulse"],
)
def test_an_infeasible_prescription_is_refused(name, infeasible):
    with pytest.raises(ValueError):
        module(name).main(**{**SMALL[name], **infeasible})


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        (
            "gre_stack_of_stars3D_sequence",
            "--partition-angle-offset-deg",
            "Angle added per partition step, in degrees.",
        ),
        (
            "gre_stack_of_spirals3D_sequence",
            "--n-arms",
            "Interleaves per partition, which is also the designed pitch.",
        ),
        ("zte3D_sequence", "--n-views", "Views per shell."),
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
