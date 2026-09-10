"""Rotation constructor forms and angle ranges against the reference toolbox."""

import math

import numpy as np
import pytest

import pypulseqpp as pp

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
toolbox = pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that owns this constructor; see reference.py",
)


def quaternion_of(event):
    return np.asarray(event.rot_quaternion, dtype=float)


@pytest.mark.parametrize(
    ("label", "args"),
    [
        ("an angle about z", (math.pi / 6,)),
        ("an angle and a polar angle", (math.pi / 6, math.pi / 12)),
        ("an axis and an angle", (np.array([1.0, 1.0, 0.0]), math.pi / 4)),
        ("a quaternion", (np.array([0.5, 0.5, 0.5, 0.5]),)),
        ("a matrix", (Rotation.from_euler("z", 30, degrees=True).as_matrix(),)),
    ],
)
def test_every_form_says_what_the_toolbox_says(label, args):
    ours = quaternion_of(pp.make_rotation(*args))
    theirs = np.asarray(toolbox.make_rotation(*args).rot_quaternion, dtype=float)

    np.testing.assert_allclose(ours, theirs, atol=1e-15)


def test_a_stack_of_matrices_gives_one_event_each():
    stack = np.stack(
        [
            Rotation.from_euler("z", angle, degrees=True).as_matrix()
            for angle in (0, 30, 60)
        ]
    )

    made = pp.make_rotation(stack)
    theirs = toolbox.make_rotation(stack)

    assert [event.type for event in made] == ["rot3D"] * 3
    for mine, other in zip(made, theirs, strict=True):
        np.testing.assert_allclose(
            quaternion_of(mine), other.rot_quaternion, atol=1e-15
        )


def test_an_angle_and_a_rotation_object_turn_a_block_the_same_way(tmp_path):
    """Whichever way it is said, the block table holds the same four numbers."""
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    gradient = pp.make_trapezoid("x", area=600, duration=1e-3, system=system)

    written = []
    for rotation in (
        pp.make_rotation(math.pi / 3),
        pp.make_rotation(Rotation.from_euler("z", math.pi / 3)),
    ):
        sequence = pp.Sequence(system)
        sequence.add_block(gradient, rotation)
        path = tmp_path / f"{len(written)}.seq"
        sequence.write(str(path))
        written.append(path.read_bytes())

    assert written[0] == written[1]


def test_a_rotation_object_is_still_taken():
    made = pp.make_rotation(Rotation.from_euler("z", 0.4))

    assert made.type == "rot3D"
    # Kept as it is, so `add_block` can ask it once per orientation rather
    # than once per block.
    assert hasattr(made.rot_quaternion, "as_quat")


@pytest.mark.parametrize(
    ("args", "complaint"),
    [
        ((), "takes a rotation"),
        ((7.0,), r"phi .* is invalid"),
        ((0.0, 4.0), r"theta .* is invalid"),
        ((np.array([1.0, 0.0, 0.0]), 4.0), r"phi .* is invalid"),
        ((np.zeros(3), 0.5), "axis vector norm"),
        ((np.zeros(4),), "quaternion norm"),
        ((np.ones(5),), "did not recognise"),
        ((np.eye(3), 1.0), "exactly one argument"),
    ],
)
def test_what_is_not_a_rotation_is_refused(args, complaint):
    with pytest.raises(ValueError, match=complaint):
        pp.make_rotation(*args)


def test_the_angle_ranges_are_the_toolbox_s():
    """`phi` within [-pi, 2*pi), `theta` within [-pi, pi]."""
    for angle in (-math.pi, 0.0, math.pi, 2 * math.pi - 1e-9):
        assert pp.make_rotation(angle).type == "rot3D"
    for refused in (-math.pi - 1e-9, 2 * math.pi):
        with pytest.raises(ValueError, match="phi"):
            pp.make_rotation(refused)
