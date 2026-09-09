"""A rotation carried as an extension plays what rotating the events plays.

Ported from `pypulseq-matlab-like`'s `test_rotation_extension`. The point of
the extension is that one waveform serves every shot: a radial base spoke is
designed once and each spoke is that spoke under a different rotation, so
nothing is redesigned and nothing is registered twice. That only holds if the
sequence plays what it would have played had the gradients been rotated when
they were made -- which is what `rotate` does, and what these compare against.

The rotation survives a file and a block-by-block rebuild, because it is
written as an extension and read back as one.
"""

import math
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

ANGLES = (0, 30, 45, 60, 90)


@pytest.fixture
def system():
    return pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def radial(system, by_extension, with_adc=True):
    """Spokes at `ANGLES`, turned by an extension or by rotating the events."""
    sequence = pp.Sequence(system)
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    spoke = pp.make_trapezoid("x", area=1000, system=system)
    adc = pp.make_adc(
        64, duration=spoke.flat_time, delay=spoke.rise_time, system=system
    )
    sampling = (adc,) if with_adc else ()
    for angle in ANGLES:
        sequence.add_block(pulse)
        if by_extension:
            sequence.add_block(
                spoke,
                *sampling,
                pp.make_rotation(Rotation.from_euler("z", angle, degrees=True)),
            )
        else:
            sequence.add_block(
                *pp.rotate(spoke, axis="z", angle=math.radians(angle)), *sampling
            )
    return sequence


def assert_plays_the_same(one, other, within):
    """Two sequences draw the same gradient on every axis.

    `within` is a fraction of what the gradient reaches, since a waveform is
    compared where it is zero as well as where it is not.
    """
    for axis, (mine, theirs) in enumerate(
        zip(one.waveforms(), other.waveforms(), strict=True)
    ):
        assert mine.shape == theirs.shape, f"axis {axis} has a different shape"
        if mine.size == 0:
            continue
        np.testing.assert_allclose(mine[0], theirs[0], atol=1e-9, rtol=0)
        allowed = within * float(np.abs(mine[1]).max())
        np.testing.assert_allclose(mine[1], theirs[1], atol=allowed, rtol=0)


def assert_samples_the_same(one, other, within):
    """The two follow the same trajectory, as a fraction of its extent."""
    mine = one.calculate_kspace()[0]
    theirs = other.calculate_kspace()[0]
    np.testing.assert_allclose(
        mine, theirs, atol=within * float(np.abs(mine).max()), rtol=0
    )


def test_turning_by_extension_plays_what_turning_the_events_plays(system):
    turned, rotated = radial(system, True), radial(system, False)

    assert_plays_the_same(turned, rotated, within=1e-12)
    assert_samples_the_same(turned, rotated, within=1e-12)


def test_the_file_is_the_one_the_toolbox_writes(system, tmp_path):
    """`tests/seq/seq_make_radial.seq` is the toolbox's own output for this."""
    written = tmp_path / "seq_make_radial.seq"
    sequence = pp.Sequence()
    pulse = pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation")
    spoke = pp.make_trapezoid("x", area=1000)
    for angle in (*ANGLES, 0):
        sequence.add_block(pulse)
        sequence.add_block(
            spoke, pp.make_rotation(Rotation.from_euler("z", angle, degrees=True))
        )
    sequence.write(str(written))

    expected = Path(__file__).parent / "seq" / "seq_make_radial.seq"
    assert written.read_bytes() == expected.read_bytes()


def test_a_turned_sequence_survives_a_file(system, tmp_path):
    original = radial(system, True)
    path = tmp_path / "turned.seq"
    original.write(str(path))

    reread = pp.Sequence(system)
    reread.read(str(path))

    assert len(reread) == len(original)
    # What a round trip costs is the precision the text format writes an
    # amplitude at, and nothing else.
    assert_plays_the_same(original, reread, within=1e-5)
    assert_samples_the_same(original, reread, within=1e-4)


def test_a_turned_sequence_survives_being_rebuilt_block_by_block(system):
    """`get_block` hands back the rotation, so `add_block` puts it back."""
    original = radial(system, True)

    rebuilt = pp.Sequence(system)
    for index in range(1, len(original) + 1):
        rebuilt.add_block(original.get_block(index))

    assert len(rebuilt) == len(original)
    assert_plays_the_same(original, rebuilt, within=0.0)
    assert_samples_the_same(original, rebuilt, within=0.0)
