"""Rotation-extension parity with explicitly rotated gradients and file round trips."""

import math
import warnings
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
    # The settings the file was written under, spelled out: it is a stored
    # artefact of the toolbox's, and what is being held is that this package
    # writes the same bytes -- not that the two happen to default alike.
    settings = pp.Opts(
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
        adc_raster_time=100e-9,
        block_duration_raster=10e-6,
    )
    sequence = pp.Sequence(settings)
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=1e-3, system=settings, use="excitation"
    )
    spoke = pp.make_trapezoid("x", area=1000, system=settings)
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


@pytest.mark.parametrize("start", [0.5, 2.0, 2.5, 3.1])
def test_gradients_ending_together_in_a_turned_block_end_at_zero(system, start):
    """An instant two axes reach by different sums is read as one corner.

    A turned block is played on the union of its axes' corners, which keeps
    one time for each instant. Two seconds into a sequence the same block end
    reached through different ramps differs by a rounding step, and reading
    the other axis there must give its own corner, not an interpolation a
    rounding step before it.
    """
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_delay(start))
    read = pp.make_trapezoid(
        "x",
        amplitude=1e6,
        rise_time=180e-6,
        flat_time=0,
        fall_time=180e-6,
        system=system,
    )
    encode = pp.make_trapezoid(
        "z",
        amplitude=195312.5,
        rise_time=80e-6,
        flat_time=0,
        fall_time=80e-6,
        delay=200e-6,
        system=system,
    )
    sequence.add_block(
        read, encode, pp.make_rotation(Rotation.from_euler("z", 10, degrees=True))
    )
    sequence.add_block(pp.make_delay(2e-3))
    sequence.add_block(
        pp.make_trapezoid(
            "z",
            amplitude=-195312.5,
            rise_time=80e-6,
            flat_time=0,
            fall_time=80e-6,
            system=system,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        times, values = sequence.waveforms()[2]

    end = start + 360e-6
    assert values[np.abs(times - end) < 1e-9].tolist() == [0.0]


def test_a_block_carries_at_most_one_rotation(system):
    spoke = pp.make_trapezoid("x", area=1000, system=system)
    turns = [
        pp.make_rotation(Rotation.from_euler("z", a, degrees=True)) for a in (30, 60)
    ]
    sequence = pp.Sequence(system)

    with pytest.raises(ValueError, match="at most one rotation"):
        sequence.add_block(spoke, *turns)
    assert sequence.num_blocks == 0


def test_each_block_names_the_rotation_it_plays(system):
    sequence = radial(system, by_extension=True, with_adc=False)

    named = sequence.block_rotations()

    turned = named > 0
    assert turned.sum() == len(ANGLES)
    rows = sequence.libraries().rotations[named[turned] - 1]
    expected = [
        Rotation.from_euler("z", angle, degrees=True).as_quat(
            canonical=True, scalar_first=True
        )
        for angle in ANGLES
    ]
    np.testing.assert_allclose(rows, expected, atol=1e-12)


def test_a_chain_holding_two_rotations_decodes_to_the_one_the_block_plays(system):
    """A file can carry two; the block plays the first, and reads back as it."""
    sequence = pp.Sequence(system)
    native = sequence._native
    gradient = native.register_trap(np.array([2000.0, 1e-4, 2e-3, 1e-4, 0.0]))
    first, second = (
        Rotation.from_euler("z", angle, degrees=True).as_quat(
            canonical=True, scalar_first=True
        )
        for angle in (30, 60)
    )
    kind = native.extension_type_id("ROTATIONS")
    tail = native.chain_extension(kind, native.register_rotation(second), 0)
    head = native.chain_extension(kind, native.register_rotation(first), tail)
    native.add_block(0, gradient, 0, 0, 0, head, 2.2e-3)

    np.testing.assert_allclose(
        sequence.get_block(1).rotation.quaternion, first, atol=1e-12
    )
    played = sequence.libraries().rotations[sequence.block_rotations()[0] - 1]
    np.testing.assert_allclose(played, first, atol=1e-12)
