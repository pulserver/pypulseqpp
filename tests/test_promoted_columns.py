"""Rotation and RF-shim columns remain consistent with extension chains."""

import numpy as np
import pytest

import pypulseqpp as pp


@pytest.fixture
def system():
    return pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


@pytest.fixture
def turned_and_shimmed(system):
    """Four blocks: neither, a shim, a rotation, and both."""
    from scipy.spatial.transform import Rotation

    pulse = pp.make_block_pulse(
        flip_angle=0.5, duration=1e-3, system=system, use="excitation"
    )
    gradient = pp.make_trapezoid("x", area=600, duration=1e-3, system=system)
    shim = pp.make_rf_shim(np.array([1 + 0j, 0.5j]))
    other = pp.make_rf_shim(np.array([0.25 + 0j, 1 + 0j]))

    sequence = pp.Sequence(system)
    sequence.add_block(pulse)
    sequence.add_block(pulse, shim)
    sequence.add_block(
        gradient, pp.make_rotation(Rotation.from_euler("z", 30, degrees=True))
    )
    sequence.add_block(
        pulse, other, pp.make_rotation(Rotation.from_euler("z", 60, degrees=True))
    )
    return sequence


def walked(sequence):
    """What each block's extension chain names, found the long way round."""
    found = {"rotation": [], "shim": []}
    for index in range(1, len(sequence) + 1):
        block = sequence.get_block(index)
        found["rotation"].append(getattr(block, "rotation", None) is not None)
        found["shim"].append(getattr(block, "rf_shim", None) is not None)
    return found


def columns(sequence):
    native = sequence._native
    return {
        "rotation": [int(value) > 0 for value in native.block_rotations()],
        "shim": [int(value) > 0 for value in native.block_shims()],
    }


def test_the_columns_say_what_the_chains_name(turned_and_shimmed):
    assert columns(turned_and_shimmed) == {
        "rotation": [False, False, True, True],
        "shim": [False, True, False, True],
    }
    assert columns(turned_and_shimmed) == walked(turned_and_shimmed)


def test_the_columns_survive_a_file(turned_and_shimmed, tmp_path):
    path = tmp_path / "promoted.seq"
    turned_and_shimmed.write(str(path))
    back = pp.Sequence(turned_and_shimmed.system)
    back.read(str(path))

    assert columns(back) == columns(turned_and_shimmed)
    assert columns(back) == walked(back)


def test_the_columns_survive_deduplication(turned_and_shimmed):
    turned_and_shimmed.remove_duplicates()

    assert columns(turned_and_shimmed) == walked(turned_and_shimmed)


def test_rewriting_a_block_rewrites_its_columns(turned_and_shimmed, system):
    """set_block puts a block where another was, columns and all."""
    turned_and_shimmed.set_block(4, pp.make_delay(1e-3))

    assert columns(turned_and_shimmed)["rotation"] == [False, False, True, False]
    assert columns(turned_and_shimmed)["shim"] == [False, True, False, False]
    assert columns(turned_and_shimmed) == walked(turned_and_shimmed)


def test_the_file_is_untouched_by_the_promotion(turned_and_shimmed, tmp_path):
    """What is written is what a block's chain says, not what its row holds."""
    first = tmp_path / "once.seq"
    again = tmp_path / "twice.seq"
    turned_and_shimmed.write(str(first))
    back = pp.Sequence(turned_and_shimmed.system)
    back.read(str(first))
    back.write(str(again))

    assert first.read_bytes() == again.read_bytes()
    assert b"ROTATIONS" in first.read_bytes()
    assert b"RF_SHIMS" in first.read_bytes()


def test_the_block_table_a_caller_sees_is_still_what_a_file_carries(
    turned_and_shimmed,
):
    """The columns are internal: `block_events` is the six the format has."""
    assert turned_and_shimmed._native.block_events().shape == (4, 6)
