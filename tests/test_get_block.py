"""Decoded-event parity and duration-preserving block replay."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert
import test_parity

from pypulseqpp import _ext

#: What each kind of event carries, and so what has to agree.
DECODED_FIELDS = {
    "rf": [
        "type",
        "signal",
        "t",
        "shape_dur",
        "center",
        "delay",
        "freq_offset",
        "phase_offset",
        "freq_ppm",
        "phase_ppm",
        "use",
    ],
    "trap": [
        "type",
        "channel",
        "amplitude",
        "rise_time",
        "flat_time",
        "fall_time",
        "delay",
        "area",
        "flat_area",
    ],
    "grad": [
        "type",
        "channel",
        "waveform",
        "tt",
        "shape_dur",
        "delay",
        "first",
        "last",
        "area",
    ],
    "adc": [
        "type",
        "num_samples",
        "dwell",
        "delay",
        "freq_offset",
        "phase_offset",
        "freq_ppm",
        "phase_ppm",
    ],
}


def assert_same(expected, got, where):
    """Hold one field equal, whether it is a number, a name or a waveform."""
    if isinstance(expected, np.ndarray) or isinstance(got, np.ndarray):
        expected, got = np.asarray(expected), np.asarray(got)
        assert expected.shape == got.shape, where
        assert np.allclose(expected, got, rtol=1e-9, atol=1e-12), where
    elif isinstance(expected, float):
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12), where
    else:
        assert got == expected, where


def test_a_decoded_block_holds_what_the_toolbox_decodes(
    reference_name, build_reference
):
    if reference_name in test_parity.NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip("the reference leaves a gap in the block numbering")

    theirs = build_reference()
    core = convert.to_core(theirs)

    for index in theirs.block_events:
        their_block = theirs.get_block(index)
        our_block = core.decode_block(index)

        for slot in ("rf", "gx", "gy", "gz", "adc"):
            their_event = getattr(their_block, slot, None)
            our_event = our_block[slot]
            assert (their_event is None) == (our_event is None), (
                f"{reference_name} block {index}: {slot}"
            )
            if their_event is None:
                continue
            for field in DECODED_FIELDS[their_event.type]:
                assert_same(
                    getattr(their_event, field),
                    getattr(our_event, field),
                    f"{reference_name} block {index}: {slot}.{field}",
                )


def test_a_decoded_block_says_how_long_it_lasts(reference_name, build_reference):
    if reference_name in test_parity.NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip("the reference leaves a gap in the block numbering")

    theirs = build_reference()
    core = convert.to_core(theirs)

    for index in theirs.block_events:
        assert core.decode_block(index)["block_duration"] == pytest.approx(
            theirs.block_durations[index]
        )


def decoded(core, index):
    """The decoded block as the object `add_block` and `set_block` unpack."""
    return SimpleNamespace(**core.decode_block(index))


def test_reading_every_block_out_and_back_writes_the_same_file(
    reference_name, build_reference
):
    if reference_name in test_parity.NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip("the reference leaves a gap in the block numbering")

    core = convert.to_core(build_reference())
    before = _ext.write_text(core, False)
    shapes = core.num_shapes()

    for index in range(1, core.num_blocks() + 1):
        core.set_block_events(index, decoded(core, index))

    assert core.num_shapes() == shapes, "a waveform was registered twice"
    core.remove_duplicates()
    assert _ext.write_text(core, False) == before


def test_a_block_that_plays_nothing_keeps_how_long_it_waits():
    """A pure delay is a duration rather than an event, so only the block
    carries it."""
    core = _ext.Sequence()
    core.set_rasters(1e-6, 10e-6, 100e-9, 10e-6)
    core.add_block(0, 0, 0, 0, 0, 0, 5e-3)

    core.set_block_events(1, decoded(core, 1))

    assert core.block_durations()[0] == pytest.approx(5e-3)


def test_a_whole_block_can_be_handed_to_another_sequence(
    reference_name, build_reference
):
    """`add_block(other.get_block(i))`, which is how a block is moved."""
    if reference_name in test_parity.NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip("the reference leaves a gap in the block numbering")

    theirs = build_reference()
    source = convert.to_core(theirs)

    copied = _ext.Sequence()
    copied.set_rasters(
        theirs.rf_raster_time,
        theirs.grad_raster_time,
        theirs.adc_raster_time,
        theirs.block_duration_raster,
    )
    # An extension is numbered in the order its name is first seen, so a copy
    # built block by block numbers them by where they appear rather than by
    # where the original put them. Pinning them first is what a caller wanting
    # the same bytes does, and what the toolboxes do too.
    for number in range(1, source.num_extensions() + 1):
        name = source.extension_type_name(number)
        if name:
            copied.set_extension_type_id(name, number)

    for index in range(1, source.num_blocks() + 1):
        copied.add_block_events(decoded(source, index))

    assert copied.num_blocks() == source.num_blocks()
    assert list(copied.block_durations()) == pytest.approx(
        list(source.block_durations())
    )
    copied.remove_duplicates()
    source.remove_duplicates()
    assert _ext.write_text(copied, False) == _ext.write_text(source, False)
