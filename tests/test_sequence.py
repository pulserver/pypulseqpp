"""The event libraries and the block table.

A sequence is libraries of events plus a table saying which of them play
together and for how long. These tests hold the parts of that model a caller
depends on: that ids are handed out in append order, that a block keeps the
ids it was given, and that collapsing duplicates renumbers every reference to
them without changing what plays.
"""

import numpy as np
import pytest

from pypulseqpp import _ext

TRAPEZOID = np.array([1.0e6, 100e-6, 1.0e-3, 100e-6, 0.0])
BLOCK_DURATION = 1.2e-3


@pytest.fixture
def sequence():
    return _ext.Sequence()


def test_a_new_sequence_holds_nothing(sequence):
    assert sequence.num_blocks() == 0
    assert len(sequence) == 0
    assert sequence.duration() == 0.0


def test_ids_are_handed_out_in_append_order(sequence):
    assert sequence.register_trap(TRAPEZOID) == 1
    assert sequence.register_trap(TRAPEZOID * 2) == 2
    assert sequence.register_trap(TRAPEZOID * 3) == 3


def test_an_identical_row_gets_an_id_of_its_own(sequence):
    first = sequence.register_trap(TRAPEZOID)
    assert sequence.register_trap(TRAPEZOID) != first


def test_a_block_keeps_the_event_ids_it_was_given(sequence):
    sequence.register_trap(TRAPEZOID)
    sequence.add_block(0, 1, 0, 0, 0, 0, BLOCK_DURATION)

    block = sequence.get_block(1)
    assert (block.gx, block.rf, block.gy, block.gz, block.adc, block.ext) == (
        1,
        0,
        0,
        0,
        0,
        0,
    )
    assert block.duration == pytest.approx(BLOCK_DURATION)


def test_block_indices_start_at_one(sequence):
    indices = [sequence.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION) for _ in range(3)]
    assert indices == [1, 2, 3]


def test_the_duration_is_the_sum_of_the_blocks(sequence):
    for _ in range(4):
        sequence.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION)
    assert sequence.duration() == pytest.approx(4 * BLOCK_DURATION)


def test_collapsing_duplicates_renumbers_the_blocks_that_referenced_them(sequence):
    sequence.register_trap(TRAPEZOID)
    sequence.register_trap(TRAPEZOID)
    sequence.register_trap(TRAPEZOID * 2)
    for identifier in (1, 2, 3):
        sequence.add_block(0, identifier, 0, 0, 0, 0, BLOCK_DURATION)

    sequence.remove_duplicates()

    referenced = [sequence.get_block(index).gx for index in (1, 2, 3)]
    assert referenced == [1, 1, 2]


def test_collapsing_duplicates_leaves_the_blocks_alone(sequence):
    sequence.register_trap(TRAPEZOID)
    sequence.register_trap(TRAPEZOID)
    for identifier in (1, 2):
        sequence.add_block(0, identifier, 0, 0, 0, 0, BLOCK_DURATION)

    sequence.remove_duplicates()

    assert sequence.num_blocks() == 2
    assert sequence.duration() == pytest.approx(2 * BLOCK_DURATION)


def test_text_definitions_round_trip(sequence):
    sequence.set_definition("Name", "demo")
    assert sequence.definitions()["Name"] == "demo"


def test_numeric_definitions_round_trip_as_a_list(sequence):
    sequence.set_definition("FOV", [0.256, 0.256, 0.005])
    sequence.set_definition("TE", 0.005)

    definitions = sequence.definitions()
    assert definitions["FOV"] == pytest.approx([0.256, 0.256, 0.005])
    assert definitions["TE"] == pytest.approx([0.005])


def test_label_ids_are_the_numbers_the_file_format_carries(sequence):
    assert sequence.label_id("SLC") == 1
    assert sequence.label_id("LIN") == 8
    assert sequence.label_id("NOISE") == 16


def test_a_label_id_maps_back_to_its_name(sequence):
    assert sequence.label_name(sequence.label_id("LIN")) == "LIN"


def test_extension_ids_are_minted_in_the_order_they_are_asked_for(sequence):
    assert sequence.extension_type_id("LABELSET") == 1
    assert sequence.extension_type_id("TRIGGERS") == 2
    assert sequence.extension_type_id("LABELSET") == 1


def test_an_extension_id_can_be_pinned_to_match_a_file(sequence):
    sequence.set_extension_type_id("DELAYS", 3)
    assert sequence.extension_type_id("DELAYS") == 3


def test_adding_a_block_goes_through_the_fast_calling_convention():
    # A design loop calls this once per block and a protocol-scale scan has
    # millions of them, so it is bound by hand with METH_FASTCALL rather than
    # through pybind11. A pybind11 binding would show up here as a
    # builtin_function_or_method, and would cost an order of magnitude more.
    assert type(_ext.Sequence.add_block).__name__ == "method_descriptor"


def test_the_block_table_is_read_back_without_copying(sequence):
    sequence.register_trap(TRAPEZOID)
    for _ in range(3):
        sequence.add_block(0, 1, 0, 0, 0, 0, BLOCK_DURATION)

    events = sequence.block_events()
    durations = sequence.block_durations()

    assert events.shape == (3, 6)
    assert durations.shape == (3,)
    np.testing.assert_array_equal(events[:, 1], [1, 1, 1])
    np.testing.assert_allclose(durations, BLOCK_DURATION)


def test_a_view_is_a_snapshot_that_later_writes_do_not_reach(sequence):
    sequence.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION)
    sequence.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION)
    taken_before = sequence.block_durations()

    sequence.set_block_duration(2, 5e-3)

    assert taken_before[1] == pytest.approx(BLOCK_DURATION)
    assert sequence.block_durations()[1] == pytest.approx(5e-3)
    assert sequence.get_block(2).duration == pytest.approx(5e-3)


def test_a_view_stays_valid_when_the_block_table_grows(sequence):
    for _ in range(4):
        sequence.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION)
    taken_before = sequence.block_durations()

    # Growing the table moves it. The view holds the buffer it was taken over,
    # so it reads what it always read rather than freed memory.
    for _ in range(10_000):
        sequence.add_block(0, 0, 0, 0, 0, 0, 2e-3)

    assert len(taken_before) == 4
    np.testing.assert_allclose(taken_before, BLOCK_DURATION)
    assert sequence.num_blocks() == 10_004


def test_a_view_outlives_the_sequence_it_was_taken_from():
    short_lived = _ext.Sequence()
    for _ in range(3):
        short_lived.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION)
    durations = short_lived.block_durations()

    del short_lived

    np.testing.assert_allclose(durations, BLOCK_DURATION)


def test_rewriting_a_block_outside_the_table_is_refused(sequence):
    sequence.add_block(0, 0, 0, 0, 0, 0, BLOCK_DURATION)
    with pytest.raises(IndexError):
        sequence.set_block_duration(9, 1e-3)
