"""What the scan repeats, found from the definitions rather than the blocks.

A scan is a handful of things played over and over with different numbers in
them. The structural fork already says which: two blocks playing the same
things for the same length share a definition id whatever their amplitudes,
so a gradient echo's definition stream reads 1 2 3 4 1 2 3 4 whatever its
phase encode is doing. Finding the repeating unit is then finding the period
of an array of integers, and that is a pass in C++ rather than a comparison
of blocks event by event.

What is held here is the period, where it starts, and when the answer goes
stale.
"""

import math

import pytest

import pypulseqpp as pp


@pytest.fixture
def gradient_echo():
    """One shot is three blocks: excite, encode, read."""

    def make(lines=8, prologue=0):
        sequence = pp.Sequence(pp.Opts())
        pulse, select, rephase = pp.make_sinc_pulse(
            flip_angle=math.pi / 12,
            duration=1e-3,
            slice_thickness=3e-3,
            use="excitation",
            return_gz=True,
        )
        read = pp.make_trapezoid("x", area=1000, duration=2e-3)
        window = pp.make_adc(num_samples=64, duration=2e-3)
        encode = pp.make_trapezoid("y", area=500, duration=1e-3)

        for _ in range(prologue):
            sequence.add_block(pp.make_delay(5e-3))
        for line in range(lines):
            sequence.add_block(pulse, select)
            sequence.add_block(pp.scale_grad(encode, (line / lines) or 1e-9), rephase)
            sequence.add_block(read, window)
        return sequence

    return make


def test_a_scan_repeats_at_the_length_of_one_shot(gradient_echo):
    sequence = gradient_echo(lines=8)

    size, start = sequence._detect_tr()

    assert size == 3
    assert start == 1


def test_a_phase_encode_does_not_make_a_shot_different(gradient_echo):
    """Every line has its own amplitude and they are all one definition."""
    sequence = gradient_echo(lines=32)

    size, _ = sequence._detect_tr()

    assert size == 3
    assert len(sequence) == 96


def test_the_blocks_before_the_scan_starts_are_not_part_of_it(gradient_echo):
    """Dummy shots and preparation are the prologue, not the repeat."""
    sequence = gradient_echo(lines=8, prologue=3)

    size, start = sequence._detect_tr()

    assert size == 3
    assert start == 4


def test_a_sequence_that_plays_each_position_once_does_not_repeat(gradient_echo):
    assert gradient_echo(lines=1)._detect_tr() == (0, 1)


def test_a_sequence_with_nothing_in_it_does_not_repeat():
    assert pp.Sequence(pp.Opts())._detect_tr() == (0, 1)


def test_the_repeating_unit_is_recorded_as_a_definition(gradient_echo):
    sequence = gradient_echo(lines=8)

    size, _ = sequence._detect_tr()

    assert sequence.get_definition("TRsize") == pytest.approx([size])


def test_a_recorded_repeating_unit_survives_a_file(gradient_echo, tmp_path):
    """Written into `[DEFINITIONS]`, so a reader does not work it out again."""
    sequence = gradient_echo(lines=8)
    sequence._detect_tr()
    path = tmp_path / "gre.seq"
    sequence.write(str(path))

    loaded = pp.Sequence(pp.Opts())
    loaded.read(str(path))

    assert loaded.get_definition("TRsize") == pytest.approx([3])
    assert loaded._detect_tr() == (3, 1)


def test_writing_does_not_record_it_unasked(gradient_echo, tmp_path):
    """Nothing writes `TRsize` unless the answer was asked for."""
    path = tmp_path / "gre.seq"
    gradient_echo(lines=8).write(str(path))

    assert "TRsize" not in path.read_text()


def test_adding_a_block_makes_the_answer_stale(gradient_echo):
    sequence = gradient_echo(lines=8)
    assert sequence._native.repetition() == (3, 0)

    sequence.add_block(pp.make_delay(5e-3))

    # The new block ends the sequence, so the shot no longer runs to the end.
    assert sequence._native.repetition() != (3, 0)


def test_collapsing_duplicates_makes_the_answer_stale():
    """The repeat only becomes visible once equal shapes are one definition.

    Six equal pulses built one at a time are six definitions until
    deduplication collapses them, so before it there is no repeat to find and
    after it there is. A remembered answer would still say there is none.
    """
    sequence = pp.Sequence(pp.Opts())
    for _ in range(6):
        # A fresh pulse each time: equal, but registered as its own shapes.
        sequence.add_block(
            pp.make_block_pulse(math.pi / 6, duration=1e-3, use="excitation")
        )
        sequence.add_block(pp.make_delay(2e-3))

    assert sequence._native.repetition() == (0, 0)

    sequence.remove_duplicates(in_place=True)

    assert sequence._native.repetition() == (2, 0)


def test_collapsing_duplicates_twice_does_the_pass_once():
    """The flag either method sets is what makes the second call free."""
    sequence = pp.Sequence(pp.Opts())
    for _ in range(4):
        sequence.add_block(
            pp.make_block_pulse(math.pi / 6, duration=1e-3, use="excitation")
        )
    sequence.remove_duplicates(in_place=True)
    collapsed = sequence._native.num_rf()

    sequence.remove_duplicates(in_place=True)

    assert sequence._native.num_rf() == collapsed


def test_a_repeating_unit_can_be_located_when_its_size_is_known(gradient_echo):
    """For a caller that already knows the period."""
    sequence = gradient_echo(lines=8, prologue=3)

    assert sequence._native.locate_repetition(3) == (3, 3)
    assert sequence._native.locate_repetition(4) == (0, 0)


def test_finding_the_repeat_of_a_long_scan_is_one_pass(gradient_echo):
    """A guard on where the work happens: this is an array, not the blocks."""
    sequence = gradient_echo(lines=20000)

    size, start = sequence._detect_tr()

    assert (size, start) == (3, 1)
    assert len(sequence) == 60000


def test_labelling_a_pulse_splits_the_definition_it_shared(tmp_path):
    """What a pulse is for is part of which definition it is.

    An excitation and a refocusing off the same shape differ only in
    amplitude, which is an instance parameter, so while both are unlabelled
    they are one definition and the sequence looks like it repeats every
    block. Labelling them splits that definition in two, and the tables the
    repeat is read off have to say so.
    """
    system = pp.Opts()
    spin_echo = pp.Sequence(system)
    for _ in range(3):
        spin_echo.add_block(
            pp.make_block_pulse(
                math.pi / 2, duration=1e-3, system=system, use="excitation"
            )
        )
        spin_echo.add_block(
            pp.make_block_pulse(math.pi, duration=1e-3, system=system, use="refocusing")
        )
    path = tmp_path / "se.seq"
    # 1.4.1 has nowhere to record what a pulse is for.
    spin_echo.write_v141(str(path))

    loaded = pp.Sequence(system)
    loaded.read(str(path))
    assert loaded._native.num_rf_definitions() == 1
    assert loaded._native.repetition() == (1, 0)

    loaded._native.detect_rf_uses(system.B0, system.gamma)

    assert loaded._native.num_rf_definitions() == 2
    assert list(loaded._native.instance_definitions()) == [1, 2, 1, 2, 1, 2]
    assert loaded._detect_tr() == (2, 1)
