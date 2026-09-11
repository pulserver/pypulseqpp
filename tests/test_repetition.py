"""Block-definition periods, prologues and cache invalidation."""

import math

import pytest
from scipy.spatial.transform import Rotation

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


def test_a_phase_encode_table_is_one_definition_at_many_amplitudes():
    """One gradient at its largest step, scaled per line.

    Which is what makes a table read as one repetition, and it is not left to
    the caller: every readout module builds its encodes with
    `make_phase_encoding` for the step and `scale_grad` for the line, so the
    amplitude is a column of the row and the timings never move. Ask
    `make_trapezoid` for an area per line instead and it derives its own rise
    and fall from each, which is a definition per line and a scan that reads
    as though it never does the same thing twice.
    """
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    step = pp.make_phase_encoding("y", 0.24 / 64, system=system)

    sequence = pp.Sequence(system)
    for line in (-1.0, -0.5, 0.25, 1.0):
        sequence.add_block(pp.scale_grad(step, line))

    assert sequence._native.num_grad_definitions() == 1
    assert sequence._detect_tr() == (1, 1)


def test_a_line_scaled_to_zero_is_the_same_definition_at_no_amplitude():
    """How a calibration line is acquired without the encode.

    Not a block with one fewer event, which would be a definition of its own
    and would break the scan into pieces around it. The amplitude is the
    instance's, so zero is a number in the row.
    """
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    step = pp.make_phase_encoding("y", 0.24 / 64, system=system)

    sequence = pp.Sequence(system)
    for line in (1.0, 0.0, -1.0):
        sequence.add_block(pp.scale_grad(step, line))

    assert sequence._native.num_grad_definitions() == 1
    assert sequence._detect_tr() == (1, 1)
    assert float(sequence.get_block(2).gy.amplitude) == 0.0


def test_a_run_of_waits_repeats_every_block_however_long_each_waits():
    """A pure delay is one definition and its duration is not part of it.

    A block that plays nothing is a position an interpreter waits at, and how
    long it waits there is set at run time -- so a fill and the pad after it
    are one position waited at twice, not two things.
    """
    sequence = pp.Sequence(pp.Opts())
    for duration in (1e-3, 2e-3, 3e-3):
        sequence.add_block(pp.make_delay(duration))

    assert sequence._detect_tr() == (1, 1)


def test_the_repeat_survives_a_shift(gradient_echo):
    """A prescription writes phases onto rows; it does not restructure a scan."""
    sequence = gradient_echo(lines=8)
    moved = pp.TransformFOV(translation=(0.01, 0.0, 0.0)).apply_to_sequence(sequence)

    assert moved._detect_tr() == sequence._detect_tr()


def test_the_repeat_survives_a_rotation(gradient_echo):
    """A rotation is four numbers on a block, not a new set of waveforms, so
    it cannot split one readout into many."""
    sequence = gradient_echo(lines=8)
    turned = pp.TransformFOV(
        rotation=Rotation.from_euler("z", 45, degrees=True)
    ).apply_to_sequence(sequence)

    assert turned._detect_tr() == sequence._detect_tr()


@pytest.mark.parametrize("dummies", [0, 2])
def test_a_slice_acquisition_with_its_preparation_is_one_repetition(dummies):
    """A 2D balanced acquisition: per slice a half-flip preparation, dummy
    shots and four lines. The lines alone repeat too, but only inside the last
    slice; the slice is the longest stretch that repeats, so it is the TR."""
    system = pp.Opts(rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)
    half = pp.make_block_pulse(math.pi / 8, duration=0.5e-3, system=system)
    flip = pp.make_block_pulse(math.pi / 4, duration=0.5e-3, system=system)
    readout = pp.make_trapezoid("x", area=1000, duration=1.5e-3, system=system)
    adc = pp.make_adc(64, duration=0.8e-3, delay=readout.rise_time, system=system)
    sequence = pp.Sequence(system)
    for _ in range(3):
        sequence.add_block(half)
        for line in range(dummies + 4):
            sequence.add_block(flip, readout, *([adc] if line >= dummies else []))

    assert sequence._detect_tr() == (1 + dummies + 4, 1)


def test_a_prologue_before_the_slice_loop_stays_a_prologue():
    """Blocks played once before the outer loop are not folded into it."""
    system = pp.Opts()
    half = pp.make_block_pulse(math.pi / 8, duration=0.5e-3, system=system)
    flip = pp.make_block_pulse(math.pi / 4, duration=0.5e-3, system=system)
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_block_pulse(math.pi, duration=2e-3, system=system))
    sequence.add_block(pp.make_delay(10e-3))
    for _ in range(3):
        sequence.add_block(half)
        for _ in range(4):
            sequence.add_block(flip)

    assert sequence._detect_tr() == (5, 3)
