"""Native definition/instance partitioning and structural block equivalence."""

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert

from pypulseqpp import _ext

# Instance columns; see INSTANCE_WIDTH in sequence.hpp.
GX_AMPLITUDE, GX_SHAPE = 0, 1
GY_AMPLITUDE = 2
RF_AMPLITUDE, RF_PHASE = 6, 8
ADC_PHASE_MODULATION = 15


def trapezoid(sequence, amplitude, rise=1e-4, flat=1e-3, fall=1e-4, delay=0.0):
    return sequence.register_trap(np.array([amplitude, rise, flat, fall, delay]))


def rf(sequence, magnitude, phase_shape, time, amplitude=500.0, delay=1e-4, phase=0.0):
    row = np.zeros(10)
    row[0], row[1], row[2], row[3] = amplitude, magnitude, phase_shape, time
    row[5], row[9] = delay, phase
    return sequence.register_rf(row, "e")


def adc(sequence, samples=256.0, dwell=1e-5, delay=0.0, phase_modulation=0.0):
    row = np.zeros(8)
    row[0], row[1], row[2], row[7] = samples, dwell, delay, phase_modulation
    return sequence.register_adc(row)


def shortest_period(ids):
    """The smallest period the stream repeats at, or its whole length."""
    ids = np.asarray(ids)
    for period in range(1, len(ids) + 1):
        if np.array_equal(ids[period:], ids[: len(ids) - period]):
            return period
    return len(ids)


@pytest.fixture
def gradient_echo():
    """A 2D gradient echo at 128 lines: four blocks a line, the encode stepping."""
    sequence = _ext.Sequence()
    magnitude = sequence.register_shape(100, np.linspace(0, 1, 100))
    phase = sequence.register_shape(100, np.zeros(100))
    time = sequence.register_shape(100, np.arange(100) * 1e-5)
    excitation = rf(sequence, magnitude, phase, time)
    slice_select = trapezoid(sequence, 1000.0)
    prewinder = trapezoid(sequence, -2000.0, flat=5e-4)
    rephaser = trapezoid(sequence, -1000.0, flat=4e-4)
    readout = trapezoid(sequence, 2000.0, flat=2e-3)
    spoiler = trapezoid(sequence, 3000.0, flat=6e-4)
    window = adc(sequence)

    lines = 128
    for line in range(lines):
        step = (line - lines / 2) * 10.0
        # As a phase-encode loop scales it: same timing, new amplitude.
        encode = trapezoid(sequence, step, flat=5e-4)
        rewind = trapezoid(sequence, -step, flat=6e-4)
        sequence.add_block(excitation, 0, 0, slice_select, 0, 0, 1.2e-3)
        sequence.add_block(0, prewinder, encode, rephaser, 0, 0, 7e-4)
        sequence.add_block(0, readout, 0, 0, window, 0, 2.2e-3)
        sequence.add_block(0, spoiler, rewind, 0, 0, 0, 8e-4)
    return sequence


def test_a_gradient_echo_repeats_at_its_four_block_positions(gradient_echo):
    assert gradient_echo.num_block_definitions() == 4
    assert shortest_period(gradient_echo.instance_definitions()) == 4


def test_a_phase_encode_is_one_definition_played_at_many_amplitudes(gradient_echo):
    # Five gradient timings -- slice select, prewinder, rephaser, readout,
    # spoiler -- with the encode sharing the prewinder's and the rewinder the
    # spoiler's, against 261 gradient events.
    assert gradient_echo.num_gradients() == 261
    assert gradient_echo.num_grad_definitions() == 5

    encodes = gradient_echo.instance_parameters()[1::4, GY_AMPLITUDE]
    np.testing.assert_allclose(encodes, (np.arange(128) - 64) * 10.0)


def test_a_waveform_that_changes_every_shot_is_one_definition():
    """A sparkling readout: a new arm each shot, one timing, one definition."""
    sequence = _ext.Sequence()
    times = sequence.register_shape(64, np.arange(64) * 1e-5)
    window = adc(sequence, samples=64.0)

    shapes = []
    for shot in range(16):
        waveform = sequence.register_shape(64, np.sin(np.linspace(0, 3 + shot, 64)))
        shapes.append(waveform)
        arm = sequence.register_arbitrary(
            np.array([1500.0, 0.0, 0.0, waveform, times, 0.0])
        )
        sequence.add_block(0, arm, 0, 0, window, 0, 7e-4)

    assert sequence.num_grad_definitions() == 1
    assert sequence.num_block_definitions() == 1
    played = sequence.instance_parameters()[:, GX_SHAPE]
    np.testing.assert_array_equal(played, np.array(shapes, dtype=float))


def test_an_rf_pulse_is_one_definition_however_its_amplitude_and_phase_are_set():
    """RF spoiling steps the phase every shot; the pulse does not change."""
    sequence = _ext.Sequence()
    magnitude = sequence.register_shape(64, np.linspace(0, 1, 64))
    zeros = sequence.register_shape(64, np.zeros(64))
    time = sequence.register_shape(64, np.arange(64) * 1e-5)

    phases = [(shot * (shot + 1) / 2 * 117.0) % 360.0 for shot in range(32)]
    for shot, phase in enumerate(phases):
        pulse = rf(
            sequence, magnitude, zeros, time, amplitude=500.0 + shot, phase=phase
        )
        sequence.add_block(pulse, 0, 0, 0, 0, 0, 1e-3)

    assert sequence.num_rf_definitions() == 1
    assert sequence.num_block_definitions() == 1
    np.testing.assert_allclose(sequence.instance_parameters()[:, RF_PHASE], phases)


def test_a_pulse_of_a_different_shape_is_a_different_definition():
    sequence = _ext.Sequence()
    time = sequence.register_shape(64, np.arange(64) * 1e-5)
    zeros = sequence.register_shape(64, np.zeros(64))
    sinc = sequence.register_shape(64, np.sinc(np.linspace(-2, 2, 64)))
    ramp = sequence.register_shape(64, np.linspace(0, 1, 64))

    rf(sequence, sinc, zeros, time)
    rf(sequence, ramp, zeros, time)

    assert sequence.num_rf_definitions() == 2


def test_a_pulse_at_a_different_delay_is_a_different_definition():
    sequence = _ext.Sequence()
    time = sequence.register_shape(64, np.arange(64) * 1e-5)
    zeros = sequence.register_shape(64, np.zeros(64))
    ramp = sequence.register_shape(64, np.linspace(0, 1, 64))

    rf(sequence, ramp, zeros, time, delay=1e-4)
    rf(sequence, ramp, zeros, time, delay=2e-4)

    assert sequence.num_rf_definitions() == 2


def test_a_readout_digitised_two_ways_is_one_definition_played_twice():
    """Which ADC a position uses is the instance's business, not the block's."""
    sequence = _ext.Sequence()
    readout = trapezoid(sequence, 2000.0, flat=2e-3)
    short = adc(sequence, samples=128.0)
    long = adc(sequence, samples=256.0)

    sequence.add_block(0, readout, 0, 0, short, 0, 2.2e-3)
    sequence.add_block(0, readout, 0, 0, long, 0, 2.2e-3)

    assert sequence.num_adc_definitions() == 2
    assert sequence.num_block_definitions() == 1
    np.testing.assert_array_equal(sequence.instance_adc_definitions(), [1, 2])


def test_a_shot_that_acquires_nothing_shares_the_definition_of_one_that_does():
    """A preparation shot plays the imaging shot with the digitiser off."""
    sequence = _ext.Sequence()
    readout = trapezoid(sequence, 2000.0, flat=2e-3)
    window = adc(sequence)

    sequence.add_block(0, readout, 0, 0, 0, 0, 2.2e-3)
    sequence.add_block(0, readout, 0, 0, window, 0, 2.2e-3)

    assert sequence.num_block_definitions() == 1
    np.testing.assert_array_equal(sequence.instance_adc_definitions(), [0, 1])


def test_the_adc_phase_modulation_travels_with_the_instance():
    sequence = _ext.Sequence()
    readout = trapezoid(sequence, 2000.0, flat=2e-3)
    first = sequence.register_shape(8, np.arange(8) * 0.1)
    second = sequence.register_shape(8, np.arange(8) * 0.2)

    sequence.add_block(
        0, readout, 0, 0, adc(sequence, phase_modulation=first), 0, 2.2e-3
    )
    sequence.add_block(
        0, readout, 0, 0, adc(sequence, phase_modulation=second), 0, 2.2e-3
    )

    assert sequence.num_adc_definitions() == 1
    assert sequence.num_block_definitions() == 1
    np.testing.assert_array_equal(
        sequence.instance_parameters()[:, ADC_PHASE_MODULATION], [first, second]
    )


def test_setting_a_block_moves_it_to_the_definition_it_now_plays():
    sequence = _ext.Sequence()
    short = trapezoid(sequence, 2000.0, flat=1e-3)
    long = trapezoid(sequence, 2000.0, flat=2e-3)
    sequence.add_block(0, short, 0, 0, 0, 0, 1.2e-3)
    sequence.add_block(0, long, 0, 0, 0, 0, 2.2e-3)
    assert sequence.instance_definitions().tolist() == [1, 2]

    sequence.set_block(2, 0, short, 0, 0, 0, 0, 1.2e-3)

    assert sequence.instance_definitions().tolist() == [1, 1]


def test_two_pulses_the_file_cannot_tell_apart_are_one_definition_once_collapsed():
    """The shapes are equal, so after deduplication so are the definitions.

    Registered separately, equal shapes are two entries with two ids, and a
    key built from those ids splits one pulse into two definitions.
    Deduplication merges the shapes; re-deriving the definitions from what
    survives is what makes the split go away.
    """
    sequence = _ext.Sequence()
    magnitude = np.linspace(0, 1, 64)
    zeros = np.zeros(64)
    time = np.arange(64) * 1e-5

    for _ in range(2):
        pulse = rf(
            sequence,
            sequence.register_shape(64, magnitude),
            sequence.register_shape(64, zeros),
            sequence.register_shape(64, time),
        )
        sequence.add_block(pulse, 0, 0, 0, 0, 0, 1e-3)

    assert sequence.num_shapes() == 6
    assert sequence.num_rf_definitions() == 2
    assert sequence.instance_definitions().tolist() == [1, 2]

    sequence.remove_duplicates()

    assert sequence.num_shapes() == 3
    assert sequence.num_rf_definitions() == 1
    assert sequence.num_block_definitions() == 1
    assert sequence.instance_definitions().tolist() == [1, 1]


def test_collapsing_duplicates_leaves_a_scan_that_was_already_distinct_alone(
    gradient_echo,
):
    before = gradient_echo.instance_definitions().copy()
    parameters = gradient_echo.instance_parameters().copy()

    gradient_echo.remove_duplicates()

    np.testing.assert_array_equal(gradient_echo.instance_definitions(), before)
    np.testing.assert_allclose(gradient_echo.instance_parameters(), parameters)
    assert gradient_echo.num_block_definitions() == 4


def test_a_pure_delay_is_one_definition_however_long_it_waits():
    """An interpreter sets how long it waits there, so the wait is not the block.

    A TI fill and the pad that follows it are one position waited at for two
    different times, which is a runtime parameter rather than two sequences.
    """
    sequence = _ext.Sequence()
    waits = [20e-3, 50e-3, 100e-3, 480e-3]
    for wait in waits:
        sequence.add_block(0, 0, 0, 0, 0, 0, wait)

    assert sequence.num_block_definitions() == 1
    assert sequence.instance_definitions().tolist() == [1, 1, 1, 1]
    # The duration is the per-playout parameter, and stays where it was.
    np.testing.assert_allclose(sequence.block_durations(), waits)


def test_a_block_that_plays_something_keys_on_its_duration():
    """A padded block lasts as long as it was told to, which is its own."""
    sequence = _ext.Sequence()
    readout = trapezoid(sequence, 2000.0, flat=2e-3)

    sequence.add_block(0, readout, 0, 0, 0, 0, 2.2e-3)
    sequence.add_block(0, readout, 0, 0, 0, 0, 3.0e-3)

    assert sequence.num_block_definitions() == 2


def chain(sequence, name, reference):
    """A one-node extension chain of type ``name``."""
    return sequence.chain_extension(sequence.extension_type_id(name), reference, 0)


def test_a_wait_that_fires_a_trigger_keys_on_its_duration():
    """A trigger is played, so the block holding it plays something."""
    sequence = _ext.Sequence()
    trigger = sequence.register_trigger(np.array([1.0, 1.0, 0.0, 2e-3]))
    fires = chain(sequence, "TRIGGERS", trigger)

    sequence.add_block(0, 0, 0, 0, 0, fires, 2e-3)
    sequence.add_block(0, 0, 0, 0, 0, fires, 4e-3)

    assert sequence.num_block_definitions() == 2


def test_a_wait_that_only_sets_a_label_is_still_a_pure_delay():
    """A label is bookkeeping for the reconstruction; nothing is played."""
    sequence = _ext.Sequence()
    label = sequence.register_label_set(1, sequence.label_id("LIN"))
    counted = chain(sequence, "LABELSET", label)

    sequence.add_block(0, 0, 0, 0, 0, counted, 2e-3)
    sequence.add_block(0, 0, 0, 0, 0, counted, 4e-3)

    assert sequence.num_block_definitions() == 1


def test_a_wait_that_is_turned_by_a_rotation_is_still_a_pure_delay():
    """A rotation remaps axes, and a block with no gradient has none to remap."""
    sequence = _ext.Sequence()
    rotation = sequence.register_rotation(np.array([1.0, 0.0, 0.0, 0.0]))
    turned = chain(sequence, "ROTATIONS", rotation)

    sequence.add_block(0, 0, 0, 0, 0, turned, 2e-3)
    sequence.add_block(0, 0, 0, 0, 0, turned, 4e-3)

    assert sequence.num_block_definitions() == 1


def test_a_trigger_deeper_in_a_chain_still_counts():
    """What a block plays is the whole chain's business, not its first link."""
    sequence = _ext.Sequence()
    trigger = sequence.register_trigger(np.array([1.0, 1.0, 0.0, 2e-3]))
    label = sequence.register_label_set(1, sequence.label_id("LIN"))
    fires = sequence.chain_extension(sequence.extension_type_id("TRIGGERS"), trigger, 0)
    behind = sequence.chain_extension(
        sequence.extension_type_id("LABELSET"), label, fires
    )

    sequence.add_block(0, 0, 0, 0, 0, behind, 2e-3)
    sequence.add_block(0, 0, 0, 0, 0, behind, 4e-3)

    assert sequence.num_block_definitions() == 2


def test_pinning_the_trigger_type_late_still_sorts_the_delays_out():
    """Reading a file forces the numbering after the chains are already built."""
    sequence = _ext.Sequence()
    trigger = sequence.register_trigger(np.array([1.0, 1.0, 0.0, 2e-3]))
    # Chained under a name that is not TRIGGERS yet, as a reader does before
    # it has seen the `extension TRIGGERS n` line.
    fires = sequence.chain_extension(7, trigger, 0)
    sequence.add_block(0, 0, 0, 0, 0, fires, 2e-3)
    sequence.add_block(0, 0, 0, 0, 0, fires, 4e-3)
    assert sequence.num_block_definitions() == 1

    sequence.set_extension_type_id("TRIGGERS", 7)

    assert sequence.num_block_definitions() == 2


# --------------------------------------------------------------------------
# The same split, held against the reference sequences rather than against a
# sequence written to have it. What a block's definition is made of is read
# out of the upstream libraries, so the partition our ids induce is compared
# with one derived independently of them.
# --------------------------------------------------------------------------


def quantised(seconds):
    """A time as the definition key holds it."""
    return round(float(seconds) * 1e9)


def trigger_type(seq):
    """The id upstream gave the `TRIGGERS` extension, or 0 if it has none."""
    for number, name in zip(
        seq.extension_numeric_idx, seq.extension_string_idx, strict=True
    ):
        if name == "TRIGGERS":
            return number
    return 0


def carries_a_trigger(seq, chain):
    """Whether an extension chain holds a trigger or a digital output."""
    wanted = trigger_type(seq)
    while chain:
        kind, _ref, chain = (int(value) for value in seq.extensions_library.data[chain])
        if wanted and kind == wanted:
            return True
    return False


#: What every pure delay is, standing where a definition would.
PURE_DELAY = "pure delay"


def upstream_definition(seq, block):
    """What the definition of ``block`` is made of, read from upstream."""
    events = seq.block_events[block]
    plays = any(int(events[i]) for i in (1, 2, 3, 4, 5))
    if not plays and not carries_a_trigger(seq, int(events[6])):
        return PURE_DELAY

    def pulse(identifier):
        if identifier == 0:
            return None
        row = seq.rf_library.data[identifier]
        return (
            row[1],
            row[2],
            row[3],
            quantised(row[4]),
            quantised(row[5]),
            seq.rf_library.type.get(identifier, "u"),
        )

    def gradient(identifier):
        if identifier == 0:
            return None
        row = seq.grad_library.data[identifier]
        if seq.grad_library.type[identifier] == "t":
            return ("t", *(quantised(row[i]) for i in (1, 2, 3, 4)))
        return ("g", row[4], quantised(row[5]))

    return (
        pulse(int(events[1])),
        gradient(int(events[2])),
        gradient(int(events[3])),
        gradient(int(events[4])),
        quantised(seq.block_durations[block]),
    )


def partition(labels):
    """The equivalence classes a labelling induces, as a set of block sets."""
    classes = {}
    for block, label in labels.items():
        classes.setdefault(label, set()).add(block)
    return {frozenset(blocks) for blocks in classes.values()}


def test_the_definitions_partition_the_blocks_the_way_their_content_does(
    build_reference,
):
    seq = build_reference()
    core = convert.to_core(seq)

    blocks = sorted(seq.block_events)
    ours = dict(zip(blocks, core.instance_definitions().tolist(), strict=True))
    theirs = {block: upstream_definition(seq, block) for block in blocks}

    assert partition(ours) == partition(theirs)
    assert core.num_block_definitions() == len(set(theirs.values()))


def test_every_block_names_a_definition(build_reference):
    core = convert.to_core(build_reference())
    definitions = core.instance_definitions()

    assert len(definitions) == core.num_blocks()
    assert definitions.min() >= 1
    assert definitions.max() <= core.num_block_definitions()


def test_collapsing_duplicates_twice_changes_nothing_the_second_time(build_reference):
    core = convert.to_core(build_reference())
    core.remove_duplicates()
    definitions = core.instance_definitions().copy()
    parameters = core.instance_parameters().copy()

    core.remove_duplicates()

    np.testing.assert_array_equal(core.instance_definitions(), definitions)
    np.testing.assert_allclose(core.instance_parameters(), parameters)
