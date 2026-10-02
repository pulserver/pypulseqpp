"""RF instances: the RF events a sequence plays, and the pulses they play."""

import numpy as np
import pytest

import pypulseqpp as pp

SYSTEM = pp.Opts()
SAMPLES = 100


def sinc(flip=np.pi / 6, duration=2e-3, **kwargs):
    return pp.make_sinc_pulse(
        flip, duration=duration, system=SYSTEM, use="excitation", **kwargs
    )


def block_pulse(flip=np.pi / 4):
    return pp.make_block_pulse(flip, duration=1e-3, system=SYSTEM, use="excitation")


def ptx(channels=3):
    """Hard pulses, each channel at its own phase."""
    phases = np.linspace(0.0, np.pi, channels, endpoint=False)
    signal = 250.0 * np.exp(1j * phases)[:, None] * np.ones((1, SAMPLES))
    return pp.make_ptx_pulse(signal, system=SYSTEM)


def sequence_of(*blocks):
    """One block per argument: an event, or a tuple of the events the block holds."""
    seq = pp.Sequence(SYSTEM)
    for events in blocks:
        seq.add_block(*(events if isinstance(events, tuple) else (events,)))
    return seq


def played(seq, block):
    """What a block's RF event plays on each transmit channel, in Hz."""
    event = seq.get_block(block)
    signal = pp.split_ptx_pulse(event.rf)
    if event.rf_shim is not None:
        signal = signal * event.rf_shim.shim_vector[:, None]
    return signal


def blocks_with_rf(seq):
    return np.flatnonzero(seq.libraries().blocks[:, 0]) + 1


def test_a_sequence_without_rf_has_no_instances():
    found = sequence_of(pp.make_delay(1e-3)).rf_instances()

    assert found.definitions == ()
    assert found.definition.tolist() == []
    assert found.amplitude.tolist() == []


def test_rf_instances_number_definitions_by_first_play():
    """The numbering follows the blocks, not the order the pulses were registered in."""
    first, second, third = sinc(), block_pulse(), sinc(np.pi / 2, duration=1e-3)
    seq = sequence_of(first, second)
    seq.set_block(1, third)
    seq.add_block(second)
    seq.add_block(third)
    assert seq.event_definitions().rf.tolist() == [1, 2, 3, 2, 3]

    found = seq.rf_instances()

    assert found.definition.tolist() == [0, 1, 1, 0]
    flips = seq.rf_flip_angles()
    assert [d.flip_deg for d in found.definitions] == pytest.approx(
        [flips[2], flips[1]]
    )


def test_a_definition_carries_the_timing_use_and_flip_of_its_event():
    pulse = pp.make_sinc_pulse(
        np.pi / 2, duration=2e-3, delay=1e-4, system=SYSTEM, use="refocusing"
    )

    found = sequence_of(pulse).rf_instances().definitions[0]

    assert found.use == "refocusing"
    assert found.delay == pulse.delay == 1e-4
    assert found.center == pulse.center
    np.testing.assert_array_equal(found.time, pulse.t)
    assert found.flip_deg == pytest.approx(90.0, rel=1e-3)


def test_the_first_instance_of_a_definition_has_amplitude_one():
    pulse, other = block_pulse(), sinc()
    seq = pp.Sequence(SYSTEM)
    for scale in (1.0, 2.0, 0.5):
        pulse.amplitude = scale * 250.0
        seq.add_block(pulse)
    for scale in (3.0, 1.0):
        other.amplitude = scale * 80.0
        seq.add_block(other)

    found = seq.rf_instances()

    assert found.definition.tolist() == [0, 0, 0, 1, 1]
    for number in range(len(found.definitions)):
        assert found.amplitude[found.definition == number][0] == 1.0


def test_one_definition_plays_at_two_amplitudes():
    pulse = block_pulse()
    seq = sequence_of(pulse)
    pulse.amplitude = 0.37 * pulse.amplitude
    seq.add_block(pulse)

    found = seq.rf_instances()

    assert len(found.definitions) == 1
    assert found.definition.tolist() == [0, 0]
    assert found.amplitude.tolist() == pytest.approx([1.0, 0.37])
    assert found.definitions[0].peak_hz == pytest.approx(125.0)
    np.testing.assert_allclose(found.definitions[0].waveform, np.ones((1, 2)))


def test_the_base_of_a_definition_is_its_first_instance_with_a_nonzero_amplitude():
    pulse = block_pulse()
    seq = pp.Sequence(SYSTEM)
    for amplitude in (0.0, 100.0, 50.0, 0.0, -25.0):
        pulse.amplitude = amplitude
        seq.add_block(pulse)

    found = seq.rf_instances()

    assert found.amplitude.tolist() == [0.0, 1.0, 0.5, 0.0, -0.25]
    assert found.definitions[0].peak_hz == 100.0
    assert found.definitions[0].flip_deg == pytest.approx(36.0)


def test_a_definition_played_only_at_zero_amplitude_has_unit_amplitudes_and_no_peak():
    pulse = block_pulse()
    pulse.amplitude = 0.0

    found = sequence_of(pulse, pulse).rf_instances()

    assert found.amplitude.tolist() == [1.0, 1.0]
    assert found.definitions[0].peak_hz == 0.0
    assert found.definitions[0].flip_deg == 0.0
    assert np.abs(found.definitions[0].waveform).max() == 1.0


def test_rf_instance_flips_match_rf_flip_angles():
    """An instance flips by its definition's flip, scaled by its amplitude."""
    pulse, other = sinc(), ptx()
    seq = pp.Sequence(SYSTEM)
    for scale in (1.0, 0.5, 2.0):
        pulse.amplitude = scale * 184.0
        seq.add_block(pulse)
    seq.add_block(other)
    seq.add_block(other, pp.make_rf_shim([1.0, 0.5j, -1.0]))
    other.amplitude = 0.25 * other.amplitude
    seq.add_block(other)

    found = seq.rf_instances()

    ids = seq.libraries().blocks[blocks_with_rf(seq) - 1, 0]
    flips = np.array([found.definitions[d].flip_deg for d in found.definition])
    np.testing.assert_allclose(
        flips * found.amplitude, seq.rf_flip_angles()[ids - 1], rtol=1e-12
    )


def test_an_instance_is_its_amplitude_times_the_normalised_definition():
    pulse = sinc(freq_offset=500.0, phase_offset=0.7)
    seq = pp.Sequence(SYSTEM)
    for scale in (1.0, 0.4):
        pulse.amplitude = scale * 184.0
        seq.add_block(pulse)
    seq.add_block(ptx())
    seq.add_block(pulse, pp.make_rf_shim([0.5, 1j]))

    found = seq.rf_instances()

    assert len(found.definitions) == 3
    for row, block in enumerate(blocks_with_rf(seq)):
        definition = found.definitions[found.definition[row]]
        np.testing.assert_allclose(
            found.amplitude[row] * definition.peak_hz * definition.waveform,
            played(seq, block),
            rtol=1e-12,
            atol=1e-9,
        )
        assert np.abs(definition.waveform).max() == pytest.approx(1.0)


def test_a_definition_keeps_its_phase():
    """The phase is the pulse's, a sweep included, and not the event's offsets."""
    samples = np.arange(200)
    times = (samples + 0.5) * SYSTEM.rf_raster_time
    envelope = np.exp(-(((samples - 100) / 40.0) ** 2))
    signal = 300.0 * envelope * np.exp(1j * (2 * np.pi * 1e8 * times**2 + 0.3))
    pulse = pp.make_arbitrary_rf(
        signal,
        0.0,
        no_signal_scaling=True,
        freq_offset=500.0,
        phase_offset=0.7,
        system=SYSTEM,
    )

    found = sequence_of(pulse).rf_instances().definitions[0]

    assert np.ptp(np.unwrap(np.angle(signal))) > 2 * np.pi
    np.testing.assert_allclose(np.angle(found.waveform[0] / signal), 0.0, atol=1e-9)
    np.testing.assert_allclose(found.time, times)


def test_a_ptx_definition_has_one_row_per_channel():
    pulse = ptx(3)

    found = sequence_of(pulse, pulse).rf_instances()

    assert found.definition.tolist() == [0, 0]
    definition = found.definitions[0]
    assert definition.waveform.shape == (3, SAMPLES)
    assert definition.time.shape == (SAMPLES,)
    np.testing.assert_allclose(
        definition.peak_hz * definition.waveform, pp.split_ptx_pulse(pulse), rtol=1e-12
    )
    assert np.abs(definition.waveform).max() == pytest.approx(1.0)


def test_equal_pulses_registered_separately_are_one_definition():
    seq = sequence_of(sinc(), sinc())
    assert seq.event_definitions().rf.tolist() == [1, 2]

    assert seq.rf_instances().definition.tolist() == [0, 0]


def test_a_shim_multiplies_the_definition_channels():
    pulse = sinc()
    weights = np.array([1.0, 0.5j, -0.25, 0.8 * np.exp(0.3j)])
    seq = sequence_of(
        (pulse, pp.make_rf_shim(weights)),
        (pulse, pp.make_rf_shim(np.ones(4))),
        (pulse, pp.make_rf_shim(weights)),
        pulse,
    )

    found = seq.rf_instances()

    assert found.definition.tolist() == [0, 1, 0, 2]
    shimmed = found.definitions[0]
    assert shimmed.waveform.shape == (4, pulse.t.size)
    np.testing.assert_allclose(
        shimmed.peak_hz * shimmed.waveform,
        weights[:, None] * np.asarray(pulse.signal),
        rtol=1e-12,
        atol=1e-9,
    )
    assert found.definitions[2].waveform.shape == (1, pulse.t.size)


def test_a_shim_multiplies_the_channels_of_a_ptx_pulse():
    pulse = ptx(3)
    weights = np.array([1.0, 0.5j, -0.25])

    found = sequence_of((pulse, pp.make_rf_shim(weights))).rf_instances()

    definition = found.definitions[0]
    np.testing.assert_allclose(
        definition.peak_hz * definition.waveform,
        weights[:, None] * pp.split_ptx_pulse(pulse),
        rtol=1e-12,
    )


@pytest.mark.parametrize("weights", [[1.0, 1.0], [1.0]])
def test_a_shim_of_another_channel_count_is_refused(weights):
    seq = sequence_of((ptx(3), pp.make_rf_shim(weights)))

    with pytest.raises(ValueError, match="3 transmit channels"):
        seq.rf_instances()


def test_dedup_does_not_change_rf_instances():
    """Equal pulses and equal shims are one definition, collapsed or not."""
    pulse, twin, spokes = sinc(), sinc(), ptx()
    shim = np.array([1.0, 0.5j, -0.25])
    seq = sequence_of(
        pulse,
        twin,
        (spokes, pp.make_rf_shim(shim)),
        (spokes, pp.make_rf_shim(shim)),
        (pulse, pp.make_rf_shim(shim)),
        (pulse, pp.make_rf_shim(shim)),
        pp.make_delay(1e-3),
        spokes,
    )
    pulse.amplitude = 0.37 * pulse.amplitude
    seq.add_block(pulse)
    collapsed = seq.remove_duplicates()
    assert len(collapsed.libraries().rf) < len(seq.libraries().rf)
    assert len(collapsed.libraries().rf_shims) < len(seq.libraries().rf_shims)

    on, off = collapsed.rf_instances(), seq.rf_instances()

    assert on.definition.tolist() == off.definition.tolist()
    assert off.definition.tolist() == [0, 0, 1, 1, 2, 2, 3, 0]
    # A collapsed sequence holds what a file records: six significant digits
    # of an amplitude, and float32 samples.
    np.testing.assert_allclose(on.amplitude, off.amplitude, rtol=1e-5)
    for left, right in zip(on.definitions, off.definitions, strict=True):
        np.testing.assert_allclose(left.waveform, right.waveform, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(left.time, right.time, rtol=1e-5)
        assert left.use == right.use
        assert left.peak_hz == pytest.approx(right.peak_hz, rel=1e-5)
        assert left.flip_deg == pytest.approx(right.flip_deg, rel=1e-5)
        assert left.center == pytest.approx(right.center, rel=1e-5)
        assert left.delay == pytest.approx(right.delay, rel=1e-5, abs=1e-12)


def test_rf_instances_leaves_the_sequence_as_it_is():
    seq = sequence_of(sinc(), sinc(), (sinc(), pp.make_rf_shim([1.0, 1.0j])))
    rows = seq.libraries().rf.copy()
    definitions = seq.event_definitions().rf.copy()

    seq.rf_instances()

    np.testing.assert_array_equal(seq.libraries().rf, rows)
    np.testing.assert_array_equal(seq.event_definitions().rf, definitions)
