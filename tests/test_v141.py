"""Pulseq 1.4.1 output parity and rejection of unrepresentable extensions."""

import numpy as np
import pytest

from pypulseqpp import _ext

#: Rotations and shims: refused rather than dropped. See the module docstring.
CANNOT_BE_EXPRESSED = {"rotated_radial", "shimmed_excitation"}


def rotating():
    """A sequence with a rotation on one block."""
    sequence = _ext.Sequence()
    gradient = sequence.register_trap(np.array([2000.0, 1e-4, 2e-3, 1e-4, 0.0]))
    rotation = sequence.register_rotation(np.array([1.0, 0.0, 0.0, 0.0]))
    chain = sequence.chain_extension(
        sequence.extension_type_id("ROTATIONS"), rotation, 0
    )
    sequence.add_block(0, gradient, 0, 0, 0, chain, 2.2e-3)
    return sequence


def shimming():
    """A sequence with an RF shim on one block."""
    sequence = _ext.Sequence()
    magnitude = sequence.register_shape(8, np.linspace(0, 1, 8))
    phase = sequence.register_shape(8, np.zeros(8))
    row = np.zeros(10)
    row[0], row[1], row[2] = 500.0, magnitude, phase
    pulse = sequence.register_rf(row, "e")
    shim = sequence.register_rf_shim(np.array([1.0, 0.0, 0.9, 0.25]))
    chain = sequence.chain_extension(sequence.extension_type_id("RF_SHIMS"), shim, 0)
    sequence.add_block(pulse, 0, 0, 0, 0, chain, 1e-3)
    return sequence


def waiting_on_a_soft_delay():
    sequence = _ext.Sequence()
    delay = sequence.register_soft_delay(
        _ext.SoftDelay(num=1, offset=-2e-4, factor=1.0, hint="TE")
    )
    chain = sequence.chain_extension(sequence.extension_type_id("DELAYS"), delay, 0)
    sequence.add_block(0, 0, 0, 0, 0, chain, 1e-3)
    return sequence


def test_it_writes_what_the_reference_writes(reference_name, build_reference, tmp_path):
    """The bytes, for every sequence 1.4.1 can express."""
    pytest.importorskip("pypulseq_matlab_like")
    import convert
    import test_parity

    if reference_name in CANNOT_BE_EXPRESSED:
        pytest.skip("1.4.1 cannot express this; see the tests below")
    if reference_name in test_parity.NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip("the reference leaves a gap in the block numbering")

    path = tmp_path / f"{reference_name}.seq"
    build_reference().write(str(path), v141_compat=True)

    ours = _ext.write_text_v141(convert.to_core(build_reference()), True)

    assert ours == path.read_bytes()


def test_a_rotating_sequence_is_refused_rather_than_flattened():
    """A rotation turns gradients; a file without it is a different scan."""
    with pytest.raises(RuntimeError, match="rotates or shims"):
        _ext.write_text_v141(rotating(), True)


def test_a_shimming_sequence_is_refused_rather_than_flattened():
    with pytest.raises(RuntimeError, match="rotates or shims"):
        _ext.write_text_v141(shimming(), True)


def test_a_soft_delay_is_left_out_with_a_warning():
    """The rest of the file is what the sequence says, so it is written."""
    with pytest.warns(UserWarning, match="soft delays"):
        written = _ext.write_text_v141(waiting_on_a_soft_delay(), True)

    assert b"DELAYS" not in written


def test_the_file_says_1_4_1():
    sequence = _ext.Sequence()
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)

    written = _ext.write_text_v141(sequence, True).decode()

    assert "[VERSION]\nmajor 1\nminor 4\nrevision 1\n" in written


def test_a_ppm_offset_becomes_the_hertz_it_stands_for():
    """1.4 has no ppm column, and an offset in ppm is one in hertz at a field."""
    sequence = _ext.Sequence()
    magnitude = sequence.register_shape(8, np.linspace(0, 1, 8))
    phase = sequence.register_shape(8, np.zeros(8))
    row = np.zeros(10)
    row[0], row[1], row[2] = 500.0, magnitude, phase
    row[6], row[8] = 3.5, 100.0  # 3.5 ppm on top of 100 Hz
    sequence.register_rf(row, "e")
    sequence.add_block(1, 0, 0, 0, 0, 0, 1e-3)

    gamma, field = 42576000.0, 1.5
    written = _ext.write_text_v141(sequence, True, gamma, field).decode()

    expected = 100.0 + 3.5 * 1e-6 * gamma * field
    rf_row = written.split("[RF]\n")[1].splitlines()[0]
    assert float(rf_row.split()[6]) == pytest.approx(expected, rel=1e-6)


def test_what_it_writes_reads_back_as_the_same_sequence():
    """A 1.4.1 file read here is converted, so the trip is a real one."""
    sequence = _ext.Sequence()
    gradient = sequence.register_trap(np.array([2000.0, 1e-4, 2e-3, 1e-4, 0.0]))
    adc = np.zeros(8)
    adc[0], adc[1] = 64.0, 1e-5
    window = sequence.register_adc(adc)
    for _ in range(4):
        sequence.add_block(0, gradient, 0, 0, window, 0, 2.2e-3)

    loaded = _ext.read(_ext.write_text_v141(sequence, True))

    assert loaded.num_blocks() == sequence.num_blocks()
    assert loaded.num_gradients() == sequence.num_gradients()
    assert loaded.num_adc() == sequence.num_adc()
    assert loaded.duration() == pytest.approx(sequence.duration())
