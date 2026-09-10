"""Complex transmit-channel weights through registration and file round trips."""

import numpy as np
import pytest

import pypulseqpp as pp


@pytest.fixture
def system():
    return pp.Opts()


@pytest.fixture
def pulse(system):
    return pp.make_block_pulse(
        np.pi / 2, duration=1e-3, system=system, use="excitation"
    )


WEIGHTS = np.array([1, np.exp(1j * np.pi / 2), 0.8 * np.exp(-1j * 2.5), 0.5])


def test_a_shim_says_what_it_is():
    assert pp.make_rf_shim([1, 0.8, 0.9, 1.1]).type == "rf_shim"


def test_real_weights_are_kept_as_complex_ones():
    """A channel weight is a magnitude and a phase, whichever way it arrives."""
    shim = pp.make_rf_shim([1, 0.8, 0.9, 1.1])

    assert shim.shim_vector.dtype == np.complex128
    np.testing.assert_allclose(shim.shim_vector.real, [1, 0.8, 0.9, 1.1])
    np.testing.assert_allclose(shim.shim_vector.imag, 0.0)


def test_complex_weights_are_kept_as_they_are():
    weights = np.array([1 + 0.5j, 0.8 - 0.3j, 0.9 + 0.1j])

    np.testing.assert_allclose(pp.make_rf_shim(weights).shim_vector, weights)


def test_one_weight_per_channel_comes_back_one_per_channel(system, pulse):
    sequence = pp.Sequence(system)
    sequence.add_block(pulse, pp.make_rf_shim(WEIGHTS))

    stored = sequence.get_block(1).rf_shim.shim_vector

    assert stored.shape == WEIGHTS.shape


def test_a_block_hands_back_the_weights_it_was_given(system, pulse):
    sequence = pp.Sequence(system)
    sequence.add_block(pulse, pp.make_rf_shim(WEIGHTS))

    stored = sequence.get_block(1).rf_shim.shim_vector

    np.testing.assert_allclose(stored, WEIGHTS, atol=1e-15)


def test_the_weights_survive_a_file(system, pulse, tmp_path):
    sequence = pp.Sequence(system)
    sequence.add_block(pulse, pp.make_rf_shim(WEIGHTS))
    path = tmp_path / "shimmed.seq"
    sequence.write(str(path))

    reread = pp.Sequence(system)
    reread.read(str(path))

    # A file stores a weight as a magnitude and a phase, at the precision the
    # format writes them at.
    np.testing.assert_allclose(
        reread.get_block(1).rf_shim.shim_vector, WEIGHTS, atol=1e-5
    )


def test_two_blocks_shimmed_alike_share_one_row(system, pulse):
    """The weights are a library row like any other, and repeat is free."""
    sequence = pp.Sequence(system)
    for _ in range(4):
        sequence.add_block(pulse, pp.make_rf_shim(WEIGHTS))
    collapsed = sequence.remove_duplicates()

    assert collapsed._native.num_rf_shims() == 1
    np.testing.assert_allclose(
        collapsed.get_block(3).rf_shim.shim_vector, WEIGHTS, atol=1e-15
    )
