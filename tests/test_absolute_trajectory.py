"""Where a readout samples k, which is what a reconstructor is handed.

A shift is a phase and the phase is ``dr . k``, so a reconstructor given the
trajectory can form the phase itself -- and form it again for a different
prescription, a pose update from motion correction among them, without the
sequence being touched. That is why the trajectory rather than the phase is
the thing handed over, and it is the same array the metadata an acquisition
is enriched with wants anyway.

It is absolute: an interleave that never passes through the centre still
carries coordinates the rest of the acquisition agrees with.

The test of record is the trajectory this package already reports at its ADC
samples, which `calculate_kspace` arrives at a completely different way --
a dense time base over the whole scan, rather than a walk from a block's own
origin.
"""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that builds the reference zoo; see reference.py",
)


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
    )


def sampled(sequence, turn_them=False):
    """Every readout's k, block by block, gathered in play order."""
    origins = _ext.block_k_origins(sequence._native)["origins"]
    turned = np.asarray(sequence._native.block_rotations())
    pieces = []
    for block in range(1, len(sequence) + 1):
        found = _ext.absolute_trajectory(
            sequence._native, block, tuple(origins[block - 1])
        )
        if not found.shape[1]:
            continue
        if turn_them and turned[block - 1] > 0:
            Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
            q = np.asarray(sequence.get_block(block).rotation.quaternion, dtype=float)
            found = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() @ found
        pieces.append(found)
    return np.concatenate(pieces, axis=1) if pieces else np.zeros((3, 0))


def from_the_zoo(name):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import convert
    import reference

    theirs = reference.ZOO[name]()
    sequence = pp.Sequence(theirs.system)
    sequence._native = convert.to_core(theirs)
    return sequence


@pytest.mark.parametrize(
    "name",
    ["gradients_all_axes", "spin_echo", "gre_with_label_inc", "extended_trapezoids"],
)
def test_it_samples_where_the_trajectory_says(name):
    sequence = from_the_zoo(name)

    np.testing.assert_allclose(
        sampled(sequence),
        np.asarray(sequence._kspace()["k_traj_adc"]),
        atol=1e-9,
    )


def test_a_turned_block_answers_in_the_logical_frame():
    """The rotation is the caller's to apply, and it is left to them.

    A consumer that turns the trajectory almost always wants to turn the
    shift with it, and turning both changes nothing -- so turning here would
    be work undone downstream.
    """
    sequence = from_the_zoo("rotated_radial")

    logical = sampled(sequence)
    physical = sampled(sequence, turn_them=True)
    reported = np.asarray(sequence._kspace()["k_traj_adc"])

    # What `calculate_kspace` reports is what the amplifiers play, so it is
    # the turned one that matches, and by a wide margin.
    np.testing.assert_allclose(physical, reported, atol=1e-9)
    assert np.abs(logical - reported).max() > 1.0


def test_a_block_that_acquires_nothing_has_no_trajectory(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3, system=system))

    assert _ext.absolute_trajectory(sequence._native, 1).shape[1] == 0


def test_an_axis_the_block_leaves_alone_holds_where_it_was(system):
    """A silent axis is not zero; it is wherever the scan has already put it."""
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", flat_area=1000, flat_time=1e-3, system=system)
    sequence.add_block(
        gradient, pp.make_adc(8, duration=1e-3, delay=gradient.rise_time, system=system)
    )

    found = _ext.absolute_trajectory(sequence._native, 1, origin=(0.0, 37.0, -4.0))

    np.testing.assert_allclose(found[1], 37.0)
    np.testing.assert_allclose(found[2], -4.0)
    assert found[0][-1] > found[0][0]


def test_a_flat_gradient_still_sweeps(system):
    """Constant is not still: it moves k at a steady rate."""
    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", flat_area=1000, flat_time=1e-3, system=system)
    sequence.add_block(
        gradient,
        pp.make_adc(8, duration=1e-3, delay=gradient.rise_time, system=system),
    )

    found = _ext.absolute_trajectory(sequence._native, 1)

    steps = np.diff(found[0])
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)
    assert steps[0] > 0
