"""Gradient-axis scaling, shared-row rejection and deduplication."""

import math

import numpy as np
import pypulseq as upstream
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext


@pytest.fixture
def three_axes():
    """One trapezoid on each axis, all different."""

    def make():
        sequence = pp.Sequence(pp.Opts())
        sequence.add_block(
            pp.make_trapezoid("x", amplitude=1000, flat_time=5e-3),
            pp.make_trapezoid("y", amplitude=2000, flat_time=3e-3),
            pp.make_trapezoid("z", amplitude=1500, flat_time=4e-3),
        )
        return sequence

    return make


def test_scaling_an_axis_scales_its_amplitude_and_its_area(three_axes):
    sequence = three_axes()
    before = sequence.get_block(1).gx

    sequence.mod_grad_axis("x", 2.0)

    after = sequence.get_block(1).gx
    assert after.amplitude == pytest.approx(2 * before.amplitude)
    assert after.area == pytest.approx(2 * before.area)


def test_scaling_an_axis_leaves_its_timing_alone(three_axes):
    sequence = three_axes()
    before = sequence.get_block(1).gx

    sequence.mod_grad_axis("x", 2.0)

    after = sequence.get_block(1).gx
    assert after.rise_time == before.rise_time
    assert after.flat_time == before.flat_time
    assert after.fall_time == before.fall_time
    assert after.delay == before.delay


def test_scaling_an_axis_leaves_the_other_axes_alone(three_axes):
    sequence = three_axes()
    before = sequence.get_block(1)
    y, z = before.gy.amplitude, before.gz.amplitude

    sequence.mod_grad_axis("x", 2.0)

    after = sequence.get_block(1)
    assert after.gy.amplitude == pytest.approx(y)
    assert after.gz.amplitude == pytest.approx(z)


def test_an_arbitrary_gradients_whole_waveform_scales():
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(
        pp.make_extended_trapezoid(
            "x",
            amplitudes=np.array([0, 1000, 1000, 0]),
            times=np.array([0, 1e-3, 4e-3, 5e-3]),
        )
    )
    before = sequence.get_block(1).gx
    waveform, area, first, last = (
        before.waveform.copy(),
        before.area,
        before.first,
        before.last,
    )

    sequence.mod_grad_axis("x", 2.0)

    after = sequence.get_block(1).gx
    assert after.waveform == pytest.approx(2 * waveform)
    assert after.area == pytest.approx(2 * area)
    assert after.first == pytest.approx(2 * first)
    assert after.last == pytest.approx(2 * last)


def test_scaling_an_axis_by_zero_silences_it(three_axes):
    sequence = three_axes()

    sequence.mod_grad_axis("y", 0)

    silenced = sequence.get_block(1).gy
    assert silenced.amplitude == 0.0
    assert silenced.area == 0.0
    # Silenced, not removed: the block still lasts as long.
    assert silenced.flat_time == 3e-3


def test_flipping_an_axis_inverts_it(three_axes):
    sequence = three_axes()
    before = sequence.get_block(1).gz.amplitude

    sequence.flip_grad_axis("z")

    assert sequence.get_block(1).gz.amplitude == pytest.approx(-before)


@pytest.mark.parametrize("factor", [2.5, -1.7, 0.3, 10.0])
def test_scaling_and_unscaling_comes_back_to_where_it_started(factor):
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(
        pp.make_trapezoid("x", amplitude=1234.5, flat_time=5e-3),
        pp.make_extended_trapezoid(
            "y",
            amplitudes=np.array([0, 987.6, 500, 0]),
            times=np.array([0, 1e-3, 3e-3, 4e-3]),
        ),
        pp.make_trapezoid("z", amplitude=-567.8, flat_time=3e-3),
    )
    before = sequence.get_block(1)
    amplitudes = (before.gx.amplitude, before.gz.amplitude)
    areas = (before.gx.area, before.gy.area, before.gz.area)
    waveform = before.gy.waveform.copy()

    for axis in "xyz":
        sequence.mod_grad_axis(axis, factor)
    for axis in "xyz":
        sequence.mod_grad_axis(axis, 1.0 / factor)

    after = sequence.get_block(1)
    assert (after.gx.amplitude, after.gz.amplitude) == pytest.approx(amplitudes)
    assert (after.gx.area, after.gy.area, after.gz.area) == pytest.approx(areas)
    assert after.gy.waveform == pytest.approx(waveform)


@pytest.mark.parametrize("axis", ["w", "xy", "X", ""])
def test_an_axis_that_is_not_an_axis_is_refused(axis, three_axes):
    with pytest.raises(ValueError, match="Invalid axis"):
        three_axes().mod_grad_axis(axis, 1.0)


def test_a_gradient_played_on_two_axes_has_no_one_answer():
    """The same row on x and on y cannot be scaled for one of them."""
    core = _ext.Sequence()
    core.set_rasters(1e-6, 10e-6, 100e-9, 10e-6)
    shared = core.register_trap([1000.0, 1e-4, 5e-3, 1e-4, 0.0])
    core.add_block(0, shared, shared, 0, 0, 0, 5.2e-3)

    with pytest.raises(RuntimeError, match="same gradient event used on multiple axes"):
        core.scale_gradient_axis(0, 2.0)


def test_scaling_an_axis_of_an_empty_sequence_does_nothing():
    sequence = pp.Sequence(pp.Opts())

    sequence.mod_grad_axis("x", 2.0)

    assert len(sequence) == 0


def test_scaling_an_axis_nothing_is_played_on_does_nothing():
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(
        pp.make_trapezoid("x", amplitude=1000, flat_time=5e-3),
        pp.make_trapezoid("z", amplitude=1500, flat_time=3e-3),
    )

    sequence.mod_grad_axis("y", 2.0)

    block = sequence.get_block(1)
    assert block.gx.amplitude == pytest.approx(1000)
    assert block.gz.amplitude == pytest.approx(1500)
    assert block.gy is None


def test_scaling_an_axis_adds_no_gradient_row():
    """The row is rewritten, not appended to."""
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(pp.make_trapezoid("x", amplitude=1000, flat_time=5e-3))
    rows = sequence._native.num_gradients()

    sequence.mod_grad_axis("x", -1.0)

    assert sequence._native.num_gradients() == rows


def test_an_inverted_gradient_collapses_onto_the_one_it_equals():
    """Scaling leaves a row deduplication can still recognise."""
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(pp.make_trapezoid("x", amplitude=1000, flat_time=5e-3))
    sequence.mod_grad_axis("x", -1.0)
    sequence.add_block(pp.make_trapezoid("x", amplitude=-1000, flat_time=5e-3))

    collapsed = sequence.remove_duplicates()

    assert collapsed._native.num_gradients() == 1
    assert collapsed.get_raw_block_content_IDs(1).gx == (
        collapsed.get_raw_block_content_IDs(2).gx
    )


def test_scaling_an_axis_matches_upstream(three_axes):
    """Held against the toolbox that defines what this method means."""
    system = upstream.Opts()
    theirs = upstream.Sequence(system=system)
    theirs.add_block(
        upstream.make_trapezoid("x", amplitude=1000, flat_time=5e-3, system=system),
        upstream.make_trapezoid("y", amplitude=2000, flat_time=3e-3, system=system),
        upstream.make_trapezoid("z", amplitude=1500, flat_time=4e-3, system=system),
    )
    ours = three_axes()

    for sequence in (theirs, ours):
        sequence.mod_grad_axis("x", 2.0)
        sequence.mod_grad_axis("y", -1.0)
        sequence.mod_grad_axis("z", 0.5)

    their_block, our_block = theirs.get_block(1), ours.get_block(1)
    for axis in ("gx", "gy", "gz"):
        assert getattr(our_block, axis).amplitude == pytest.approx(
            getattr(their_block, axis).amplitude
        )
        assert getattr(our_block, axis).area == pytest.approx(
            getattr(their_block, axis).area
        )


def test_scaling_a_million_blocks_is_one_pass():
    """A guard on where the work happens, not on how fast it is."""
    sequence = pp.Sequence(pp.Opts())
    gradient = pp.make_trapezoid("x", amplitude=1000, flat_time=1e-3)
    for _ in range(1000):
        sequence.add_block(gradient)

    sequence.mod_grad_axis("x", 2.0)

    assert all(
        sequence.get_block(index).gx.amplitude == pytest.approx(2000)
        for index in (1, 500, 1000)
    )
    assert math.isclose(sequence.get_block(1).gx.flat_time, 1e-3)
