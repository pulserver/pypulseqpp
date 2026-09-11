"""Paper diagram: one repetition drawn over decimated others."""

import math

import matplotlib
import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext
from pypulseqpp.plot._paper import select_trs

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def encoded(system, repeats, strongest=None):
    """A phase-encoded gradient echo: RF, encode, readout with ADC, spoiler."""
    rf = pp.make_block_pulse(np.pi / 12, duration=0.5e-3, system=system)
    # Weaker than the encode's largest step, so the encode decides which
    # repetition is the strongest.
    readout = pp.make_trapezoid("x", flat_area=1000, flat_time=2e-3, system=system)
    adc = pp.make_adc(128, duration=2e-3, delay=readout.rise_time, system=system)
    spoiler = pp.make_trapezoid("z", area=800, duration=1e-3, system=system)
    # One encode shape, scaled per view, so every repetition has the same
    # definitions and the first full one starts at block 1.
    encode = pp.make_trapezoid("y", area=900, duration=1e-3, system=system)
    seq = pp.Sequence(system)
    for index in range(repeats):
        scale = (-2 + 4 * index / max(repeats - 1, 1)) / 3
        if index == strongest:
            scale = 1.0
        seq.add_block(rf)
        seq.add_block(pp.scale_grad(encode, scale))
        seq.add_block(readout, adc)
        seq.add_block(spoiler)
    return seq


# -- which repetitions ----------------------------------------------------------


def test_the_solid_repetition_is_the_one_with_the_strongest_gradient(system):
    seq = encoded(system, 40, strongest=23)

    size, start, main, _ = select_trs(seq)

    assert (size, start) == (4, 1)
    assert main == 24


def test_a_repetition_can_be_asked_for(system):
    seq = encoded(system, 40)

    _, _, main, underlays = select_trs(seq, tr=7)

    assert main == 7
    assert 7 not in underlays


def test_underlays_are_decimated_to_the_cap_and_keep_the_extremes(system):
    seq = encoded(system, 1000)

    _, _, main, underlays = select_trs(seq, tr=500, max_underlays=16)

    stride = math.ceil(1000 / 16)
    evenly = [1 + k * stride for k in range(16) if 1 + k * stride != main]
    assert set(evenly) <= set(underlays)
    # The phase encode runs from its most negative in the first repetition to
    # its most positive in the last.
    assert {1, 1000} <= set(underlays)
    assert len(underlays) <= 16 + 6


def test_no_underlays_when_none_are_asked_for(system):
    _, _, _, underlays = select_trs(encoded(system, 40), max_underlays=0)

    assert underlays == []


def test_a_repetition_outside_the_sequence_is_refused(system):
    with pytest.raises(ValueError, match="repetitions"):
        select_trs(encoded(system, 10), tr=11)


def test_the_block_extremes_are_the_played_extremes(system):
    seq = encoded(system, 3)
    turned = pp.make_trapezoid("x", area=2000, duration=1.5e-3, system=system)
    seq.add_block(turned, pp.make_rotation(0.4, 0.3))
    n = 200
    wave = 5e4 * np.sin(2 * np.pi * (np.arange(n) + 0.5) / n)
    seq.add_block(pp.make_arbitrary_grad("y", wave, first=0.0, last=0.0, system=system))

    found = np.asarray(_ext.block_extremes(seq._native))

    for block in range(1, seq.num_blocks + 1):
        for axis, channel in enumerate(seq.waveforms(block_range=(block, block))):
            values = np.concatenate(([0.0], channel[1]))
            assert found[block - 1, axis] == pytest.approx(
                [values.min(), values.max()], abs=1e-6
            )


# -- the drawing ------------------------------------------------------------------


def test_the_diagram_has_one_row_per_channel(system):
    drawn = encoded(system, 40).paper_plot()

    ax = drawn.diagram.plot
    ax.figure.canvas.draw()
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["ADC", "Gx", "Gy", "Gz", "RF"]


def test_the_repetitions_underneath_are_the_ones_chosen(system):
    seq = encoded(system, 40)

    drawn = seq.paper_plot(max_underlays=8)

    assert (drawn.tr, drawn.underlays) == select_trs(seq, max_underlays=8)[2:]
    collections = drawn.diagram.plot.collections
    assert collections
    # One collection per gradient row that varies; the encode varies on y.
    assert all(len(c.get_segments()) <= len(drawn.underlays) * 4 for c in collections)


def test_a_long_scan_draws_no_more_than_a_short_one(system):
    short = encoded(system, 100).paper_plot(max_underlays=8)
    long = encoded(system, 5000).paper_plot(max_underlays=8)

    assert len(long.underlays) <= len(short.underlays) + 6
    for few, many in zip(
        short.diagram.plot.collections, long.diagram.plot.collections, strict=True
    ):
        assert len(many.get_segments()) <= len(few.get_segments()) + 6 * 4


def test_a_time_range_is_drawn_alone(system):
    drawn = encoded(system, 40).paper_plot(time_range=(0, 0.02))

    assert drawn.tr is None
    assert drawn.underlays == []
    assert not drawn.diagram.plot.collections


def test_a_sequence_that_does_not_repeat_is_drawn_whole(system):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3, system=system))
    seq.add_block(pp.make_trapezoid("y", area=2000, duration=2e-3, system=system))

    drawn = seq.paper_plot()

    assert drawn.tr is None
    assert drawn.underlays == []


def test_rf_is_drawn_as_its_magnitude_real_or_imaginary_part(system):
    seq = encoded(system, 4)

    for part in ("abs", "real", "imag"):
        seq.paper_plot(rf_plot=part)
    with pytest.raises(ValueError, match="rf_plot"):
        seq.paper_plot(rf_plot="phase")


def test_the_diagram_can_be_drawn_into_given_axes(system):
    _, ax = plt.subplots()

    drawn = encoded(system, 10).paper_plot(ax=ax)

    assert drawn.diagram.plot is ax
