"""How wide in frequency a pulse is.

Ported from the reference toolbox's `test_calc_rf_bandwidth`. The answers
here are PyPulseq's, except where its own answer is no answer at all: a hard
pulse is stored as the two ends of a rectangle, and the spectrum of a
rectangle held on either side of them is a delta.
"""

import math

import numpy as np
import pypulseq as upstream
import pytest

import pypulseqpp as pp

#: The width of a rectangle's transform at half its height, in units of 1/T.
RECTANGLE_HALF_WIDTH = 1.2067


def test_upstream_answers_a_hard_pulse_with_nothing():
    """The gap this stands in for; the answer below is against it."""
    hard = upstream.make_block_pulse(math.pi / 2, duration=1e-3)

    assert float(np.ravel(upstream.calc_rf_bandwidth(hard))[0]) == 0.0


@pytest.mark.parametrize("duration", [2e-3, 1e-3, 0.5e-3, 0.2e-3])
def test_a_hard_pulse_is_as_wide_as_it_is_short(duration):
    hard = pp.make_block_pulse(math.pi / 2, duration=duration)

    bandwidth = pp.calc_rf_bandwidth(hard)

    assert bandwidth == pytest.approx(RECTANGLE_HALF_WIDTH / duration, rel=0.05)


@pytest.mark.parametrize("time_bw_product", [4, 8])
def test_a_sinc_is_as_wide_as_its_time_bandwidth_product_says(time_bw_product):
    duration = 4e-3
    sinc = pp.make_sinc_pulse(
        math.pi / 2, duration=duration, time_bw_product=time_bw_product
    )

    bandwidth = pp.calc_rf_bandwidth(sinc)

    assert bandwidth == pytest.approx(time_bw_product / duration, rel=0.15)


def test_a_shaped_pulse_is_answered_within_a_bin_of_upstream():
    """The flanks move by less than the grid they are found on."""
    sinc = pp.make_sinc_pulse(math.pi / 2, duration=2e-3, time_bw_product=4)
    theirs = upstream.make_sinc_pulse(math.pi / 2, duration=2e-3, time_bw_product=4)

    assert pp.calc_rf_bandwidth(sinc, dw=10) == pytest.approx(
        float(np.ravel(upstream.calc_rf_bandwidth(theirs, dw=10))[0]), abs=2 * 10
    )


@pytest.mark.parametrize("dw", [0.5, 5, 25, 50])
def test_the_answer_does_not_depend_on_the_resolution_it_was_measured_at(dw):
    """A flank is a smooth function, so where it crosses is not a bin index.

    Rounding to the nearest bin instead makes the answer's precision the bin
    width, and the only way to buy precision is a finer grid -- which is the
    whole cost, since the grid is what is transformed. At a 2 us raster,
    0.5 Hz is a million points.
    """
    slr = pp.make_slr_pulse(math.pi / 15, duration=3e-3, time_bw_product=4)

    assert pp.calc_rf_bandwidth(slr, dw=dw) == pytest.approx(
        pp.calc_rf_bandwidth(slr, dw=0.5), rel=1e-3
    )


def test_a_lower_cutoff_is_measured_further_down_the_flanks():
    hard = pp.make_block_pulse(math.pi / 2, duration=1e-3)

    assert pp.calc_rf_bandwidth(hard, cutoff=0.25) > pp.calc_rf_bandwidth(
        hard, cutoff=0.5
    )


def test_retuning_a_pulse_moves_its_band_without_widening_it():
    sinc = pp.make_sinc_pulse(math.pi / 2, duration=2e-3, time_bw_product=4)
    tuned = pp.make_sinc_pulse(
        math.pi / 2, duration=2e-3, time_bw_product=4, freq_offset=1500.0
    )

    assert pp.calc_rf_bandwidth(tuned) == pytest.approx(
        pp.calc_rf_bandwidth(sinc), rel=0.05
    )


def test_a_retuned_pulse_is_measured_about_where_it_is_tuned():
    tuned = pp.make_sinc_pulse(
        math.pi / 2, duration=2e-3, time_bw_product=4, freq_offset=1500.0
    )

    _, spectrum, axis = pp.calc_rf_bandwidth(
        tuned, return_spectrum=True, return_axis=True
    )

    # The midpoint of the flanks, not the tallest bin: across the flat top of
    # a main lobe it is truncation ripple that decides which bin is highest.
    band = np.flatnonzero(np.abs(spectrum) >= 0.5 * np.abs(spectrum).max())
    assert 0.5 * (axis[band[0]] + axis[band[-1]]) == pytest.approx(1500.0, abs=50.0)


def test_a_sinc_this_far_off_resonance_defeats_upstream():
    """The gap the baseband measurement stands in for."""
    theirs = upstream.make_sinc_pulse(
        math.pi / 2, duration=2e-3, time_bw_product=4, freq_offset=1500.0
    )

    assert float(np.ravel(upstream.calc_rf_bandwidth(theirs))[0]) < 0.5 * 4 / 2e-3


def test_a_bandwidth_is_a_number_rather_than_an_array_holding_one():
    hard = pp.make_block_pulse(math.pi / 2, duration=1e-3)

    assert isinstance(pp.calc_rf_bandwidth(hard), float)


@pytest.mark.parametrize(
    ("kwargs", "count"),
    [
        ({"return_spectrum": True}, 2),
        ({"return_axis": True}, 2),
        ({"return_spectrum": True, "return_axis": True}, 3),
    ],
)
def test_the_spectrum_and_its_axis_come_back_when_asked_for(kwargs, count):
    hard = pp.make_block_pulse(math.pi / 6, duration=0.5e-3)

    answer = pp.calc_rf_bandwidth(hard, **kwargs)

    assert isinstance(answer, tuple) and len(answer) == count
    assert isinstance(answer[0], float)
    assert all(np.asarray(part).ndim == 1 for part in answer[1:])
    if count == 3:
        assert len(answer[1]) == len(answer[2])
