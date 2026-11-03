"""An arbitrary gradient sampled every half raster, as it is played and analysed.

The scanner plays a gradient on its raster, so it plays every other sample: the
ones at the raster centres.
"""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import safety

INTERVALS = 10
PEAK = 1e4


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def half_raster(system):
    """A half sine over ten raster intervals, its samples every half raster."""
    raster = system.grad_raster_time
    times = np.arange(1, 2 * INTERVALS) * raster / 2
    samples = PEAK * np.sin(np.pi * times / (INTERVALS * raster))
    gradient = pp.make_arbitrary_grad(
        "x", samples, oversampling=True, first=0, last=0, system=system
    )
    return times, samples, gradient


def played_line(system, times, samples):
    """The raster-centre samples between the recorded zero edges."""
    edge = INTERVALS * system.grad_raster_time
    return (
        np.concatenate(([0.0], times[0::2], [edge])),
        np.concatenate(([0.0], samples[0::2], [0.0])),
    )


def test_a_half_raster_gradient_plays_its_raster_centre_samples_between_its_edges(
    system,
):
    times, samples, gradient = half_raster(system)
    seq = pp.Sequence(system)
    seq.add_block(gradient)

    played = seq.waveforms_and_times()[0][0]

    at, values = played_line(system, times, samples)
    np.testing.assert_allclose(played[0], at)
    np.testing.assert_allclose(played[1], values, rtol=1e-6)


def test_a_half_raster_gradient_moves_k_space_by_its_area(system):
    times, samples, gradient = half_raster(system)
    seq = pp.Sequence(system)
    seq.add_block(gradient)
    seq.add_block(pp.make_adc(4, duration=4 * system.grad_raster_time, system=system))

    at, values = played_line(system, times, samples)
    area = np.trapezoid(values, at)
    np.testing.assert_allclose(seq.adc_kspace()[0], area, rtol=1e-6)


def test_the_gradient_checks_see_the_peak_and_slew_of_the_raster_centre_samples(
    system,
):
    times, samples, gradient = half_raster(system)
    seq = pp.Sequence(system)
    seq.add_block(gradient)

    _, amplitude = safety.check_max_grad(seq, system)
    _, slew = safety.check_max_slew(seq, system)

    at, values = played_line(system, times, samples)
    assert amplitude.axes[0].value == pytest.approx(np.abs(values).max(), rel=1e-6)
    steepest = np.max(np.abs(np.diff(values) / np.diff(at)))
    assert slew.axes[0].value == pytest.approx(steepest, rel=1e-6)


def test_a_half_raster_gradient_read_back_from_a_file_plays_as_written(
    system, tmp_path
):
    _, _, gradient = half_raster(system)
    seq = pp.Sequence(system)
    seq.add_block(gradient)
    seq.add_block(pp.make_adc(4, duration=4 * system.grad_raster_time, system=system))
    seq.write(str(tmp_path / "half.seq"))

    back = pp.Sequence()
    back.read(str(tmp_path / "half.seq"))

    np.testing.assert_allclose(back.adc_kspace(), seq.adc_kspace(), rtol=1e-6)
