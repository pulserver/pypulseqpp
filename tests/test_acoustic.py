"""Acoustic check: the loudest repetition, filtered as a periodic waveform."""

import math

import h5py
import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import safety
from pypulseqpp.safety import AcousticResponse, a_weighting

P0 = 20e-6


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
    )


def flat(gain, axes=(0, 1, 2), bins=4097, step=5.0):
    """A response of ``gain`` Pa per mT/m on ``axes``, up to ``(bins - 1) * step`` Hz."""
    transfer = np.zeros((3, bins), complex)
    transfer[list(axes)] = gain
    return AcousticResponse(transfer, step)


def trapezoids(system, amplitudes_mt_per_m, channel="x"):
    seq = pp.Sequence(system)
    for amplitude in amplitudes_mt_per_m:
        seq.add_block(
            pp.make_trapezoid(
                channel,
                amplitude=amplitude * 1e-3 * system.gamma,
                flat_time=2e-3,
                system=system,
            )
        )
    return seq


def ramps(system, amplitudes_mt_per_m):
    """One triangle per block, scaled per block: a repetition of one block."""
    shape = np.concatenate([np.linspace(0, 1, 101), np.linspace(1, 0, 101)[1:]])
    seq = pp.Sequence(system)
    for amplitude in amplitudes_mt_per_m:
        wave = amplitude * 1e-3 * system.gamma * shape
        seq.add_block(
            pp.make_arbitrary_grad("x", wave, first=0.0, last=0.0, system=system)
        )
    return seq


def tone(system, amplitude_mt_per_m, frequency, cycles, channel="x"):
    """Whole cycles of a sinusoidal gradient, so that the block repeats seamlessly."""
    dt = system.grad_raster_time
    n = round(cycles / frequency / dt)
    hz = amplitude_mt_per_m * 1e-3 * system.gamma
    wave = hz * np.sin(2 * math.pi * frequency * (np.arange(n) + 0.5) * dt)
    seq = pp.Sequence(system)
    for _ in range(3):
        seq.add_block(
            pp.make_arbitrary_grad(channel, wave, first=0.0, last=0.0, system=system)
        )
    return seq


def db(pressure):
    return 20 * math.log10(pressure / P0)


def test_a_flat_response_reads_the_peak_gradient_as_the_peak_pressure(system):
    seq = trapezoids(system, [20.0, 20.0])
    _, report = safety.check_spl(seq, flat(0.5))
    assert report.peak == pytest.approx(db(0.5 * 20.0), abs=0.05)


def test_a_tone_reads_its_amplitude_rms_and_a_weighted_rms(system):
    frequency = 1250.0
    seq = tone(system, 10.0, frequency, cycles=25)
    _, report = safety.check_spl(seq, flat(2.0))
    assert report.peak == pytest.approx(db(20.0), abs=0.05)
    assert report.average_unweighted == pytest.approx(db(20.0 / math.sqrt(2)), abs=0.05)
    weighted = 20.0 * float(a_weighting(frequency)) / math.sqrt(2)
    assert report.average == pytest.approx(db(weighted), abs=0.05)


def test_the_repetition_of_most_gradient_energy_is_evaluated(system):
    seq = ramps(system, [5.0, 15.0, 10.0])
    _, report = safety.check_spl(seq, flat(1.0))
    assert seq.repetition()[0] == 1
    assert report.repetition == (2, 2)


def test_of_equally_loud_repetitions_the_earliest_is_evaluated(system):
    seq = ramps(system, [15.0, 5.0, 15.0])
    _, report = safety.check_spl(seq, flat(1.0))
    assert seq.repetition()[0] == 1
    assert report.repetition == (1, 1)


def test_the_prescription_rotation_carries_the_gradient_onto_the_physical_axes(
    system,
):
    seq = trapezoids(system, [20.0], channel="y")
    x_only = flat(1.0, axes=(0,))
    y_to_x = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _, logical = safety.check_spl(seq, x_only)
    _, physical = safety.check_spl(seq, x_only, rotation=y_to_x)
    assert logical.peak == -math.inf
    assert physical.peak == pytest.approx(db(20.0), abs=0.05)


def test_a_level_above_its_limit_fails_the_check(system):
    seq = trapezoids(system, [20.0])
    quiet = flat(0.01)
    is_ok, report = safety.check_spl(seq, quiet)
    assert is_ok
    is_ok, _ = safety.check_spl(seq, quiet, peak_limit=report.peak - 1.0)
    assert not is_ok
    is_ok, _ = safety.check_spl(seq, quiet, average_limit=report.average - 1.0)
    assert not is_ok


def test_a_sequence_without_gradients_is_silent(system):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_delay(1e-3))
    is_ok, report = safety.check_spl(seq, flat(1.0))
    assert is_ok
    assert report.peak == report.average == -math.inf


def test_a_response_that_is_not_three_axes_is_refused(system):
    seq = trapezoids(system, [20.0])
    with pytest.raises(ValueError, match=r"\(3, bins\)"):
        safety.check_spl(seq, AcousticResponse(np.ones((2, 16), complex), 5.0))


def write_response(path, count, interval, gain_per_g_per_cm):
    frequency = np.fft.fftfreq(count, interval)
    weighting = a_weighting(np.abs(frequency)).astype(complex)
    with h5py.File(path, "w") as held:
        for name in (
            "X_AXIS_TRANSFER_FUNCTION",
            "Y_AXIS_TRANSFER_FUNCTION",
            "Z_AXIS_TRANSFER_FUNCTION",
        ):
            held[name] = np.stack([np.full(count, gain_per_g_per_cm), np.zeros(count)])
        held["A_WEIGHTED_FILTER"] = np.stack([weighting.real, weighting.imag])


def test_a_response_file_is_read_per_mt_per_m_on_its_positive_bins(tmp_path):
    path = tmp_path / "response.h5"
    write_response(path, 1024, 40e-6, 3.0)
    response = safety.read_acoustic_response(path, 40e-6)
    assert response.transfer.shape == (3, 513)
    assert response.frequency_step == pytest.approx(1.0 / (1024 * 40e-6))
    np.testing.assert_allclose(response.transfer, 0.3)


def test_reading_a_response_at_another_interval_than_its_own_warns(tmp_path):
    path = tmp_path / "response.h5"
    write_response(path, 4096, 30e-6, 1.0)
    with pytest.warns(UserWarning, match="sampling interval"):
        safety.read_acoustic_response(path, 48e-6)


def test_a_response_file_without_an_axis_is_refused(tmp_path):
    path = tmp_path / "response.h5"
    with h5py.File(path, "w") as held:
        held["X_AXIS_TRANSFER_FUNCTION"] = np.zeros((2, 8))
    with pytest.raises(ValueError, match="Y_AXIS"):
        safety.read_acoustic_response(path, 30e-6)
