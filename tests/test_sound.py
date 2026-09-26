"""Sequence.sound and gradient_sound against MATLAB Pulseq's definition."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal.windows import gaussian

import pypulseqpp as pp

EMPTY = np.zeros((2, 0))


def _gradient_echo():
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    rf, gz, gz_reph = pp.make_sinc_pulse(
        np.pi / 6,
        duration=1e-3,
        slice_thickness=5e-3,
        return_gz=True,
        system=system,
    )
    gx = pp.make_trapezoid("x", flat_area=256, flat_time=3.2e-3, system=system)
    adc = pp.make_adc(64, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    for line in range(4):
        gy = pp.make_trapezoid("y", area=(line - 2) * 4.0, duration=1e-3, system=system)
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy, gz_reph)
        seq.add_block(gx, adc)
        seq.add_block(pp.make_delay(2e-3))
    return seq


def _matlab_sound(wave_data, duration, weights=(1.0, 1.0, 1.0), sample_rate=44100):
    """MATLAB Pulseq's ``sound``, with SciPy's Gaussian window as ``gausswin``."""
    dwell_time = 1.0 / sample_rate
    length = int(np.floor(duration / dwell_time)) + 1
    t = np.arange(length) * dwell_time
    data = np.zeros((2, length))
    for channel in (0, 1):
        if wave_data[channel].size:
            data[channel] = np.interp(
                t, wave_data[channel][0], wave_data[channel][1] * weights[channel], 0, 0
            )
    if wave_data[2].size:
        z = np.interp(t, wave_data[2][0], 0.5 * wave_data[2][1] * weights[2], 0, 0)
        data += z
    # gausswin(L) is exp(-(alpha n / ((L - 1) / 2))^2 / 2) with alpha = 2.5,
    # and MATLAB's round takes 6.5 to 7.
    half = int(np.floor(sample_rate / 6000 + 0.5))
    window = gaussian(2 * half + 1, std=max(half, 1) / 2.5)
    window /= window.sum()
    data = np.stack([np.convolve(row, window, "same") for row in data])
    return 0.95 * data / np.abs(data).max()


@pytest.mark.parametrize("sample_rate", [44100, 39000, 8000, 2000])
def test_sound_follows_matlab_pulseq_definition(sample_rate):
    seq = _gradient_echo()
    expected = _matlab_sound(
        seq.waveforms(), seq.duration()[0], sample_rate=sample_rate
    )
    np.testing.assert_allclose(
        seq.sound(sample_rate=sample_rate), expected, rtol=0, atol=1e-12
    )


def test_sound_plays_the_gradients_after_each_block_rotation():
    gx = pp.make_trapezoid("x", flat_area=256, flat_time=3.2e-3)
    straight, rotated = pp.Sequence(pp.Opts()), pp.Sequence(pp.Opts())
    for _ in range(3):
        straight.add_block(gx)
        straight.add_block(pp.make_delay(1e-3))
        # A quarter turn about z carries x onto y.
        rotated.add_block(gx, pp.make_rotation(np.pi / 2))
        rotated.add_block(pp.make_delay(1e-3))
    heard = straight.sound()
    turned = rotated.sound()
    np.testing.assert_allclose(turned[::-1], heard, rtol=0, atol=1e-12)
    assert not turned[0].any()


def test_channel_weights_route_x_and_y_apart_and_split_z():
    ramp = np.array([[0.0, 1e-3, 2e-3], [0.0, 1e5, 0.0]])
    x = pp.gradient_sound([ramp, EMPTY, EMPTY], 100)
    y = pp.gradient_sound([EMPTY, ramp, EMPTY], 100)
    z = pp.gradient_sound([EMPTY, EMPTY, ramp], 100)
    assert not y[0].any() and not x[1].any()
    np.testing.assert_array_equal(x[0], y[1])
    np.testing.assert_array_equal(z[0], z[1])
    np.testing.assert_allclose(z[0], x[0], rtol=0, atol=1e-12)
    halved = pp.gradient_sound(
        [ramp, EMPTY, ramp], 100, channel_weights=(1.0, 1.0, 0.0), peak=1e5
    )
    unweighted = pp.gradient_sound([ramp, EMPTY, EMPTY], 100, peak=1e5)
    np.testing.assert_array_equal(halved, unweighted)


def test_consecutive_calls_with_one_peak_join_into_the_whole():
    seq = _gradient_echo()
    waves = seq.waveforms()
    total = int(np.floor(seq.duration()[0] * 44100)) + 1
    peak = 1.5 * seq.system.max_grad
    whole = pp.gradient_sound(waves, total, peak=peak)
    edges = [0, 1, 37, 400, 401, total]
    parts = [
        pp.gradient_sound(waves, stop - start, first_sample=start, peak=peak)
        for start, stop in pairwise(edges)
    ]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), whole)


def test_a_given_peak_is_scaled_to_095_without_clipping():
    ramp = np.array([[0.0, 1e-3, 2e-3], [0.0, 1e5, 0.0]])
    # A peak of 0.95 leaves the filtered signal in Hz/m.
    filtered = pp.gradient_sound([ramp, EMPTY, EMPTY], 100, peak=0.95)
    largest = np.abs(filtered).max()
    normalised = pp.gradient_sound([ramp, EMPTY, EMPTY], 100)
    np.testing.assert_allclose(normalised, 0.95 * filtered / largest, rtol=1e-12)
    loud = pp.gradient_sound([ramp, EMPTY, EMPTY], 100, peak=largest / 2)
    assert np.abs(loud).max() == pytest.approx(1.9)


def test_silence_stays_zero():
    audio = pp.gradient_sound([EMPTY, EMPTY, EMPTY], 10)
    assert audio.shape == (2, 10) and not audio.any()
    assert pp.gradient_sound([EMPTY, EMPTY, EMPTY], 0).shape == (2, 0)


def test_a_block_range_sounds_for_as_long_as_its_blocks():
    seq = _gradient_echo()
    durations = np.asarray(list(seq.block_durations.values()))
    audio = seq.sound(block_range=(5, 8))
    assert audio.shape[1] == int(np.floor(durations[4:8].sum() * 44100)) + 1
    whole = _matlab_sound(seq.waveforms(block_range=(5, 8)), durations[4:8].sum())
    np.testing.assert_allclose(audio, whole, rtol=0, atol=1e-12)


def test_sound_writes_the_samples_as_16_bit_wav(tmp_path):
    seq = _gradient_echo()
    path = tmp_path / "gre.wav"
    audio = seq.sound(path=path)
    rate, pcm = wavfile.read(path)
    assert rate == 44100
    assert pcm.dtype == np.int16 and pcm.shape == (audio.shape[1], 2)
    np.testing.assert_array_equal(pcm, np.round(audio.T * 32767).astype(np.int16))


def test_a_wav_file_needs_an_integer_sample_rate(tmp_path):
    with pytest.raises(ValueError, match="integer sample rate"):
        _gradient_echo().sound(sample_rate=22050.5, path=tmp_path / "x.wav")
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        _gradient_echo().sound(sample_rate=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"waveforms": [EMPTY, EMPTY]}, "three gradient axes"),
        ({"waveforms": [np.zeros((3, 2)), EMPTY, EMPTY]}, "time over amplitude"),
        ({"channel_weights": (1.0, 1.0)}, "three weights"),
        ({"num_samples": -1}, "negative"),
        ({"sample_rate": 0.0}, "sample_rate must be positive"),
        ({"peak": 0.0}, "peak must be positive"),
    ],
)
def test_gradient_sound_refuses_malformed_arguments(kwargs, message):
    arguments = {"waveforms": [EMPTY, EMPTY, EMPTY], "num_samples": 10, **kwargs}
    with pytest.raises(ValueError, match=message):
        pp.gradient_sound(**arguments)
