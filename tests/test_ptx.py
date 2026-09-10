"""Dynamic pTx pulses in the channel layout of Roos et al., Magn Reson Med 2025.

Every channel sits in one RF event over a shared time base, so what is held
here is that the layout survives everything a sequence does with a pulse --
the timing check, writing in either form, reading back here and upstream --
and that the report reads it as several pulses played together.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pypulseq
import pytest

import pypulseqpp as pp

SYSTEM = pp.Opts()
SAMPLES = 250


def waveforms(channels, amplitude_hz=250.0):
    """Hard pulses, each channel at its own phase."""
    phases = np.linspace(0.0, np.pi, channels, endpoint=False)
    return amplitude_hz * np.exp(1j * phases)[:, None] * np.ones((1, SAMPLES))


def sequence_playing(rf):
    seq = pp.Sequence(SYSTEM)
    seq.add_block(rf)
    return seq


def test_the_channels_come_back_as_they_went_in():
    """To a rounding: the compiled event holds magnitude and phase."""
    signal = waveforms(4)
    split = pp.split_ptx_pulse(pp.make_ptx_pulse(signal))
    assert np.allclose(split, signal, rtol=1e-12, atol=0.0)


def test_the_time_base_restarts_once_per_channel():
    rf = pp.make_ptx_pulse(waveforms(3), system=SYSTEM)
    one = (np.arange(SAMPLES) + 0.5) * SYSTEM.rf_raster_time
    assert np.array_equal(np.asarray(rf.t), np.tile(one, 3))
    assert rf.shape_dur == pytest.approx(SAMPLES * SYSTEM.rf_raster_time)


def test_a_single_channel_is_an_ordinary_arbitrary_pulse():
    signal = waveforms(1)
    ptx = pp.make_ptx_pulse(signal, system=SYSTEM)
    plain = pp.make_arbitrary_rf(signal[0], 0.0, no_signal_scaling=True, system=SYSTEM)
    assert np.array_equal(np.asarray(ptx.t), np.asarray(plain.t))
    assert np.array_equal(np.asarray(ptx.signal), np.asarray(plain.signal))


def test_a_pulse_lasts_one_channel_however_many_it_drives():
    single = pp.calc_duration(pp.make_ptx_pulse(waveforms(1), system=SYSTEM))
    many = pp.calc_duration(pp.make_ptx_pulse(waveforms(8), system=SYSTEM))
    assert many == pytest.approx(single)


def test_the_timing_check_judges_each_channel_as_a_pulse():
    ok, report = sequence_playing(
        pp.make_ptx_pulse(waveforms(8), system=SYSTEM)
    ).check_timing()
    assert ok, report


@pytest.mark.parametrize("form", ["text", "binary"])
def test_the_layout_survives_writing_and_reading(tmp_path, form):
    signal = waveforms(4)
    seq = sequence_playing(pp.make_ptx_pulse(signal, system=SYSTEM))
    path = tmp_path / f"ptx.{'seq' if form == 'text' else 'bseq'}"
    (seq.write if form == "text" else seq.write_binary)(str(path))
    back = pp.Sequence(SYSTEM)
    back.read(str(path))
    split = pp.split_ptx_pulse(back.get_block(1).rf)
    assert split.shape == signal.shape
    assert np.allclose(split, signal, rtol=1e-6)


def test_upstream_reads_the_same_channels(tmp_path):
    signal = waveforms(4)
    path = tmp_path / "ptx.seq"
    sequence_playing(pp.make_ptx_pulse(signal, system=SYSTEM)).write(str(path))
    upstream = pypulseq.Sequence()
    upstream.read(str(path))
    assert np.allclose(pp.split_ptx_pulse(upstream.get_block(1).rf), signal, rtol=1e-6)


def flips_reported(rf):
    report = sequence_playing(rf).test_report()
    line = next(row for row in report.splitlines() if row.startswith("Flip angle"))
    values = []
    for token in line.split(":")[1].split():
        try:
            values.append(float(token))
        except ValueError:  # the unit printed after the angles
            continue
    return values


def test_the_report_sums_channels_played_in_phase():
    """The flip where every channel has unit, in-phase sensitivity."""
    amplitude = 250.0
    in_phase = amplitude * np.ones((2, SAMPLES), dtype=complex)
    # Each channel integrated on its own time base, forward differences.
    one_channel = amplitude * (SAMPLES - 1) * SYSTEM.rf_raster_time * 360.0
    assert flips_reported(pp.make_ptx_pulse(in_phase, system=SYSTEM)) == pytest.approx(
        [2 * one_channel]
    )


def test_the_report_cancels_channels_played_in_antiphase():
    signal = 250.0 * np.ones((2, SAMPLES), dtype=complex)
    signal[1] *= -1.0
    flips = flips_reported(pp.make_ptx_pulse(signal, system=SYSTEM))
    assert flips == pytest.approx([0.0], abs=1e-9)


def test_a_waveform_that_is_not_two_dimensional_is_refused():
    with pytest.raises(ValueError, match="num_channels, num_samples"):
        pp.make_ptx_pulse(np.ones(10))


def test_a_time_base_that_does_not_repeat_is_refused():
    """Two channels starting together, the second on a time base of its own."""
    one = (np.arange(SAMPLES) + 0.5) * SYSTEM.rf_raster_time
    other = one.copy()
    other[1:] += 0.25 * SYSTEM.rf_raster_time
    rf = SimpleNamespace(
        t=np.concatenate([one, other]), signal=np.ones(2 * SAMPLES, dtype=complex)
    )
    with pytest.raises(ValueError, match="repeated per channel"):
        pp.split_ptx_pulse(rf)
