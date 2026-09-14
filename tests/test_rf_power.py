"""RF power of a pulse and of a sequence, as MATLAB Pulseq's calcRfPower."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import pypulseqpp as pp


@pytest.fixture
def system():
    return pp.Opts(rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)


# -- one pulse (ported from pypulseq-matlab-like's test_calc_rf_power) -----------


def test_a_block_pulse_carries_its_amplitude_squared_over_its_duration():
    duration = 1e-3
    rf = pp.make_block_pulse(math.pi / 2, duration=duration)
    amplitude = np.max(np.abs(rf.signal))

    total_energy, peak_power, rf_rms = pp.calc_rf_power(rf)

    assert total_energy == pytest.approx(amplitude**2 * duration, rel=0.02)
    assert peak_power == pytest.approx(amplitude**2, rel=0.02)
    assert rf_rms == pytest.approx(amplitude, rel=0.02)


def test_the_rms_is_the_energy_over_the_shape_duration():
    rf = pp.make_sinc_pulse(math.pi / 2, duration=4e-3, time_bw_product=4)

    total_energy, _, rf_rms = pp.calc_rf_power(rf)

    assert rf_rms == pytest.approx(math.sqrt(total_energy / rf.shape_dur), rel=1e-6)


def test_the_energy_does_not_depend_on_the_resampling_step():
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3)

    coarse, _, _ = pp.calc_rf_power(rf, 1e-6)
    fine, _, _ = pp.calc_rf_power(rf, 0.5e-6)

    assert fine == pytest.approx(coarse, rel=0.02)


def test_the_energy_is_a_positive_scalar():
    total_energy = pp.calc_rf_power(pp.make_gauss_pulse(math.pi / 2, duration=4e-3))[0]

    assert np.isscalar(total_energy)
    assert total_energy > 0


@pytest.mark.parametrize("dt", [1e-6, 0.3e-6])
def test_a_pulse_reads_as_the_reference_toolbox_reads_it(dt):
    reference = pytest.importorskip("pypulseq_matlab_like")
    rf = pp.make_sinc_pulse(math.pi / 3, duration=2e-3, time_bw_product=4)
    same = SimpleNamespace(
        t=np.asarray(rf.t), signal=np.asarray(rf.signal), shape_dur=rf.shape_dur
    )

    assert pp.calc_rf_power(rf, dt) == pytest.approx(
        reference.calc_rf_power(same, dt), rel=1e-12
    )


def test_ptx_channels_add_their_powers_and_never_cancel():
    one = np.ones(100) * 200.0
    rf = pp.make_ptx_pulse(np.vstack([one, -one]))
    single = pp.make_ptx_pulse(one[None, :])

    energy, peak, _ = pp.calc_rf_power(rf)
    alone, alone_peak, _ = pp.calc_rf_power(single)

    assert energy == pytest.approx(2 * alone, rel=1e-12)
    assert peak == pytest.approx(2 * alone_peak, rel=1e-12)


# -- a sequence -------------------------------------------------------------------


def excited(system, flips=(30, 60, 90, 45), tr=10e-3):
    seq = pp.Sequence(system)
    for flip in flips:
        rf = pp.make_sinc_pulse(math.radians(flip), duration=2e-3, system=system)
        seq.add_block(rf)
        seq.add_block(pp.make_delay(tr - pp.calc_duration(rf)))
    return seq


def test_the_sequence_power_sums_its_pulses(system):
    seq = excited(system)

    mean_pwr, peak_pwr, rf_rms, total_energy = seq.calc_rf_power()

    pulses = [seq.get_block(n).rf for n in (1, 3, 5, 7)]
    powers = [pp.calc_rf_power(rf) for rf in pulses]
    duration = seq.duration()[0]
    assert total_energy == pytest.approx(sum(p[0] for p in powers), rel=1e-9)
    assert peak_pwr == pytest.approx(max(p[1] for p in powers), rel=1e-9)
    assert mean_pwr == pytest.approx(total_energy / duration, rel=1e-9)
    assert rf_rms == pytest.approx(math.sqrt(total_energy / duration), rel=1e-9)


def test_a_window_keeps_its_loudest_stretch(system):
    seq = excited(system, flips=(10, 10, 90, 90, 10, 10))
    energies = [pp.calc_rf_power(seq.get_block(n).rf)[0] for n in (1, 3, 5, 7, 9, 11)]

    mean_pwr, _, rf_rms, total_energy = seq.calc_rf_power(window_duration=20e-3)

    assert total_energy == pytest.approx(energies[2] + energies[3], rel=1e-9)
    assert mean_pwr == pytest.approx(total_energy / 20e-3, rel=1e-9)
    assert rf_rms == pytest.approx(math.sqrt(total_energy / 20e-3), rel=1e-9)


def test_a_block_range_counts_only_its_blocks(system):
    seq = excited(system)

    _, _, _, total_energy = seq.calc_rf_power(block_range=(3, 4))

    assert total_energy == pytest.approx(
        pp.calc_rf_power(seq.get_block(3).rf)[0], rel=1e-9
    )


@pytest.mark.parametrize("window", [None, 25e-3])
def test_the_sequence_power_matches_the_reference_toolbox(system, tmp_path, window):
    reference = pytest.importorskip("pypulseq_matlab_like")
    path = tmp_path / "excited.seq"
    excited(system, flips=(10, 45, 90, 20, 70)).write(path)
    ours = pp.Sequence(system)
    ours.read(path)
    mirror = reference.Sequence()
    mirror.read(str(path))

    found = ours.calc_rf_power(window_duration=window)
    expected = mirror.calc_rf_power(windowDuration=np.nan if window is None else window)

    assert found == pytest.approx(expected, rel=1e-6)


def test_an_empty_sequence_has_no_power(system):
    assert pp.Sequence(system).calc_rf_power() == (0.0, 0.0, 0.0, 0.0)
