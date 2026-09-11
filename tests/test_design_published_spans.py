"""Amplitudes and pads a hand-written scan loop needs from the modules."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences as design


def test_a_refocusing_pulse_selects_under_its_plateau_not_its_crushers():
    refocusing = design.SpatialSelectiveRefocusing(
        pp.Opts(), 5e-3, duration_s=3e-3, time_bw_product=4.0, spoiling_cycles=4.0
    )

    assert refocusing.selection_amplitude == pytest.approx(
        4.0 / (3e-3 * 5e-3), rel=1e-3
    )
    assert abs(refocusing.gz.amplitude) > 2 * refocusing.selection_amplitude


def test_a_slab_selects_under_its_lobe_not_the_rephaser_merged_into_it():
    slab = design.SpatialSelectiveExcitation(pp.Opts(), 90.0, 0.1, is_slab=True)
    plain = design.SpatialSelectiveExcitation(pp.Opts(), 90.0, 0.1)

    assert slab.selection_amplitude == pytest.approx(plain.gz.amplitude)


def test_the_prewinder_and_rewinder_blocks_last_what_their_pads_say():
    system = pp.Opts()
    slab = design.SpatialSelectiveExcitation(system, 10.0, 0.128, is_slab=True)
    readout = design.SpiralStackReadout(
        system,
        slab.rf,
        slab.gz,
        None,
        fov=0.22,
        matrix=32,
        fov_z=0.128,
        matrix_z=4,
        design_interleaves=8,
    )
    durations = [pp.calc_duration(*block) for block in readout.blocks]

    for pad in (readout.wait_pre, readout.wait_rew):
        assert any(d == pytest.approx(pad.delay) for d in durations)


def test_a_spectral_spatial_slice_moves_where_it_is_shifted():
    """The profile a Bloch simulation of the shifted pulse excites is centred there."""
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    water = design.SpspExcitation(
        system, 30.0, thickness_m=10e-3, spectral_bandwidth_hz=300.0
    )
    rf, gz = water.rf, water.gz
    samples = rf.delay + np.asarray(rf.t)
    gradient = np.interp(samples, gz.delay + np.asarray(gz.tt), np.asarray(gz.waveform))
    z = np.linspace(-25e-3, 25e-3, 201)
    dt = float(np.diff(rf.t).mean())
    designed = np.array(rf.signal)

    def centre_and_peak():
        m = pp.sim_bloch(np.asarray(rf.signal), z[:, None] * gradient[None, :], dt)
        excited = np.abs(m[:, 0] + 1j * m[:, 1])
        return (excited * z).sum() / excited.sum(), excited.max()

    centred, peak = centre_and_peak()
    water.shift(6e-3, phase=0.3)
    moved, moved_peak = centre_and_peak()

    assert centred == pytest.approx(0.0, abs=0.1e-3)
    assert moved == pytest.approx(6e-3, abs=0.2e-3)
    assert moved_peak == pytest.approx(peak, rel=1e-3)
    assert rf.phase_offset == pytest.approx(0.3)
    assert np.allclose(np.asarray(water.shift(0.0).signal), designed)
