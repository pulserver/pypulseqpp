"""Amplitudes and pads a hand-written scan loop needs from the modules."""

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
