"""Pulses fitted to the system's peak-B1 limit: kept when within it, lengthened when not."""

from __future__ import annotations

import copy
import importlib

import numpy as np
import pypulseq as upstream
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences as design
from pypulseqpp.sequences.excitation._base import b1_fitted

se2D = importlib.import_module("pypulseqpp.sequences.sequence.se2D_sequence")
zte3D = importlib.import_module("pypulseqpp.sequences.sequence.zte3D_sequence")

GAMMA = 42.576e6


def _limits(**kwargs):
    return pp.Opts(
        max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s", **kwargs
    )


def _peak(rf):
    return float(np.abs(np.asarray(rf.signal)).max())


def _design_limit(system):
    """The peak a fitted pulse may reach (Hz): the derated ``max_b1``."""
    return pp.MAX_B1_DERATE * system.max_b1


def _refocusing_thickness(module):
    return pp.calc_rf_bandwidth(module.rf_ref) / abs(module.selection_amplitude)


# --- Opts and cap_system --------------------------------------------------------


def test_opts_store_the_limit_as_gamma_b1():
    system = pp.Opts(max_b1=11.2)
    assert system.max_b1 == pytest.approx(11.2e-6 * system.gamma)
    assert pp.Opts(max_b1=0.112, b1_unit="G").max_b1 == pytest.approx(system.max_b1)
    assert pp.Opts(max_b1=500.0, b1_unit="Hz").max_b1 == 500.0


def test_opts_without_a_limit_have_none():
    assert pp.Opts().max_b1 is None


def test_an_unknown_b1_unit_is_refused():
    with pytest.raises(ValueError, match="b1_unit"):
        pp.Opts(max_b1=11.2, b1_unit="mT/m")


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_limit_that_is_not_positive_is_refused(value):
    with pytest.raises(ValueError, match="positive"):
        pp.Opts(max_b1=value)
    with pytest.raises(ValueError, match="positive"):
        pp.cap_system(pp.Opts(), max_b1=value)


def test_copies_and_the_text_form_carry_the_limit():
    system = pp.Opts(max_b1=11.2)
    assert copy.copy(system).max_b1 == system.max_b1
    assert copy.deepcopy(system).max_b1 == system.max_b1
    assert "max_b1" in str(system)


def test_cap_system_sets_a_limit_on_a_system_without_one():
    capped = pp.cap_system(pp.Opts(), max_b1=11.2)
    assert capped.max_b1 == pytest.approx(11.2e-6 * capped.gamma)


def test_cap_system_lowers_and_never_raises_the_limit():
    system = pp.Opts(max_b1=15.0)
    lowered = pp.cap_system(system, max_b1=11.2)
    assert lowered.max_b1 == pytest.approx(11.2e-6 * system.gamma)
    assert system.max_b1 == pytest.approx(15.0e-6 * system.gamma)
    assert pp.cap_system(system, max_b1=20.0).max_b1 == system.max_b1
    assert pp.cap_system(system, max_grad=20.0).max_b1 == system.max_b1


def test_cap_system_takes_a_plain_pypulseq_system():
    plain = upstream.Opts()
    assert getattr(pp.cap_system(plain, max_grad=20.0), "max_b1", None) is None
    capped = pp.cap_system(plain, max_b1=1.0, b1_unit="G")
    assert capped.max_b1 == pytest.approx(1e-4 * plain.gamma)
    assert not hasattr(plain, "max_b1")


# --- Derate ---------------------------------------------------------------------


def test_derating_scales_the_limit_once_and_leaves_the_scanners():
    system = pp.Opts(max_b1=11.2)
    once = pp.apply_system_derates(system)
    assert once.max_b1 == pytest.approx(pp.MAX_B1_DERATE * system.max_b1)
    assert pp.apply_system_derates(once).max_b1 == pytest.approx(once.max_b1)
    assert system.max_b1 == pytest.approx(11.2e-6 * system.gamma)
    half = pp.apply_system_derates(system, b1_derate=0.5)
    assert half.max_b1 == pytest.approx(0.5 * system.max_b1)


def test_derating_a_system_without_a_limit_sets_none():
    assert pp.apply_system_derates(pp.Opts()).max_b1 is None
    assert not hasattr(pp.apply_system_derates(upstream.Opts()), "max_b1")


def test_a_pulse_is_fitted_to_the_margin_whether_or_not_the_system_was_derated():
    system = _limits(max_b1=11.2)
    raw = design.SpatialSelectiveRefocusing(system, 5e-3)
    derated = design.SpatialSelectiveRefocusing(pp.apply_system_derates(system), 5e-3)
    assert raw.pulse_duration == derated.pulse_duration
    np.testing.assert_array_equal(raw.rf_ref.signal, derated.rf_ref.signal)
    custom = pp.apply_system_derates(system, b1_derate=0.5)
    assert design.SpatialSelectiveRefocusing(custom, 5e-3).pulse_duration > (
        raw.pulse_duration
    )


# --- b1_fitted ------------------------------------------------------------------


def _block(system):
    return lambda duration: pp.make_block_pulse(np.pi, duration=duration, system=system)


def test_a_pulse_within_the_limit_is_designed_as_given():
    system = pp.Opts(max_b1=20.0)
    duration, rf = b1_fitted(system, 1e-3, _block(system))
    assert duration == 1e-3
    assert rf.shape_dur == pytest.approx(1e-3)


def test_a_pulse_above_the_limit_is_lengthened_by_the_ratio():
    """A 180 hard pulse peaks at 1/(2 duration): 2.610 ms at 0.9 of 5 uT, 2.62 ms on the raster."""
    system = pp.Opts(max_b1=5.0)
    duration, rf = b1_fitted(system, 1e-3, _block(system))
    assert duration == pytest.approx(2.62e-3)
    assert _peak(rf) <= _design_limit(system)
    assert duration - system.grad_raster_time < 1 / (2 * _design_limit(system))


def test_a_pulse_that_cannot_fit_is_refused():
    system = pp.Opts(max_b1=5.0)
    fixed = pp.make_block_pulse(np.pi, duration=1e-3, system=system)
    with pytest.raises(ValueError, match="designed to"):
        b1_fitted(system, 1e-3, lambda _: fixed)


# --- Modules --------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda system: design.SpatialSelectiveExcitation(system, 90.0, 5e-3).rf,
        lambda system: design.SpatialSelectiveRefocusing(system, 5e-3).rf_ref,
        lambda system: design.NonSelectiveExcitation(system, 90.0, 0.2e-3).rf,
        lambda system: design.NonSelectiveRefocusing(system, duration_s=0.2e-3).rf_ref,
    ],
    ids=[
        "selective",
        "selective-refocusing",
        "nonselective",
        "nonselective-refocusing",
    ],
)
def test_without_a_limit_or_within_it_the_pulse_is_unchanged(build):
    reference = build(_limits())
    rf = build(_limits(max_b1=100.0))
    np.testing.assert_array_equal(rf.signal, reference.signal)
    assert rf.shape_dur == reference.shape_dur


def test_a_plain_pypulseq_system_designs_every_fitted_module():
    plain = upstream.Opts(
        max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s"
    )
    assert design.SpatialSelectiveExcitation(plain, 90.0, 5e-3).pulse_duration == 3e-3
    assert design.SpatialSelectiveRefocusing(plain, 5e-3).pulse_duration == 3e-3
    assert design.NonSelectiveExcitation(plain, 90.0).pulse_duration == 1e-3
    assert design.NonSelectiveRefocusing(plain).pulse_duration == 1e-3


def test_a_selective_excitation_is_lengthened_and_keeps_its_slice():
    free = design.SpatialSelectiveExcitation(_limits(), 90.0, 5e-3)
    system = _limits(max_b1=5.0)
    held = design.SpatialSelectiveExcitation(system, 90.0, 5e-3)
    assert _peak(free.rf) > _design_limit(system)
    assert held.pulse_duration > free.pulse_duration
    assert held.rf.shape_dur == pytest.approx(held.pulse_duration)
    assert _peak(held.rf) <= _design_limit(system)
    assert held.slice_thickness == pytest.approx(5e-3, rel=1e-3)
    # The time-bandwidth product is held, so the selection plateau is lower.
    ratio = held.pulse_duration / free.pulse_duration
    assert abs(held.selection_amplitude) == pytest.approx(
        abs(free.selection_amplitude) / ratio, rel=0.02
    )


def test_a_selective_refocusing_is_lengthened_and_keeps_its_slice():
    """The 7 T case: a 3 ms 180 of 30 uT, under 11.2 uT held to 10.08 uT, is about 8.9 ms."""
    free = design.SpatialSelectiveRefocusing(_limits(), 5e-3)
    system = _limits(max_b1=11.2)
    held = design.SpatialSelectiveRefocusing(system, 5e-3)
    assert _peak(free.rf_ref) / GAMMA * 1e6 == pytest.approx(29.9, abs=0.1)
    assert held.pulse_duration == pytest.approx(8.9e-3, abs=0.1e-3)
    assert _peak(held.rf_ref) <= _design_limit(system)
    assert _refocusing_thickness(held) == pytest.approx(
        _refocusing_thickness(free), rel=1e-2
    )


@pytest.mark.parametrize("refocusing", [False, True])
def test_a_nonselective_pulse_is_lengthened_with_its_centre_on_the_raster(refocusing):
    system = _limits(max_b1=11.2)
    if refocusing:
        module = design.NonSelectiveRefocusing(system, duration_s=0.2e-3)
        rf = module.rf_ref
    else:
        module = design.NonSelectiveExcitation(system, 90.0, 0.2e-3)
        rf = module.rf
    assert module.pulse_duration > 0.2e-3
    assert _peak(rf) <= _design_limit(system)
    raster = system.block_duration_raster
    assert module.center / raster == pytest.approx(round(module.center / raster))


def test_a_hard_pulse_off_the_block_raster_is_lengthened_on_the_rf_raster():
    """A ZTE pulse is brief beside the dwell, so it is not padded to two block rasters."""
    system = _limits(max_b1=11.2)
    module = design.NonSelectiveExcitation(system, 4.0, 10e-6)
    assert 10e-6 < module.pulse_duration < 2 * system.block_duration_raster
    samples = module.pulse_duration / system.rf_raster_time
    assert samples == pytest.approx(round(samples))
    assert _peak(module.rf) <= _design_limit(system)


# --- Sequences ------------------------------------------------------------------

SMALL = {"n_dummy": 0, "n_x": 32, "n_y": 16, "n_acs_y": 0, "tr": None, "te": None}


def _rf_peaks(seq):
    return [
        _peak(block.rf)
        for block in (seq.get_block(i) for i in range(1, seq.num_blocks + 1))
        if getattr(block, "rf", None) is not None
    ]


def test_a_spin_echo_designs_under_the_7t_limit():
    free = se2D.se2d(**SMALL)
    system = pp.Opts(max_b1=11.2)
    held = se2D.se2d(system, **SMALL)
    assert held.check_timing()[0]
    assert max(_rf_peaks(held)) <= _design_limit(system)
    assert max(_rf_peaks(free)) > system.max_b1
    durations = sorted(
        block.rf.shape_dur
        for block in (held.get_block(i) for i in range(1, held.num_blocks + 1))
        if getattr(block, "rf", None) is not None
    )
    assert durations[-1] >= se2D.PULSE_DURATION
    # The shortest echo time follows the lengthened refocusing pulse.
    assert held.get_definition("TE")[0] > free.get_definition("TE")[0]


def test_a_zte_designs_under_the_7t_limit_at_a_flip_its_dwell_admits():
    """At 3 degrees the 10 us pulse needs 20 us under 10.08 uT, two dwells; 2.5 fits in 18."""
    system = pp.Opts(max_b1=11.2)
    with pytest.raises(ValueError, match="dwell"):
        zte3D.main(system, n=32, n_shots=2, n_dummy=0)
    held = zte3D.main(system, n=32, n_shots=2, n_dummy=0, flip_angle_deg=2.5)
    seq = held[-1] if isinstance(held, list) else held
    assert seq.check_timing()[0]
    assert max(_rf_peaks(seq)) <= _design_limit(system)
