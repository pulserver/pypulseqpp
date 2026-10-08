"""Fat saturation: its selectivity, and a prescription leaving it as it is."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences as design

no_transform = pytest.mark.skipif(
    not hasattr(pp, "TransformFOV"),
    reason="TransformFOV is not in this namespace yet",
)


def response(module, **kwargs):
    """MATLAB's six outputs, under the names MATLAB gives them."""
    mz_z, mz_xy, frequency, ref_eff, mx_xy, my_xy = module.sim_rf(**kwargs)
    return SimpleNamespace(
        mz_z=mz_z,
        mz_xy=mz_xy,
        frequency=frequency,
        ref_eff=ref_eff,
        mx_xy=mx_xy,
        my_xy=my_xy,
    )


FAT_SHIFT_PPM = -3.45


@pytest.fixture
def system():
    return pp.Opts(
        B0=3.0, max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s"
    )


def flags(block):
    """The label events a block carries, as (kind, name, value)."""
    return [(event.type, event.label, int(event.value)) for event in block.label or ()]


def rf_phase_ramp(reference, moved):
    """Hz of linear phase the placement wrote across a pulse."""
    times = np.asarray(moved.t, dtype=float)
    difference = np.unwrap(
        np.angle(np.asarray(moved.signal)) - np.angle(np.asarray(reference.signal))
    )
    return float(np.polyfit(times, difference, 1)[0] / (2 * np.pi))


# ----------------------------------------------------------------------
# The pulse
# ----------------------------------------------------------------------


def test_it_sits_on_the_fat_resonance(system):
    fatsat = design.FatSaturation(system)
    assert fatsat.freq_offset_ppm == pytest.approx(FAT_SHIFT_PPM)
    assert float(fatsat.rf_prep.freq_ppm) == pytest.approx(FAT_SHIFT_PPM)
    assert fatsat.freq_offset_hz == pytest.approx(
        FAT_SHIFT_PPM * 1e-6 * system.B0 * system.gamma
    )


def test_the_offset_is_carried_in_ppm_so_it_follows_the_field(system):
    """Hertz would be wrong at any field but the one it was designed at."""
    fatsat = design.FatSaturation(system, freq_offset_ppm=-3.2)
    assert float(fatsat.rf_prep.freq_ppm) == pytest.approx(-3.2)
    assert fatsat.freq_offset_hz == pytest.approx(-3.2e-6 * system.B0 * system.gamma)
    stronger = design.FatSaturation(pp.Opts(B0=7.0), freq_offset_ppm=-3.2)
    assert float(stronger.rf_prep.freq_ppm) == pytest.approx(-3.2)
    assert abs(stronger.freq_offset_hz) > 2 * abs(fatsat.freq_offset_hz)


def test_it_overshoots_the_transverse_plane(system):
    """110 degrees, so a transmit shortfall still leaves fat near null."""
    answer = response(design.FatSaturation(system))
    assert float(np.min(answer.mz_z)) == pytest.approx(
        np.cos(np.deg2rad(110.0)), abs=0.02
    )


def test_it_leaves_water_alone(system):
    """The stopband has to reach the water peak, 3.45 ppm away."""
    fatsat = design.FatSaturation(system)
    answer = response(fatsat)
    water = float(np.interp(0.0, answer.frequency, answer.mz_z))
    assert water > 0.95


def test_the_pulse_duration_follows_the_bandwidth_asked_for(system):
    narrow = design.FatSaturation(system, bandwidth_hz=125.0)
    wide = design.FatSaturation(system, bandwidth_hz=500.0)
    assert pp.calc_duration(narrow.rf_prep) == pytest.approx(
        4.0 * pp.calc_duration(wide.rf_prep), rel=0.02
    )


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_it_is_a_pulse_and_a_three_axis_spoiler(system):
    fatsat = design.FatSaturation(system)
    assert len(fatsat.blocks) == 2
    assert {
        event.channel for event in fatsat.blocks[1] if hasattr(event, "channel")
    } == {
        "x",
        "y",
        "z",
    }


def test_it_plays_no_gradient_and_no_flags(system):
    """Nothing about it depends on where the image is, so nothing exempts it."""
    fatsat = design.FatSaturation(system)
    assert not hasattr(fatsat.events, "gz")
    assert [event.type for event in fatsat.blocks[0]] == ["rf"]
    assert flags(fatsat.seq.get_block(1)) == flags(fatsat.seq.get_block(2)) == []
    assert fatsat.check_timing()[0]


# ----------------------------------------------------------------------
# A prescription applied afterwards
# ----------------------------------------------------------------------


def _scan_with(fatsat, system):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    seq = pp.Sequence(system)
    for block in fatsat.blocks:
        seq.add_block(*block)
    for block in excitation.blocks:
        seq.add_block(*block)
    return seq


@no_transform
def test_a_later_offset_moves_the_imaging_pulse_and_leaves_the_fat_saturation(system):
    """Under no gradient, a shift writes no frequency or phase onto the pulse."""
    scan = _scan_with(design.FatSaturation(system), system)
    moved = pp.TransformFOV(translation=(0.01, -0.02, 0.04)).apply_to_sequence(scan)

    saturation_before, saturation_after = scan.get_block(1).rf, moved.get_block(1).rf
    imaging_before, imaging_after = scan.get_block(3).rf, moved.get_block(3).rf

    assert rf_phase_ramp(saturation_before, saturation_after) == pytest.approx(
        0.0, abs=1e-9
    )
    assert float(saturation_after.freq_offset) == pytest.approx(
        float(saturation_before.freq_offset)
    )
    assert float(saturation_after.freq_ppm) == pytest.approx(FAT_SHIFT_PPM)
    assert float(imaging_after.freq_offset) != pytest.approx(
        float(imaging_before.freq_offset)
    )


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_an_unspoiled_saturation_is_refused(system):
    with pytest.raises(ValueError, match="has to be spoiled"):
        design.FatSaturation(system, spoiling_cycles=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
        ({"voxel_size_m": 0.0}, "voxel_size_m"),
    ],
)
def test_it_checks_its_arguments(system, kwargs, message):
    with pytest.raises(ValueError, match=message):
        design.FatSaturation(system, **kwargs)


# ----------------------------------------------------------------------
# The flip that nulls fat at the excitation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delay", "period"), [(5e-3, 50e-3), (3e-3, 0.5), (10e-3, 30e-3)]
)
def test_the_null_flip_leaves_fat_at_zero_at_the_excitation_in_the_steady_state(
    delay, period
):
    flip = np.deg2rad(design.fat_null_flip_deg(delay, period))
    t1 = design.FAT_T1_S
    at_excitation = 1.0
    for _ in range(2000):
        before = 1.0 - np.exp(-(period - delay) / t1)  # recovered from zero
        at_excitation = 1.0 - (1.0 - before * np.cos(flip)) * np.exp(-delay / t1)
    assert at_excitation == pytest.approx(0.0, abs=1e-12)
    assert 90.0 < np.rad2deg(flip) < 180.0


def test_fat_recovering_faster_than_any_flip_can_null_is_refused():
    with pytest.raises(ValueError, match="recovers too fast"):
        design.fat_null_flip_deg(20e-3, 25e-3)


def test_the_2d_epi_saturates_fat_at_the_flip_that_nulls_it_at_its_excitation(
    monkeypatch,
):
    asked = []
    null_flip = design.fat_null_flip_deg

    def recorded(delay, period, **kwargs):
        asked.append((delay, period, null_flip(delay, period, **kwargs)))
        return asked[-1][2]

    monkeypatch.setattr(design, "fat_null_flip_deg", recorded)
    *_, seq = design.epi2D_sequence(n_x=32, n_y=16, n_dummy=0, fat_saturation=True)
    saturation = next(
        rf
        for i in seq.block_events
        if (rf := seq.get_block(i).rf) is not None and rf.use == "saturation"
    )

    ((delay, period, flip),) = asked
    assert 0 < delay < period
    assert 90.0 < flip < 180.0
    expected = design.FatSaturation(seq.system, flip_angle_deg=flip).rf_prep
    np.testing.assert_allclose(
        np.abs(saturation.signal), np.abs(expected.signal), rtol=1e-6
    )
