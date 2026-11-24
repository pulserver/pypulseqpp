"""Fat inversion (SPAIR): a flat, B1-insensitive band over fat, and the time fat crosses zero after it."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences as design

FAT_PEAKS_PPM = (-3.8, -3.4, -2.6)


def mz_after(system, ppm, *, b1=1.0, **kwargs):
    """Longitudinal magnetisation after the pulse at ``ppm`` from water, B1 scaled by ``b1``.

    The band is designed at zero offset, so the simulation's frequency axis is
    the offset from its centre.
    """
    centre = kwargs.pop(
        "freq_offset_ppm", design.preparation.fatsat.FAT_INVERSION_CENTRE_PPM
    )
    pulse = design.FatInversion(system, freq_offset_ppm=0.0, **kwargs).rf_prep
    pulse.amplitude = pulse.amplitude * b1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mz, _, frequency, *_ = pp.sim_rf(
            pulse, rephase_factor=0.0, bandwidth_multiplier=6.0, df=0.5
        )
    hz = (np.asarray(ppm) - centre) * 1e-6 * system.gamma * system.B0
    return np.interp(hz, frequency, mz)


@pytest.mark.parametrize("field_t", [3.0, 7.0])
def test_every_methylene_and_methyl_peak_is_inverted_alike(field_t):
    mz = mz_after(pp.Opts(B0=field_t), FAT_PEAKS_PPM)
    assert np.all(mz < -0.85)


@pytest.mark.parametrize("field_t", [3.0, 7.0])
def test_water_moved_150_hz_by_the_head_is_left_alone(field_t):
    system = pp.Opts(B0=field_t)
    hz = np.array([-150.0, 0.0, 150.0])
    mz = mz_after(system, hz / (1e-6 * system.gamma * field_t))
    assert np.all(mz > 0.99)


def test_the_inversion_holds_where_the_7t_head_receives_half_the_nominal_b1():
    system = pp.Opts(B0=7.0)
    for b1 in (0.5, 0.8, 1.0, 1.3):
        assert mz_after(system, -3.4, b1=b1) < -0.6


def test_a_higher_adiabaticity_inverts_at_lower_b1():
    system = pp.Opts(B0=7.0)
    assert mz_after(system, -3.4, b1=0.4, adiabaticity=16.0) < mz_after(
        system, -3.4, b1=0.4
    )


def test_the_band_is_carried_in_ppm_so_it_follows_the_field():
    spair = design.FatInversion(pp.Opts(B0=7.0))
    assert float(spair.rf_prep.freq_ppm) == pytest.approx(spair.freq_offset_ppm)
    stronger, weaker = (
        design.FatInversion(pp.Opts(B0=7.0)),
        design.FatInversion(pp.Opts(B0=3.0)),
    )
    assert stronger.bandwidth_hz == pytest.approx(7.0 / 3.0 * weaker.bandwidth_hz)
    assert weaker.pulse_duration == pytest.approx(
        7.0 / 3.0 * stronger.pulse_duration, rel=0.01
    )


def test_a_pulse_over_the_peak_b1_is_lengthened_to_fit_it():
    free = design.FatInversion(pp.Opts(B0=7.0), adiabaticity=16.0)
    peak = float(np.max(np.abs(free.rf_prep.signal)))
    limited = design.FatInversion(
        pp.Opts(B0=7.0, max_b1=0.8 * peak / 42.577e6 * 1e6, b1_unit="uT"),
        adiabaticity=16.0,
    )
    assert limited.pulse_duration > free.pulse_duration


def test_it_is_a_pulse_and_a_three_axis_spoiler():
    spair = design.FatInversion(pp.Opts(B0=7.0))
    assert len(spair.blocks) == 2
    assert {e.channel for e in spair.blocks[1] if hasattr(e, "channel")} == {
        "x",
        "y",
        "z",
    }


def test_an_unspoiled_inversion_is_refused():
    with pytest.raises(ValueError, match="spoiled"):
        design.FatInversion(pp.Opts(B0=7.0), spoiling_cycles=0.0)


@pytest.mark.parametrize("period_s", [0.05, 0.08, 0.3, 2.0])
def test_fat_is_zero_at_the_null_time_in_the_steady_state_of_one_inversion_every_period(
    period_s,
):
    t1 = design.FAT_T1_S
    mz = 1.0
    for _ in range(400):
        mz = 1.0 - (1.0 + mz) * math.exp(-period_s / t1)
    null = design.fat_null_inversion_time(period_s)
    assert 1.0 - (1.0 + mz) * math.exp(-null / t1) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < null < period_s / 2


def test_a_period_that_is_not_positive_is_refused():
    with pytest.raises(ValueError):
        design.fat_null_inversion_time(0.0)


# ----------------------------------------------------------------------
# In the 2D EPI
# ----------------------------------------------------------------------


def rf_centres(seq):
    """``(time of the pulse's centre, use)`` of every RF event, in play order."""
    centres, start = [], 0.0
    for index in range(1, len(seq.block_events) + 1):
        block = seq.get_block(index)
        if block.rf is not None:
            centres.append(
                (start + float(block.rf.delay) + float(block.rf.center), block.rf.use)
            )
        start += float(seq.block_durations[index])
    return centres


@pytest.mark.parametrize("tr", [None, 2.0])
@pytest.mark.parametrize("file", [0, -1], ids=["calibration", "imaging"])
def test_every_2d_epi_excitation_comes_when_fat_crosses_zero_after_its_inversion(
    tr, file
):
    chain = design.epi2D_sequence(
        pp.Opts(B0=7.0),
        n_x=32,
        n_y=32,
        ry=2,
        n_shots=3,
        tr=tr,
        n_dummy=2,
        fat_inversion=True,
        field_pair=False,
    )
    played = rf_centres(chain[file])
    inversions = [t for t, use in played if use == "inversion"]
    excitations = [t for t, use in played if use == "excitation"]
    assert len(inversions) == len(excitations) > 2
    periods = np.diff(inversions)
    assert np.ptp(periods) < 1e-4
    null = design.fat_null_inversion_time(float(periods[0]))
    np.testing.assert_allclose(
        np.subtract(excitations, inversions), null, atol=2 * pp.Opts().grad_raster_time
    )


def test_a_2d_epi_that_both_saturates_and_inverts_fat_is_refused():
    with pytest.raises(ValueError, match="saturated or inverted"):
        design.epi2D_sequence(
            n_x=32, n_y=16, fat_saturation=True, fat_inversion=True, field_pair=False
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"bandwidth_ppm": 0.0},
        {"adiabaticity": 0.0},
        {"voxel_size_m": 0.0},
        {"duration_s": -1e-3},
    ],
)
def test_a_band_adiabaticity_voxel_or_duration_that_is_not_positive_is_refused(bad):
    with pytest.raises(ValueError, match="positive"):
        design.FatInversion(pp.Opts(B0=7.0), **bad)
