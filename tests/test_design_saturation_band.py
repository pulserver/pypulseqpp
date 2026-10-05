"""Spatial saturation band: selection along a normal, and its use in the 2D EPI."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences as design

DURATION = 3e-3
TBW = 4.0
THICKNESS = 0.02
POSITION = 0.05
NORMAL = np.array([1.0, 2.0, 2.0]) / 3.0


@pytest.fixture
def system():
    return pp.Opts()


def test_band_gradient_is_parallel_to_normal_with_the_sinc_selection_amplitude(
    system,
):
    band = design.SaturationBand(
        system,
        (3.0, 6.0, 6.0),
        POSITION,
        THICKNESS,
        duration_s=DURATION,
        time_bw_product=TBW,
    )

    amplitude = TBW / (DURATION * THICKNESS)
    vector = np.array(
        [band.gx_sel.amplitude, band.gy_sel.amplitude, band.gz_sel.amplitude]
    )
    assert np.linalg.norm(vector) == pytest.approx(amplitude, rel=1e-6)
    assert vector / np.linalg.norm(vector) == pytest.approx(NORMAL, abs=1e-9)


def test_band_frequency_offset_is_selection_amplitude_times_position(system):
    band = design.SaturationBand(
        system, (0, 0, 1), POSITION, THICKNESS, duration_s=DURATION, time_bw_product=TBW
    )

    amplitude = TBW / (DURATION * THICKNESS)
    assert band.rf_prep.freq_offset == pytest.approx(amplitude * POSITION, rel=1e-6)
    assert band.rf_prep.phase_offset == pytest.approx(
        -2 * np.pi * band.rf_prep.freq_offset * band.rf_prep.center
    )


def test_band_omits_the_channels_its_normal_does_not_touch(system):
    band = design.SaturationBand(system, (0, 1, 0), 0.0, THICKNESS)

    assert hasattr(band, "gy_sel")
    assert not hasattr(band, "gx_sel")
    assert not hasattr(band, "gz_sel")


def test_band_with_a_zero_normal_raises(system):
    with pytest.raises(ValueError):
        design.SaturationBand(system, (0, 0, 0), 0.0, THICKNESS)


SMALL = {"n_x": 32, "n_y": 16, "n_dummy": 0}
BAND_1 = {"sat1_normal_y": 1.0, "sat1_position": 0.04, "sat1_thickness": 0.02}
BAND_2 = {"sat2_normal_z": 1.0, "sat2_position": -0.03, "sat2_thickness": 0.02}


def saturation_pulses(seq):
    return sum(
        1
        for index in seq.block_events
        if (rf := seq.get_block(index).rf) is not None and rf.use == "saturation"
    )


@pytest.mark.parametrize(
    "bands, count", [({}, 0), (BAND_1, 1), ({**BAND_1, **BAND_2}, 2)]
)
def test_epi2d_plays_one_saturation_pulse_per_enabled_band_per_shot(bands, count):
    *_, seq = design.epi2D_sequence(**SMALL, n_shots=2, **bands)

    assert saturation_pulses(seq) == count * 2


def test_epi2d_band_counts_in_the_shot_duration():
    *_, plain = design.epi2D_sequence(**SMALL)
    *_, banded = design.epi2D_sequence(**SMALL, **BAND_1, **BAND_2)

    band = design.SaturationBand(
        plain.system, (0, 1, 0), 0.04, 0.02, duration_s=3e-3, voxel_size_m=5e-3
    )
    gained = banded.definitions["TR"][0] - plain.definitions["TR"][0]
    assert gained == pytest.approx(2 * band.duration, abs=1e-6)


def test_epi2d_band_with_a_zero_normal_raises():
    with pytest.raises(ValueError):
        design.epi2D_sequence(**SMALL, sat1_thickness=0.02)
