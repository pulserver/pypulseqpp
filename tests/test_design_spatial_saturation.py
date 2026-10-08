"""Spatial saturation band: selection along a physical normal, its exemption from the prescription, and the 2D sequences that play it."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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


def flags(block):
    """The label events a block carries, as (name, value)."""
    return [(event.label, int(event.value)) for event in block.label or ()]


def test_band_gradient_is_parallel_to_normal_with_the_sinc_selection_amplitude(
    system,
):
    band = design.SpatialSaturation(
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
    band = design.SpatialSaturation(
        system, (0, 0, 1), POSITION, THICKNESS, duration_s=DURATION, time_bw_product=TBW
    )

    amplitude = TBW / (DURATION * THICKNESS)
    assert band.rf_prep.freq_offset == pytest.approx(amplitude * POSITION, rel=1e-6)
    assert band.rf_prep.phase_offset == pytest.approx(
        -2 * np.pi * band.rf_prep.freq_offset * band.rf_prep.center
    )


def test_band_omits_the_channels_its_normal_does_not_touch(system):
    band = design.SpatialSaturation(system, (0, 1, 0), 0.0, THICKNESS)

    assert hasattr(band, "gy_sel")
    assert not hasattr(band, "gx_sel")
    assert not hasattr(band, "gz_sel")


def test_band_with_a_zero_normal_raises(system):
    with pytest.raises(ValueError):
        design.SpatialSaturation(system, (0, 0, 0), 0.0, THICKNESS)


# ----------------------------------------------------------------------
# The exemption from the prescription
# ----------------------------------------------------------------------


def test_the_pulse_block_is_exempt_and_the_spoiler_block_clears_it(system):
    """Pulseq labels are sticky, so an uncleared flag would exempt the whole scan."""
    band = design.SpatialSaturation(
        system, (0, 1, 0), POSITION, THICKNESS, labels=("LIN",)
    )

    assert flags(band.seq.get_block(1)) == [("NOPOS", 1), ("NOROT", 1), ("LIN", 0)]
    assert flags(band.seq.get_block(2)) == [("NOPOS", 0), ("NOROT", 0)]


def _scan_with(band, system):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    seq = pp.Sequence(system)
    for block in band.blocks:
        seq.add_block(*block)
    for block in excitation.blocks:
        seq.add_block(*block)
    return seq


def test_a_later_offset_moves_the_imaging_pulse_and_not_the_band(system):
    band = design.SpatialSaturation(system, (0, 0, 1), POSITION, THICKNESS)
    scan = _scan_with(band, system)
    moved = pp.TransformFOV(translation=(0.0, 0.0, 0.04)).apply_to_sequence(scan)

    assert float(moved.get_block(1).rf.freq_offset) == pytest.approx(
        float(scan.get_block(1).rf.freq_offset)
    )
    assert float(moved.get_block(3).rf.freq_offset) != pytest.approx(
        float(scan.get_block(3).rf.freq_offset)
    )


def test_a_later_rotation_turns_the_imaging_pulse_and_not_the_band(system):
    band = design.SpatialSaturation(system, (0, 1, 0), POSITION, THICKNESS)
    scan = _scan_with(band, system)
    turned = pp.TransformFOV(
        rotation=Rotation.from_euler("x", 20, degrees=True).as_matrix()
    ).apply_to_sequence(scan)

    assert turned.get_block(1).rotation is None
    assert turned.get_block(3).rotation is not None


# ----------------------------------------------------------------------
# A sequence function's bands
# ----------------------------------------------------------------------


def test_a_band_without_a_thickness_is_not_played(system):
    bands = design.spatial_saturations(
        system, (((0, 1, 0), 0.04, 0.02), ((0, 0, 0), 0.0, 0.0))
    )

    assert len(bands) == 1


@pytest.mark.parametrize(
    "band",
    [((0, 0, 0), 0.0, 0.02), ((0, 1, 0), 0.0, -0.02)],
    ids=["zero normal", "negative thickness"],
)
def test_a_band_that_cannot_be_played_is_refused(system, band):
    with pytest.raises(ValueError):
        design.spatial_saturations(system, (band,))


SMALL = {"n_x": 32, "n_y": 16, "n_dummy": 0}
BAND_1 = {"sat1_normal_y": 1.0, "sat1_position": 0.04, "sat1_thickness": 0.02}
BAND_2 = {"sat2_normal_z": 1.0, "sat2_position": -0.03, "sat2_thickness": 0.02}
SEQUENCES = {"gre2D": design.gre2D_sequence, "se2D": design.se2D_sequence}


def saturation_pulses(seq):
    return sum(
        1
        for index in seq.block_events
        if (rf := seq.get_block(index).rf) is not None and rf.use == "saturation"
    )


def excitations(seq):
    return sum(
        1
        for index in seq.block_events
        if (rf := seq.get_block(index).rf) is not None and rf.use == "excitation"
    )


@pytest.mark.parametrize("name", SEQUENCES)
@pytest.mark.parametrize(
    "bands, count", [({}, 0), (BAND_1, 1), ({**BAND_1, **BAND_2}, 2)]
)
def test_a_2d_sequence_plays_each_band_before_every_excitation(name, bands, count):
    seq = SEQUENCES[name](**SMALL, tr=None, **bands)

    assert saturation_pulses(seq) == count * excitations(seq)
    assert seq.check_timing()[0]


@pytest.mark.parametrize("name", SEQUENCES)
def test_a_2d_sequence_counts_its_bands_in_the_repetition_time(name):
    plain = SEQUENCES[name](**SMALL, tr=None)
    banded = SEQUENCES[name](**SMALL, tr=None, **BAND_1, **BAND_2)

    band = design.SpatialSaturation(
        plain.system, (0, 1, 0), 0.04, 0.02, duration_s=3e-3, voxel_size_m=5e-3
    )
    gained = banded.definitions["TR"][0] - plain.definitions["TR"][0]
    assert gained == pytest.approx(2 * band.duration, abs=1e-6)


@pytest.mark.parametrize("name", SEQUENCES)
def test_a_2d_sequence_refuses_a_band_with_a_thickness_and_no_normal(name):
    with pytest.raises(ValueError):
        SEQUENCES[name](**SMALL, sat1_thickness=0.02)


def test_the_2d_epi_plays_no_spatial_band():
    *_, seq = design.epi2D_sequence(**SMALL)

    assert saturation_pulses(seq) == 0
    with pytest.raises(TypeError):
        design.epi2D_sequence(**SMALL, **BAND_1)
