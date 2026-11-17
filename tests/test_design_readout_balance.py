"""A non-Cartesian repetition leaves the same gradient moment at every angle and partition."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pypulseqpp as pp
from pypulseqpp import calc_golden_angles
from pypulseqpp import sequences as design

FOV = 0.22
MATRIX = 64
#: Residual moment no image can tell from zero: a thousandth of a cycle
#: across the field of view (1/m).
NEGLIGIBLE = 1e-3 / FOV


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


@pytest.fixture
def slab(system):
    return design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)


IN_PLANE = {
    "radial": (design.RadialReadout2D, design.RadialStackReadout, {}),
    "spiral_out": (
        design.SpiralReadout2D,
        design.SpiralStackReadout,
        {"design_interleaves": 8},
    ),
    "spiral_in": (
        design.SpiralReadout2D,
        design.SpiralStackReadout,
        {"design_interleaves": 8, "direction": "inward"},
    ),
    "spiral_in_out": (
        design.SpiralReadout2D,
        design.SpiralStackReadout,
        {"design_interleaves": 8, "direction": "in_out"},
    ),
    "rosette": (design.RosetteReadout2D, design.RosetteStackReadout, {}),
}


def played_area(event):
    """Net area (1/m) an event plays, ramps to its first and last value included."""
    if event is None:
        return 0.0
    if event.type == "trap":
        return float(event.area)
    times = float(event.delay) + np.asarray(event.tt, dtype=float)
    values = np.asarray(event.waveform, dtype=float)
    if times[0] > float(event.delay):
        end = float(event.delay) + float(event.shape_dur)
        times = np.concatenate([[float(event.delay)], times, [end]])
        values = np.concatenate([[float(event.first)], values, [float(event.last)]])
    return float(np.trapezoid(values, times))


def shot_moment(readout, angle, kz):
    """Moment the readout and spoiler blocks leave for one spoke or interleaf."""
    in_plane = np.array(
        [
            played_area(readout.gx),
            played_area(readout.gy),
            kz * played_area(readout.gz_enc),
        ]
    )
    turned = Rotation.from_euler("z", angle).as_matrix() @ in_plane
    spoil = getattr(readout.events, "gz_spoil", None)
    return turned + np.array([0.0, 0.0, played_area(spoil)])


@pytest.mark.parametrize("family", IN_PLANE)
def test_a_2d_readout_rewinds_both_in_plane_axes(system, family):
    factory, _, kwargs = IN_PLANE[family]
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    readout = factory(
        system, excitation.rf, excitation.gz, fov=FOV, matrix=MATRIX, **kwargs
    )
    for event in (readout.gx, readout.gy):
        assert played_area(event) == pytest.approx(0.0, abs=NEGLIGIBLE)


@pytest.mark.parametrize("family", IN_PLANE)
def test_every_spoke_and_partition_leaves_the_same_moment(system, slab, family):
    _, factory, kwargs = IN_PLANE[family]
    readout = factory(
        system,
        slab.rf,
        slab.gz,
        fov=FOV,
        matrix=MATRIX,
        fov_z=0.12,
        matrix_z=8,
        spoiling_cycles=4.0,
        **kwargs,
    )
    moments = np.array(
        [
            shot_moment(readout, float(angle), kz)
            for angle in calc_golden_angles(5)
            for kz in np.linspace(-1.0, 1.0, 5)
        ]
    )
    assert np.ptp(moments, axis=0) == pytest.approx(np.zeros(3), abs=NEGLIGIBLE)
    assert moments[0, 2] == pytest.approx(played_area(readout.gz_spoil))


@pytest.mark.parametrize("family", IN_PLANE)
def test_the_spoiler_plays_on_a_fixed_axis_in_a_block_of_its_own(system, slab, family):
    _, factory, kwargs = IN_PLANE[family]
    readout = factory(
        system,
        slab.rf,
        slab.gz,
        fov=FOV,
        matrix=MATRIX,
        fov_z=0.12,
        matrix_z=8,
        spoiling_cycles=4.0,
        **kwargs,
    )
    holding = [block for block in readout.blocks if readout.gz_spoil in block]
    assert holding == [(readout.gz_spoil,)]


@pytest.mark.parametrize("family", IN_PLANE)
def test_every_partition_plays_one_block_definition(system, slab, family):
    """Scaling the partition wave changes amplitudes only, never the waveform shape."""
    _, factory, kwargs = IN_PLANE[family]
    readout = factory(
        system,
        slab.rf,
        slab.gz,
        fov=FOV,
        matrix=MATRIX,
        fov_z=0.12,
        matrix_z=8,
        **kwargs,
    )
    block = next(block for block in readout.blocks if readout.adc in block)
    seq = pp.Sequence(system)
    for kz in np.linspace(-1.0, 1.0, 5):
        seq.add_block(
            *(pp.scale_grad(e, kz) if e is readout.gz_enc else e for e in block)
        )
    assert len(set(np.asarray(seq.block_definitions()).tolist())) == 1


@pytest.mark.parametrize("family", IN_PLANE)
def test_every_sample_of_a_partition_sits_at_one_kz(system, slab, family):
    _, factory, kwargs = IN_PLANE[family]
    readout = factory(
        system,
        slab.rf,
        slab.gz,
        fov=FOV,
        matrix=MATRIX,
        fov_z=0.12,
        matrix_z=8,
        **kwargs,
    )
    kz = readout.calculate_kspace()[0][2, : int(readout.adc.num_samples)]
    assert np.ptp(kz) == pytest.approx(0.0, abs=1e-9)
    assert kz[0] == pytest.approx(8 / (2 * 0.12))


@pytest.mark.parametrize("family", IN_PLANE)
def test_the_first_partition_rewinds_into_a_spoiler_of_the_opposite_sign(
    system, slab, family
):
    """The hardest switching falls on the edge of k-space a safety check plays."""
    _, factory, kwargs = IN_PLANE[family]
    readout = factory(
        system,
        slab.rf,
        slab.gz,
        fov=FOV,
        matrix=MATRIX,
        fov_z=0.12,
        matrix_z=8,
        spoiling_cycles=4.0,
        **kwargs,
    )
    rewinder = -1.0 * np.asarray(readout.gz_enc.waveform)[-3]
    assert np.sign(rewinder) == -np.sign(readout.gz_spoil.amplitude)


def test_a_block_refuses_two_gradients_on_one_axis(system):
    first = pp.make_trapezoid("z", area=100.0, system=system)
    second = pp.make_trapezoid("z", area=-50.0, system=system)
    seq = pp.Sequence(system)
    with pytest.raises(ValueError, match="sum the gradients"):
        seq.add_block(first, second)


def test_a_spoke_encodes_its_partition_up_to_the_opening_of_its_readout_lobe(
    system, slab
):
    """The z lobes end where the spoke crosses zero into the plateau, and resume where it leaves."""
    readout = design.RadialStackReadout(
        system, slab.rf, slab.gz, fov=FOV, matrix=MATRIX, fov_z=0.12, matrix_z=8
    )
    raster = system.grad_raster_time
    gx_times = float(readout.gx.delay) + np.asarray(readout.gx.tt, dtype=float)
    gx = np.asarray(readout.gx.waveform, dtype=float)
    up = np.flatnonzero((gx[:-1] <= 0) & (gx[1:] > 0))[0]
    down = np.flatnonzero((gx[:-1] > 0) & (gx[1:] <= 0))[-1]
    opening = np.interp(0.0, gx[up : up + 2], gx_times[up : up + 2])
    closing = np.interp(0.0, gx[down : down + 2][::-1], gx_times[down : down + 2][::-1])
    z_times = float(readout.gz_enc.delay) + np.asarray(readout.gz_enc.tt, dtype=float)
    z_on = z_times[np.abs(np.asarray(readout.gz_enc.waveform)) > 0]
    encode_end = z_on[z_on < opening + raster].max() + 0.5 * raster
    rewind_start = z_on[z_on > closing - raster].min() - 0.5 * raster
    assert opening - 2 * raster <= encode_end <= opening + raster
    assert closing - raster <= rewind_start <= closing + 2 * raster
