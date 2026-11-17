"""Slice rephasing in the first post-RF block and its effect on TE."""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences as design

FOV = 0.22
MATRIX = 128


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


@pytest.fixture
def excitation(system):
    """A 2D slice excitation, whose rephaser is a separate event."""
    return design.SpatialSelectiveExcitation(system, 15.0, 5e-3)


#: The 2D families, each as the extra arguments that pick it out.
FAMILIES = {
    "line": (design.LineReadout2D, {}),
    "radial": (design.RadialReadout2D, {}),
    "spiral_out": (design.SpiralReadout2D, {"design_interleaves": 16}),
    "spiral_in": (
        design.SpiralReadout2D,
        {"design_interleaves": 16, "direction": "inward"},
    ),
}


def readout(family, system, excitation, rephase=True, **kwargs):
    """One 2D readout of ``family``, carrying the excitation's rephaser."""
    factory, family_kwargs = FAMILIES[family]
    return factory(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph if rephase else None,
        fov=FOV,
        matrix=MATRIX,
        **family_kwargs,
        **kwargs,
    )


def peak_kz(module):
    """The largest |kz| reached over the acquisition window."""
    k_traj_adc, *_ = module.calculate_kspace()
    return float(np.abs(k_traj_adc[2, : int(module.adc.num_samples)]).max())


def readout_lobe_opening(module):
    """Time (s) from the start of its block at which the readout lobe opens.

    A spoke's lobe opens where its gradient crosses zero into the plateau,
    rounded down to the gradient raster; a spiral arm's opens with its ADC.
    """
    gx = module.gx[0] if isinstance(module.gx, list) else module.gx
    if not isinstance(module, design.RadialReadout2D):
        return float(module.adc.delay)
    times = float(gx.delay) + np.asarray(gx.tt, dtype=float)
    values = np.asarray(gx.waveform, dtype=float)
    i = np.flatnonzero((values[:-1] <= 0) & (values[1:] > 0))[-1]
    crossing = times[i] - values[i] * (times[i + 1] - times[i]) / (
        values[i + 1] - values[i]
    )
    raster = module.seq.system.grad_raster_time
    return np.floor(crossing / raster + 1e-9) * raster


def block_of(module, event):
    """Index of the block ``event`` was added to."""
    return next(
        index
        for index, block in enumerate(module.blocks)
        if any(played is event for played in block)
    )


# ----------------------------------------------------------------------
# What it is for
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", ["line", "radial", "spiral_out", "spiral_in"])
def test_the_rephaser_brings_the_slice_axis_back_to_zero(system, excitation, family):
    assert peak_kz(readout(family, system, excitation)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("family", ["line", "radial"])
def test_without_it_the_whole_acquisition_sits_off_the_slice_plane(
    system, excitation, family
):
    unrephased = {
        "line": design.LineReadout2D,
        "radial": design.RadialReadout2D,
    }[family](system, excitation.rf, excitation.gz, fov=FOV, matrix=MATRIX)
    assert peak_kz(unrephased) == pytest.approx(0.5 * excitation.gz.area, rel=1e-6)


# ----------------------------------------------------------------------
# Where it goes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("family", ["line", "radial", "spiral_out", "spiral_in"])
def test_at_minimum_te_it_shares_the_block_that_holds_the_prephaser(
    system, excitation, family
):
    module = readout(family, system, excitation)
    assert block_of(module, module.gz_reph) == 1
    assert float(module.gz_reph.delay) == 0.0
    assert not hasattr(module.events, "wait_te")


@pytest.mark.parametrize("family", ["radial", "spiral_out", "spiral_in"])
def test_at_minimum_te_it_is_done_before_the_readout_lobe_opens(
    system, excitation, family
):
    module = readout(family, system, excitation)
    assert pp.calc_duration(module.gz_reph) <= readout_lobe_opening(module) + 1e-9


@pytest.mark.parametrize("family", ["line", "radial", "spiral_out", "spiral_in"])
def test_a_te_wait_takes_it_over_and_it_still_starts_the_block(
    system, excitation, family
):
    short = readout(family, system, excitation)
    module = readout(family, system, excitation, te=short.echo_time + 2e-3)
    assert block_of(module, module.gz_reph) == block_of(module, module.wait_te) == 1
    assert float(module.gz_reph.delay) == 0.0


@pytest.mark.parametrize("family", ["line", "spiral_in"])
def test_a_wait_too_short_to_hold_it_leaves_it_where_it_was(system, excitation, family):
    """The requested TE wins: a rephaser that cannot fit the wait does not move.

    Moving it into a wait shorter than itself would stretch that block, and the
    echo would land later than the TE that was asked for. The prewinder block
    absorbs the wait instead, by starting later.
    """
    short = readout(family, system, excitation)
    reph_span = pp.calc_duration(excitation.gz_reph)
    requested = short.echo_time + 0.5 * reph_span
    module = readout(family, system, excitation, te=requested)
    assert block_of(module, module.gz_reph) == block_of(short, short.gz_reph) + (
        hasattr(module.events, "wait_te")
    )
    assert module.echo_time == pytest.approx(requested)


def test_a_spoke_carries_it_however_long_the_wait_is(system, excitation):
    """A radial spoke prephases inside its own block, so the two never compete."""
    short = readout("radial", system, excitation)
    reph_span = pp.calc_duration(excitation.gz_reph)
    module = readout("radial", system, excitation, te=short.echo_time + 0.5 * reph_span)
    assert block_of(module, module.gz_reph) == block_of(module, module.gx)
    assert module.echo_time == pytest.approx(short.echo_time + 0.5 * reph_span)


@pytest.mark.parametrize("family", ["line", "radial", "spiral_out", "spiral_in"])
@pytest.mark.parametrize("extra", [0.0, 1e-4, 3e-4, 1e-3, 4e-3])
def test_carrying_it_never_costs_the_requested_echo_time(
    system, excitation, family, extra
):
    short = readout(family, system, excitation)
    requested = short.echo_time + extra
    module = readout(family, system, excitation, te=requested)
    assert module.echo_time == pytest.approx(requested)
    assert module.check_timing()[0]


def test_it_lengthens_te_only_by_what_it_outlasts_the_prephaser(system, excitation):
    """The rephaser shares the prephaser's time up to the opening of the readout lobe."""
    with_reph = design.RadialReadout2D(
        system, excitation.rf, excitation.gz, excitation.gz_reph, fov=FOV, matrix=MATRIX
    )
    without = design.RadialReadout2D(
        system, excitation.rf, excitation.gz, fov=FOV, matrix=MATRIX
    )
    reph_span = pp.calc_duration(excitation.gz_reph)
    assert with_reph.echo_time == pytest.approx(
        without.echo_time + max(0.0, reph_span - readout_lobe_opening(without))
    )


def test_an_outward_spiral_starts_its_arm_after_the_rephaser(system, excitation):
    """Nothing prephases an arm that starts at k = 0, so the arm waits for the rephaser."""
    without = design.SpiralReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        fov=FOV,
        matrix=MATRIX,
        design_interleaves=16,
    )
    with_reph = design.SpiralReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=FOV,
        matrix=MATRIX,
        design_interleaves=16,
    )
    assert len(with_reph.blocks) == len(without.blocks)
    assert block_of(with_reph, with_reph.gz_reph) == block_of(with_reph, with_reph.gx)
    assert with_reph.echo_time == pytest.approx(
        without.echo_time + pp.calc_duration(excitation.gz_reph)
    )


def test_only_the_first_echo_of_a_train_is_rephased(system, excitation):
    module = design.RadialReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=FOV,
        matrix=MATRIX,
        n_echoes=3,
    )
    carried = [block for block in module.blocks if module.gz_reph in block]
    assert len(carried) == 1
    assert block_of(module, module.gz_reph) == 1


# ----------------------------------------------------------------------
# When it cannot be carried
# ----------------------------------------------------------------------


def test_a_partition_encode_already_owns_the_slice_axis(system, excitation):
    with pytest.raises(ValueError, match="already plays a gradient on z"):
        design.LineReadout3D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=(FOV, FOV, 0.12),
            matrix=(MATRIX, MATRIX, 64),
        )


def test_a_stack_rephases_the_slice_before_its_partition_encode(system, excitation):
    module = design.RadialStackReadout(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=FOV,
        matrix=MATRIX,
        fov_z=0.12,
        matrix_z=32,
    )
    assert block_of(module, module.gz_reph) < block_of(module, module.gz_enc)
    assert module.check_timing()[0]


@pytest.mark.parametrize(
    "factory",
    [design.RadialProjectionReadout, design.SpiralProjectionReadout],
)
def test_a_projection_rotates_every_axis_so_none_can_hold_it(
    system, excitation, factory
):
    kwargs = (
        {"design_interleaves": 16} if factory is design.SpiralProjectionReadout else {}
    )
    with pytest.raises(ValueError, match="would be turned with the interleaf"):
        factory(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=FOV,
            matrix=MATRIX,
            **kwargs,
        )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_an_in_plane_rotation_refuses_a_rephaser_it_would_turn(system, axis):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3, axis=axis)
    with pytest.raises(ValueError, match=f"rephaser on {axis} would be turned"):
        design.RadialReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=FOV,
            matrix=MATRIX,
        )


def test_a_rephaser_longer_than_the_head_room_delays_the_spoke(system):
    """A thin slice under a long pulse rephases for longer than a short spoke prephases."""
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 2e-3, 6e-3)
    reph_span = pp.calc_duration(excitation.gz_reph)
    without = design.RadialReadout2D(
        system, excitation.rf, excitation.gz, fov=0.4, matrix=32
    )
    assert reph_span > readout_lobe_opening(without)

    module = design.RadialReadout2D(
        system, excitation.rf, excitation.gz, excitation.gz_reph, fov=0.4, matrix=32
    )
    assert len(module.blocks) == len(without.blocks)
    assert block_of(module, module.gz_reph) == block_of(module, module.gx)
    assert module.echo_time == pytest.approx(
        without.echo_time + reph_span - readout_lobe_opening(without)
    )
    assert peak_kz(module) == pytest.approx(0.0, abs=1e-6)
    assert module.check_timing()[0]
