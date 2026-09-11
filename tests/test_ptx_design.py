"""pTx pulse design as a script reaches it: shims, tailored 2D pulses and spokes.

Profiles are checked in a Bloch simulation in which each position sees its own
field -- the transmit channels summed through their B1 maps -- and its own
gradient field, so what is held is what the pulses do rather than the model
they were designed with.
"""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
FOV = 0.2
FLIP = np.deg2rad(10)


def spread(values):
    magnitude = np.abs(values)
    return magnitude.std() / magnitude.mean()


def loops(channels, grid):
    """Loop coils around a disc, each bright on its own side: ``(channels, grid, grid)``."""
    angle = np.linspace(0, 2 * np.pi, channels, endpoint=False)
    x, y = np.meshgrid(
        np.linspace(-1, 1, grid), np.linspace(-1, 1, grid), indexing="ij"
    )
    distance = np.hypot(
        x[None] - 1.5 * np.cos(angle)[:, None, None],
        y[None] - 1.5 * np.sin(angle)[:, None, None],
    )
    return np.exp(-distance) * np.exp(1j * (angle[:, None, None] + 2.0 * x * y))


def bright_centre(grid):
    """One channel, 60 % brighter at the centre than at the edge of the disc."""
    x, y = np.meshgrid(
        np.linspace(-1, 1, grid), np.linspace(-1, 1, grid), indexing="ij"
    )
    return (1.6 - 0.6 * (x**2 + y**2))[None].astype(complex)


def grid_positions(grid, extent=FOV):
    axis = (np.arange(grid) - (grid - 1) / 2) * (extent / grid)
    return np.meshgrid(axis, axis, indexing="ij")


def gradient_at(event, times):
    """What one gradient event plays at ``times`` from its block's start, in Hz/m."""
    if event.type == "trap":
        edges = event.delay + np.cumsum(
            [0, event.rise_time, event.flat_time, event.fall_time]
        )
        return np.interp(
            times, edges, [0, event.amplitude, event.amplitude, 0], left=0, right=0
        )
    # An arbitrary gradient's times are its samples; an extended trapezoid's, its corners.
    return np.interp(
        times,
        event.delay + np.asarray(event.tt),
        np.asarray(event.waveform),
        left=0,
        right=0,
    )


def transverse(rf, gradients, maps, picks, positions):
    """|Mxy| after the pulse at grid points ``picks`` of ``maps``, at ``positions`` (m, 3-D)."""
    waveforms = pp.split_ptx_pulse(rf)
    dwell = float(rf.t[1] - rf.t[0])
    times = rf.delay + (np.arange(waveforms.shape[1]) + 0.5) * dwell
    fields = {g.channel: gradient_at(g, times) for g in gradients}
    drive = np.stack([maps[(slice(None), *pick)] @ waveforms for pick in picks])
    idle = np.zeros_like(times)
    bz = sum(
        positions[:, [i]] * fields.get(axis, idle)[None, :]
        for i, axis in enumerate("xyz")
    )
    m = pp.sim_bloch(drive, np.broadcast_to(bz, drive.shape), dwell)
    return np.abs(m[:, 0] + 1j * m[:, 1])


# %% RF shimming


def test_a_shim_evens_out_the_field_of_loop_coils():
    maps = loops(8, 24)
    x, y = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24), indexing="ij")
    disc = x**2 + y**2 <= 1
    weights = pp.calc_rf_shim(maps, disc)
    combined = np.tensordot(weights, maps, axes=1)[disc]
    unit = np.tensordot(np.ones(8), maps, axes=1)[disc]
    assert spread(combined) < 0.5 * spread(unit)


def test_a_shim_delivers_the_magnitude_asked_for():
    maps = loops(8, 24)
    weights = pp.calc_rf_shim(maps, target=2.0)
    assert np.abs(np.tensordot(weights, maps, axes=1)).mean() == pytest.approx(
        2.0, rel=0.1
    )


def test_a_shim_is_what_make_rf_shim_takes():
    weights = pp.calc_rf_shim(loops(4, 12))
    assert pp.make_rf_shim(weights).shim_vector.shape == (4,)


def test_a_mask_that_does_not_match_the_maps_is_refused():
    with pytest.raises(ValueError, match="mask must be"):
        pp.calc_rf_shim(loops(4, 12), np.ones((5, 5), dtype=bool))


# %% spatial-domain 2D pulses


def selective(maps=None, **kwargs):
    arguments = {"fov": FOV, "matrix": 16, "selective_size": 0.08, "system": SYSTEM}
    return pp.make_2d_selective_pulse(FLIP, b1_maps=maps, **(arguments | kwargs))


def in_plane(picks, grid=16):
    x, y = grid_positions(grid)
    return np.array([[x[p], y[p], 0.0] for p in picks])


def test_a_tailored_pulse_drives_every_channel():
    rf, _, _ = selective(loops(4, 16))
    assert pp.split_ptx_pulse(rf).shape[0] == 4


def test_one_uniform_channel_excites_the_disc_and_not_beyond():
    maps = np.ones((1, 16, 16), dtype=complex)
    rf, gradients, _ = selective(maps)
    centre, outside = (8, 8), (1, 8)
    magnitude = transverse(
        rf, gradients, maps, [centre, outside], in_plane([centre, outside])
    )
    assert magnitude[0] == pytest.approx(np.sin(FLIP), rel=0.2)
    assert magnitude[1] < 0.2 * magnitude[0]


def test_a_tailored_pulse_flattens_the_b1_it_was_designed_for():
    """Inside the disc, where the target asks for the flip; the plain design
    carries the map's 60 % hump into the profile."""
    maps = bright_centre(16)
    picks = [(8, 8), (8, 5), (5, 8), (6, 6), (9, 9)]
    positions = in_plane(picks)
    tailored, gradients, _ = selective(maps)
    plain_rf, plain_gradients, _ = pp.make_2d_selective_pulse(
        FLIP, fov=FOV, matrix=16, selective_size=0.08, system=SYSTEM
    )
    plain = pp.make_ptx_pulse(
        np.asarray(plain_rf.signal)[None],
        dwell=float(plain_rf.t[1] - plain_rf.t[0]),
        delay=plain_rf.delay,
        system=SYSTEM,
    )
    achieved = transverse(tailored, gradients, maps, picks, positions)
    assert achieved[0] == pytest.approx(np.sin(FLIP), rel=0.15)
    assert spread(achieved) < spread(
        transverse(plain, plain_gradients, maps, picks, positions)
    )


def test_an_off_resonance_map_that_does_not_match_is_refused():
    with pytest.raises(ValueError, match="off_resonance"):
        selective(np.ones((1, 16, 16)), off_resonance=np.zeros((4, 4)))


# %% spokes


THICKNESS = 5e-3


def spokes(maps, count):
    return pp.make_spokes_pulse(
        FLIP, maps, FOV, THICKNESS, n_spokes=count, system=SYSTEM
    )


def test_spokes_flatten_a_b1_that_one_spoke_cannot():
    """Over the grid the weights were fitted on: the fit trades error between
    points, so a handful of them need not improve."""
    maps = bright_centre(12)
    picks = [(i, j) for i in range(12) for j in range(12)]
    x, y = grid_positions(12)
    positions = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    flat = [
        spread(transverse(*spokes(maps, count)[:2], maps, picks, positions))
        for count in (1, 3)
    ]
    assert flat[1] < 0.7 * flat[0]


def test_a_spokes_pulse_plays_what_its_design_predicts():
    """The built pulse, simulated, against the kernel's own small-tip sum."""
    from pypulseqpp._ext import ptx

    maps = bright_centre(12)
    x, y = grid_positions(12)
    flat = np.column_stack([x.ravel(), y.ravel()])
    reach = 8 / FOV
    pitch = np.arange(-reach / 2, reach / 2 - 1 / FOV + 1e-9, 1 / FOV)
    grid = np.array([(a, b) for a in pitch for b in pitch if a or b], dtype=float)
    where, amounts = ptx.spokes(
        maps.reshape(1, -1),
        flat,
        np.full(flat.shape[0], FLIP),
        np.ones(flat.shape[0]),
        grid,
        3,
    )
    predicted = np.abs(
        (amounts[0][None, :] * np.exp(-2j * np.pi * flat @ where.T)).sum(1)
        * maps.ravel()
    )
    rf, gradients, _ = spokes(maps, 3)
    picks = [(i, j) for i in range(12) for j in range(12)]
    positions = np.column_stack([flat, np.zeros(flat.shape[0])])
    simulated = transverse(rf, gradients, maps, picks, positions)
    assert np.allclose(simulated, np.sin(predicted), rtol=0.01, atol=0.0)


def test_spokes_still_select_a_slice():
    maps = np.ones((2, 8, 8), dtype=complex)
    rf, gradients, _ = spokes(maps, 3)
    picks = [(4, 4), (4, 4)]
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * THICKNESS]])
    inside, outside = transverse(rf, gradients, maps, picks, positions)
    assert outside < 0.1 * inside


def test_a_spokes_block_passes_the_timing_check():
    rf, gradients, rephasers = spokes(loops(4, 8), 3)
    seq = pp.Sequence(SYSTEM)
    seq.add_block(rf, *gradients)
    seq.add_block(*rephasers)
    ok, report = seq.check_timing()
    assert ok, report


def test_spokes_alternate_the_slice_gradient():
    _, gradients, _ = spokes(np.ones((1, 8, 8), dtype=complex), 4)
    z = next(g for g in gradients if g.channel == "z")
    levels = np.asarray(z.waveform)
    plateaus = np.sign(levels[np.abs(levels) > 0.99 * np.abs(levels).max()])
    changes = np.count_nonzero(np.diff(plateaus))
    assert changes == 3


def test_a_spokes_pulse_needs_a_spoke():
    with pytest.raises(ValueError, match="n_spokes"):
        spokes(np.ones((1, 8, 8), dtype=complex), 0)
