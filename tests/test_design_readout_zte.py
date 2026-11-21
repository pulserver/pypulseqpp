"""Continuous-gradient ZTE shells, central sampling gaps and shot rotations."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pypulseqpp as pp
from pypulseqpp import calc_projection_shell, calc_raga_size
from pypulseqpp import sequences as design

FOV = 0.24
MATRIX = 96
#: The default grid of a turn's end ratio, within which a view lands on its direction.
STEP = 1.0 / 128.0


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def hard(system):
    return design.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)


def readout(system, hard, **kwargs):
    kwargs.setdefault("fov", FOV)
    kwargs.setdefault("matrix", MATRIX)
    kwargs.setdefault("n_views", 32)
    return design.ZteReadout(system, hard.rf, **kwargs)


def spokes(module, seq=None):
    """Every acquisition's k-space, shaped ``(views, samples, 3)``."""
    k_traj_adc = (seq or module).calculate_kspace()[0]
    count = int(module.adc.num_samples)
    return np.asarray(k_traj_adc).T.reshape(-1, count, 3)


def plateau(events, sample=0):
    """The gradient vector the three axes of ``events`` hold at ``sample``."""
    return np.array([float(np.asarray(g.waveform)[sample]) for g in events])


def build_shot(system, zte, shot):
    """One shell turned onto where a shot samples, written the docstring's way."""
    seq = pp.Sequence(system)
    turn = pp.make_rotation(Rotation.from_matrix(shot))
    seq.add_block(*zte.g_ramp, turn)
    for view in range(len(zte.directions)):
        seq.add_block(zte.rf, *zte.g_hold[view], turn)
        seq.add_block(zte.adc, *zte.g_read[view], turn)
    return seq


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def test_the_module_lays_out_a_whole_shell(system, hard):
    zte = readout(system, hard)
    views = len(zte.directions)
    assert len(zte.blocks) == 1 + 2 * views
    assert zte.blocks[0] == tuple(zte.g_ramp)
    assert zte.blocks[1] == (zte.rf, *zte.g_hold[0])
    assert zte.blocks[2] == (zte.adc, *zte.g_read[0])
    assert zte.blocks[-1] == (zte.adc, *zte.g_read[-1])


def test_the_readout_is_valid_pulseq(system, hard):
    assert readout(system, hard).check_timing()[0]


def test_every_view_replays_the_same_pulse_and_acquisition(system, hard):
    """Only the gradient differs from one view to the next."""
    zte = readout(system, hard)
    for view in range(len(zte.directions)):
        assert zte.blocks[1 + 2 * view][0] is zte.rf
        assert zte.blocks[2 + 2 * view][0] is zte.adc


def test_labels_ride_the_acquisition_block(system, hard):
    zte = readout(system, hard, labels=("LIN",))
    assert zte.adc_labels in zte.blocks[2]
    assert zte.adc_labels.label == "LIN"


# ----------------------------------------------------------------------
# The shell is written out, not rotated into place
# ----------------------------------------------------------------------


def test_the_shell_carries_no_rotation_of_its_own(system, hard):
    """A shot is a rotation; a view is a waveform."""
    zte = readout(system, hard)
    for index in zte.seq.block_events:
        assert zte.seq.get_block(index).rotation is None


def test_every_view_holds_its_own_direction(system, hard):
    zte = readout(system, hard)
    held = np.array([plateau(hold) for hold in zte.g_hold])
    assert held == pytest.approx(
        zte.gradient_amplitude * zte.directions, abs=STEP * zte.gradient_amplitude
    )


def test_the_shell_plays_the_directions_it_was_given(system, hard):
    zte = readout(system, hard, n_views=12)
    ends = spokes(zte)[:, -1, :]
    played = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    assert played == pytest.approx(zte.directions, abs=2 * STEP)


def test_a_rotated_shell_lands_where_the_shot_rotation_says(system, hard):
    zte = readout(system, hard, n_views=12, n_shots=5)
    shot = zte.shot_rotations[3]
    seq = build_shot(system, zte, shot)
    ends = spokes(zte, seq)[:, -1, :]
    played = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    assert played == pytest.approx(zte.directions @ shot.T, abs=2 * STEP)


def test_an_ordering_whose_step_wanders_is_played_as_it_stands(system, hard):
    """Nothing has to carry one view onto the next, so nothing constrains them."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    index = np.arange(24, dtype=float)
    height = 1.0 - 2.0 * (index + 0.5) / 24
    radius = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    azimuth = index * golden
    phyllotaxis = np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), height)
    )
    zte = readout(system, hard, n_views=None, directions=phyllotaxis)
    ends = spokes(zte)[:, -1, :]
    played = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    assert played == pytest.approx(zte.directions, abs=2 * STEP)


def test_a_turned_shell_plays_the_turned_directions_without_a_rotation(system, hard):
    zte = readout(system, hard, n_views=12, n_shots=5)
    shot = zte.shot_rotations[3]
    shell = zte.shell(shot)
    seq = pp.Sequence(system)
    seq.add_block(*shell.ramp)
    for hold, read in zip(shell.hold, shell.read, strict=True):
        seq.add_block(zte.rf, *hold)
        seq.add_block(zte.adc, *read)
    ends = spokes(zte, seq)[:, -1, :]
    played = ends / np.linalg.norm(ends, axis=1, keepdims=True)
    assert played == pytest.approx(zte.directions @ shot.T, abs=2 * STEP)
    assert shell.directions == pytest.approx(zte.directions @ shot.T, abs=2 * STEP)
    assert seq.check_timing()[0]


def test_every_turn_on_an_axis_is_one_of_a_few_shapes_at_its_own_amplitude(
    system, hard
):
    """A shape is fixed by the ratio of its two ends, on a grid of the turn step."""
    zte = readout(system, hard, n_views=24, n_shots=60)
    shapes = set()
    for shot in zte.shot_rotations:
        for read in zte.shell(shot).read:
            for gradient in read:
                waveform = np.asarray(gradient.waveform, dtype=float)
                peak = np.abs(waveform).max()
                if peak > 0.0:
                    shapes.add((gradient.channel, *np.round(waveform / peak, 9)))
    assert len(shapes) <= 3 * (4 / STEP + 2)


def test_a_coarser_turn_step_shares_fewer_shapes(system, hard):
    def count(step):
        zte = readout(system, hard, n_views=24, n_shots=30, turn_step=step)
        return len(
            {
                (
                    g.channel,
                    *np.round(np.asarray(g.waveform) / np.abs(g.waveform).max(), 9),
                )
                for shot in zte.shot_rotations
                for read in zte.shell(shot).read
                for g in read
                if np.abs(g.waveform).max() > 0
            }
        )

    assert count(1 / 16) < count(1 / 128)


# ----------------------------------------------------------------------
# The gradient never comes down
# ----------------------------------------------------------------------


def test_the_pulse_plays_on_a_gradient_already_at_full_amplitude(system, hard):
    zte = readout(system, hard)
    for ramp, hold in zip(zte.g_ramp, zte.g_hold[0], strict=True):
        assert float(np.asarray(ramp.waveform)[0]) == pytest.approx(0.0)
        assert float(np.asarray(ramp.waveform)[-1]) == pytest.approx(
            float(np.asarray(hold.waveform)[0])
        )
        assert np.ptp(np.asarray(hold.waveform)) == pytest.approx(0.0)


def test_a_view_hands_the_gradient_to_the_next_without_passing_through_zero(
    system, hard
):
    zte = readout(system, hard)
    for view in range(len(zte.directions) - 1):
        held = plateau(zte.g_read[view])
        handover = plateau(zte.g_read[view], -1)
        assert np.linalg.norm(held) == pytest.approx(
            zte.gradient_amplitude, rel=2 * STEP
        )
        assert np.linalg.norm(handover) == pytest.approx(
            zte.gradient_amplitude, rel=2 * STEP
        )
        # It turned onto the next view rather than dropping and climbing again.
        assert handover == pytest.approx(
            plateau(zte.g_hold[view + 1]), abs=STEP * zte.gradient_amplitude
        )


def test_only_the_last_view_ramps_down(system, hard):
    zte = readout(system, hard)
    assert np.linalg.norm(plateau(zte.g_read[-1], -1)) == pytest.approx(0.0, abs=1e-9)
    for view in range(len(zte.directions) - 1):
        assert np.linalg.norm(plateau(zte.g_read[view], -1)) > 0.0


def test_the_shell_stays_within_the_slew_limit(system, hard):
    zte = readout(system, hard)
    gradients = zte.waveforms_and_times()[0]
    raster = system.grad_raster_time
    grid = np.arange(0.0, zte.duration, raster)
    played = np.stack(
        [
            np.interp(
                grid, np.asarray(gradients[axis][0]), np.asarray(gradients[axis][1])
            )
            for axis in range(3)
        ]
    )
    slew = np.linalg.norm(np.diff(played, axis=1), axis=0) / raster
    assert slew.max() <= system.max_slew * 1.001


def test_the_gradient_amplitude_is_one_step_of_k_per_dwell(system, hard):
    zte = readout(system, hard)
    assert zte.gradient_amplitude == pytest.approx(zte.delta_k * zte.bandwidth_hz)


# ----------------------------------------------------------------------
# The spoke
# ----------------------------------------------------------------------


def test_a_spoke_starts_near_the_centre_and_reaches_the_edge(system, hard):
    """It has to arrive at the edge, and it must not run past it."""
    zte = readout(system, hard)
    radius = np.linalg.norm(spokes(zte), axis=-1)
    assert np.all(radius[:, 0] < 5.0 * zte.delta_k)
    # Within one group of samples of the edge: the count is a whole
    # number of the receiver's divisor, so the last sample can fall
    # short of kmax by that many cells rather than by one.
    reach = zte.kmax - system.adc_samples_divisor * zte.delta_k
    assert np.all(radius[:, -1] > reach)
    assert np.all(radius[:, -1] <= zte.kmax)


def test_the_reported_gap_is_the_time_to_the_first_sample(system, hard):
    """It counts the ADC's own delay and the half dwell to the sample centre."""
    for dead_time_s in (None, 40e-6):
        zte = readout(system, hard, dead_time_s=dead_time_s)
        first = np.linalg.norm(spokes(zte)[0, 0])
        assert first == pytest.approx(zte.gap * zte.gradient_amplitude, rel=1e-9)
        assert zte.gap >= float(zte.rf.ringdown_time) + (
            dead_time_s or system.adc_dead_time
        )


def test_a_spoke_is_a_straight_line_out_of_the_centre(system, hard):
    zte = readout(system, hard)
    first = spokes(zte)[0]
    direction = first[-1] / np.linalg.norm(first[-1])
    along = first @ direction
    assert np.linalg.norm(first - np.outer(along, direction)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert np.all(np.diff(along) > 0.0)


def test_the_dropped_samples_are_counted_not_compressed(system, hard):
    zte = readout(system, hard)
    assert zte.n_samples + zte.n_missing == zte.n_nominal
    assert zte.n_missing >= 1
    # What survives is spaced by delta_k, not squeezed into the full extent.
    radius = np.linalg.norm(spokes(zte)[0], axis=-1)
    assert np.diff(radius) == pytest.approx(zte.delta_k, rel=1e-6)


def test_a_faster_readout_loses_fewer_samples(system, hard):
    slow = readout(system, hard, readout_bandwidth_hz=31.25e3)
    fast = readout(system, hard, readout_bandwidth_hz=125e3)
    assert fast.n_missing > slow.n_missing


def test_every_view_traces_the_same_spoke_from_the_centre(system, hard):
    """The trajectory has to restart at each pulse, not accumulate."""
    zte = readout(system, hard)
    radius = np.linalg.norm(spokes(zte), axis=-1)
    assert radius[1] == pytest.approx(radius[0], rel=2 * STEP)


# ----------------------------------------------------------------------
# The ordering: generated or supplied
# ----------------------------------------------------------------------


def test_a_shorter_shell_steps_further_between_views(system, hard):
    """The step is the polar gap at the pole, which ``n_views`` alone fixes."""
    short = readout(system, hard, n_views=32)
    long = readout(system, hard, n_views=128)
    assert short.step_rad == pytest.approx(np.arccos(1.0 - 2.0 / 31.0))
    assert short.step_rad > long.step_rad


def test_it_generates_a_nyquist_matched_shell_when_given_none(system, hard):
    zte = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX, n_shots=89)
    assert len(zte.directions) == -(-int(np.ceil(np.pi * MATRIX**2)) // 89)
    assert zte.shot_rotations.shape == (89, 3, 3)
    assert len(zte.directions) * 89 >= np.pi * MATRIX**2


def test_splitting_into_more_shots_shortens_the_shell(system, hard):
    """``n_shots`` divides a fixed sphere, so it buys segment length."""
    few = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX, n_shots=32)
    many = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX, n_shots=128)
    assert len(many.directions) * 4 == pytest.approx(len(few.directions), abs=4)


def test_the_default_split_matches_the_two_spacings(system, hard):
    """Ring to ring and within a ring agree at the equator: n_shots = pi n_views."""
    zte = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX)
    n_views, n_shots = len(zte.directions), len(zte.shot_rotations)
    assert n_views * n_shots >= np.pi * MATRIX**2
    assert n_shots == pytest.approx(np.pi * n_views, rel=0.05)
    # Which is the same as saying the shell is one spoke per matrix step.
    assert n_views == pytest.approx(MATRIX, rel=0.05)


def test_the_default_sampling_is_isotropic_at_the_equator(system, hard):
    """Polar and azimuthal neighbours land the same distance apart."""
    zte = design.ZteReadout(system, hard.rf, fov=FOV, matrix=MATRIX)
    equator = len(zte.directions) // 2
    heights = np.arccos(np.clip(zte.directions[equator : equator + 2, 2], -1, 1))
    ring_gap = float(np.diff(heights)[0])
    azimuth_gap = 2.0 * np.pi / len(zte.shot_rotations)
    assert ring_gap == pytest.approx(azimuth_gap, rel=0.1)
    assert azimuth_gap == pytest.approx(2.0 / MATRIX, rel=0.1)
    # The shell's own step strides across many rings' worth of azimuth; it is
    # the shots, not the shell, that sample a ring.
    assert zte.step_rad > 5.0 * azimuth_gap


def test_a_generated_shell_runs_pole_to_pole(system, hard):
    """The ends sit on the shot rotation axis, so every shot shares them."""
    zte = readout(system, hard, n_shots=8)
    assert zte.directions[0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert zte.directions[-1] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)
    ends = np.einsum("sij,vj->svi", zte.shot_rotations, zte.directions[[0, -1]])
    assert ends == pytest.approx(np.broadcast_to(zte.directions[[0, -1]], ends.shape))


def test_the_shots_put_even_azimuth_on_every_polar_ring(system, hard):
    """Full coverage: one shell per ring, turned by 2 pi / n_shots."""
    zte = readout(system, hard, n_views=9, n_shots=6)
    sphere = np.einsum("sij,vj->svi", zte.shot_rotations, zte.directions)
    for ring in sphere.transpose(1, 0, 2)[1:-1]:
        azimuth = np.sort(np.arctan2(ring[:, 1], ring[:, 0]))
        assert np.diff(azimuth) == pytest.approx(2.0 * np.pi / 6.0)


def sphere_of(zte):
    """Every direction the shots sample, as sorted unique rows."""
    sphere = np.einsum("sij,vj->svi", zte.shot_rotations, zte.directions)
    return np.unique(np.round(sphere.reshape(-1, 3), 9), axis=0)


def angular_steps(directions):
    return np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))


def test_the_quiet_shell_samples_the_spiral_sphere_in_shorter_steps(system, hard):
    """Same rings, same spokes; each ring's spoke is moved by whole shot turns."""
    spiral = readout(system, hard, n_views=None, n_shots=150)
    quiet = readout(system, hard, n_views=None, n_shots=150, scheme="quiet")
    assert sphere_of(quiet) == pytest.approx(sphere_of(spiral), abs=1e-8)
    # A step crosses about one ring gap, against the spiral's stride.
    ring_gap = 2.0 / MATRIX
    assert np.median(angular_steps(quiet.directions)) < 2.0 * ring_gap
    assert np.median(angular_steps(spiral.directions)) > 5.0 * ring_gap
    # Only the steps off the poles are shorter: the polar gap stays the widest.
    assert quiet.step_rad == pytest.approx(spiral.step_rad)


@pytest.mark.parametrize("scheme", ["spiral", "quiet", "meridian"])
def test_the_reversal_turns_a_generated_shell_onto_itself_backwards(
    system, hard, scheme
):
    zte = readout(system, hard, n_views=33, n_shots=8, scheme=scheme)
    backwards = zte.directions @ np.asarray(zte.reversal).T
    assert backwards == pytest.approx(zte.directions[::-1], abs=1e-9)
    assert np.linalg.det(zte.reversal) == pytest.approx(1.0)


def test_a_supplied_ordering_is_used_as_it_stands(system, hard):
    directions, _ = calc_projection_shell(20, scheme="meridian")
    zte = readout(system, hard, n_views=None, directions=directions)
    assert zte.directions == pytest.approx(directions)
    assert not hasattr(zte.events, "shot_rotations")


def test_a_supplied_ordering_silences_the_generator_arguments(system, hard):
    directions, _ = calc_projection_shell(20, scheme="meridian")
    zte = readout(system, hard, n_views=7, n_shots=5, directions=directions)
    assert zte.directions == pytest.approx(directions)
    assert not hasattr(zte.events, "shot_rotations")


def test_the_directions_are_published_read_only(system, hard):
    zte = readout(system, hard)
    with pytest.raises(ValueError):
        zte.directions[0, 0] = 5.0


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


def test_a_longer_repetition_turns_the_gradient_more_slowly(system, hard):
    """The spare time goes into the slew, not into a wait."""
    tight = readout(system, hard)
    padded = readout(system, hard, tr=tight.tr + 1e-3)
    assert padded.tr == pytest.approx(tight.tr + 1e-3)
    assert pp.calc_duration(padded.g_read[0][0]) > pp.calc_duration(tight.g_read[0][0])
    assert plateau(padded.g_read[0], -1) == pytest.approx(plateau(tight.g_read[0], -1))


def test_a_repetition_shorter_than_the_slew_is_refused(system, hard):
    tight = readout(system, hard)
    with pytest.raises(ValueError, match="TR"):
        readout(system, hard, tr=0.5 * tight.tr)


def test_a_shell_takes_one_ramp_and_a_repetition_per_view(system, hard):
    zte = readout(system, hard, n_views=12)
    ramp = pp.calc_duration(zte.g_ramp[0])
    closing = pp.calc_duration(zte.g_hold[-1][0]) + pp.calc_duration(zte.g_read[-1][0])
    assert zte.duration == pytest.approx(ramp + 11 * zte.tr + closing, abs=1e-9)


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------


def test_a_pulse_too_long_for_the_dwell_is_refused(system):
    long_pulse = design.NonSelectiveExcitation(system, 4.0, duration_s=200e-6)
    with pytest.raises(ValueError, match="brief enough"):
        design.ZteReadout(system, long_pulse.rf, fov=FOV, matrix=MATRIX, n_views=8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fov": 0.0}, "fov"),
        ({"matrix": 1}, "matrix"),
        ({"turn_step": 0.0}, "turn_step"),
        ({"oversampling": 0.5}, "oversampling"),
        ({"readout_bandwidth_hz": 0.0}, "readout_bandwidth_hz"),
        ({"dead_time_s": 0.0}, "dead_time_s"),
    ],
)
def test_an_impossible_request_is_refused(system, hard, kwargs, message):
    with pytest.raises(ValueError, match=message):
        readout(system, hard, **kwargs)


def test_a_single_direction_is_not_a_shell(system, hard):
    with pytest.raises(ValueError, match="at least two views"):
        readout(system, hard, n_views=None, directions=[[1.0, 0.0, 0.0]])


def test_two_views_in_a_row_may_not_coincide(system, hard):
    with pytest.raises(ValueError, match="coincide"):
        readout(
            system,
            hard,
            n_views=None,
            directions=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        )


# ----------------------------------------------------------------------
# Shot ordering


def _azimuths(rotations):
    return np.mod(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]), 2 * np.pi)


def _widest_gap(azimuths):
    """Widest azimuth gap of a set of shots, in units of their mean spacing."""
    ordered = np.sort(azimuths)
    gaps = np.diff(np.append(ordered, ordered[0] + 2 * np.pi))
    return gaps.max() / (2 * np.pi / len(azimuths))


@pytest.mark.parametrize("n_shots", [13, 40, 128, 401])
def test_golden_ordering_plays_every_turn_of_the_set_once(n_shots):
    _, sequential = calc_projection_shell(8, n_shots)
    _, golden = calc_projection_shell(8, n_shots, ordering="golden")
    assert np.sort(_azimuths(golden)) == pytest.approx(np.sort(_azimuths(sequential)))


@pytest.mark.parametrize("n_shots", [40, 128, 401])
@pytest.mark.parametrize("fraction", [8, 4, 2])
def test_every_run_of_golden_shots_spreads_evenly_in_azimuth(n_shots, fraction):
    _, golden = calc_projection_shell(8, n_shots, ordering="golden")
    azimuths = _azimuths(golden)
    run = n_shots // fraction
    widest = max(
        _widest_gap(azimuths[start : start + run]) for start in range(n_shots - run + 1)
    )
    assert widest <= 2.0 + 1e-9


def test_the_first_sequential_shots_leave_a_wedge_of_the_sphere_unsampled():
    _, sequential = calc_projection_shell(8, 128)
    assert _widest_gap(_azimuths(sequential[:32])) > 20


def test_an_unknown_shot_ordering_is_refused():
    with pytest.raises(ValueError, match="ordering"):
        calc_projection_shell(8, 4, ordering="random")


def test_consecutive_gradients_of_a_turned_shell_meet_without_a_step(system, hard):
    zte = readout(system, hard, n_views=24, n_shots=7)
    for shot in zte.shot_rotations:
        shell = zte.shell(shot)
        events = [shell.ramp]
        for hold, read in zip(shell.hold, shell.read, strict=True):
            events += [hold, read]
        for before, after in itertools.pairwise(events):
            for left, right in zip(before, after, strict=True):
                assert float(right.first) == pytest.approx(
                    float(left.last), abs=1e-9 * zte.gradient_amplitude
                )


@pytest.mark.parametrize("ordering", ["golden", "raga"])
@pytest.mark.parametrize("tiny_index", [1, 3, 5])
def test_every_golden_ordering_plays_every_turn_of_the_set_once(ordering, tiny_index):
    n_shots = calc_raga_size(198, tiny_index=tiny_index)
    _, sequential = calc_projection_shell(8, n_shots)
    _, played = calc_projection_shell(
        8, n_shots, ordering=ordering, tiny_index=tiny_index
    )
    assert np.sort(_azimuths(played)) == pytest.approx(np.sort(_azimuths(sequential)))


@pytest.mark.parametrize("tiny_index", [1, 3, 5])
def test_raga_on_its_support_size_is_the_golden_order(tiny_index):
    n_shots = calc_raga_size(198, tiny_index=tiny_index)
    _, golden = calc_projection_shell(
        8, n_shots, ordering="golden", tiny_index=tiny_index
    )
    _, raga = calc_projection_shell(8, n_shots, ordering="raga", tiny_index=tiny_index)
    assert _azimuths(raga) == pytest.approx(_azimuths(golden))


def test_raga_turns_every_shot_by_the_same_step_across_repetitions():
    _, raga = calc_projection_shell(8, 191, ordering="raga", tiny_index=5)
    azimuths = _azimuths(np.concatenate([raga, raga]))
    steps = np.mod(np.diff(azimuths), 2 * np.pi)
    assert np.ptp(steps) == pytest.approx(0.0, abs=1e-9)


def test_raga_refuses_a_shot_count_off_its_support_and_names_the_nearest():
    with pytest.raises(ValueError, match="nearest to 198 is 233"):
        calc_projection_shell(8, 198, ordering="raga")


@pytest.mark.parametrize(
    ("n", "tiny_index", "size"),
    [(198, 1, 233), (198, 5, 191), (1, 1, 1), (150, 1, 144)],
)
def test_the_raga_support_size_is_the_generalized_fibonacci_number_nearest_in_ratio(
    n, tiny_index, size
):
    assert calc_raga_size(n, tiny_index=tiny_index) == size


def test_the_raga_support_size_refuses_a_count_below_one():
    with pytest.raises(ValueError, match="n and tiny_index"):
        calc_raga_size(0)


def test_a_raga_readout_without_a_shot_count_takes_the_nearest_support_size(
    system, hard
):
    zte = readout(system, hard, matrix=64, n_views=8, ordering="raga", tiny_index=5)
    assert len(zte.shot_rotations) == 191


def test_a_larger_tiny_index_turns_consecutive_shots_by_less():
    def median_turn(tiny_index):
        _, played = calc_projection_shell(
            8, 198, ordering="golden", tiny_index=tiny_index
        )
        return np.median(np.abs(np.angle(np.exp(1j * np.diff(_azimuths(played))))))

    assert median_turn(5) < median_turn(3) < median_turn(1)
