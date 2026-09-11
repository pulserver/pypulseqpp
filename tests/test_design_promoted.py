"""Trajectory, wave-gradient, half-passage and extended-trapezoid helpers in the
public namespace, checked against geometry and physics rather than stored output.
"""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp

SYSTEM = pp.Opts.default
FOV = 0.24
MATRIX = 96
KMAX = MATRIX / (2 * FOV)


# %% trajectories


def test_a_radial_spoke_runs_from_minus_to_plus_kmax_through_the_centre():
    spoke = pp.calc_radial_trajectory(FOV, MATRIX, num_points=MATRIX + 1)
    assert spoke[0, 0] == pytest.approx(-KMAX)
    assert spoke[-1, 0] == pytest.approx(KMAX)
    assert np.allclose(spoke[MATRIX // 2], 0.0)
    assert np.allclose(spoke[:, 1], 0.0)


@pytest.mark.parametrize("density", ["constant", "variable", "dual"])
def test_a_spiral_starts_at_the_centre_and_ends_at_kmax(density):
    arm = pp.calc_spiral_trajectory(
        FOV, MATRIX, 4, density=density, outer_design_interleaves=12
    )
    radius = np.hypot(arm[:, 0], arm[:, 1])
    assert np.allclose(arm[0], 0.0)
    assert radius[-1] == pytest.approx(KMAX)
    assert np.all(np.diff(radius) >= 0.0)


def test_a_single_shot_spiral_turns_half_the_matrix_times():
    arm = pp.calc_spiral_trajectory(FOV, MATRIX, 1, num_points=20001)
    turns = np.unwrap(np.arctan2(arm[1:, 1], arm[1:, 0]))
    assert (turns[-1] - turns[0]) / (2 * np.pi) == pytest.approx(MATRIX / 2, rel=1e-3)


def test_a_coarser_outer_pitch_turns_fewer_times():
    def turns(**kwargs):
        arm = pp.calc_spiral_trajectory(FOV, MATRIX, 4, **kwargs)
        angle = np.unwrap(np.arctan2(arm[1:, 1], arm[1:, 0]))
        return angle[-1] - angle[0]

    assert turns(density="variable") < turns(density="constant")


def test_a_dual_density_spiral_needs_its_outer_pitch():
    with pytest.raises(ValueError, match="outer_design_interleaves"):
        pp.calc_spiral_trajectory(FOV, MATRIX, 4, density="dual")


@pytest.mark.parametrize("petals", [1, 4, 7])
def test_a_rosette_returns_to_the_centre_between_petals(petals):
    num_points = 100 * petals + 1
    path = pp.calc_rosette_trajectory(FOV, MATRIX, petals=petals, num_points=num_points)
    radius = np.hypot(path[:, 0], path[:, 1])
    crossings = radius[:: (num_points - 1) // petals]
    assert crossings.size == petals + 1
    assert np.allclose(crossings, 0.0, atol=1e-9 * KMAX)
    assert radius.max() == pytest.approx(KMAX, rel=1e-3)


def test_an_anisotropic_field_of_view_is_refused():
    with pytest.raises(ValueError, match="isotropic"):
        pp.calc_spiral_trajectory((0.24, 0.20), MATRIX, 4)


def test_a_trajectory_is_a_path_traj_to_grad_can_trace():
    arm = pp.calc_spiral_trajectory(FOV, MATRIX, 8)
    gradient, _ = pp.traj_to_grad(arm.T, system=SYSTEM)
    assert np.isfinite(gradient).all()
    assert np.abs(gradient).max() <= SYSTEM.max_grad * (1 + 1e-6)


# %% wave gradients


def test_both_wave_axes_leave_no_net_area():
    sine, cosine = pp.make_wave_gradients(4e-3, 5, 8e-3, system=SYSTEM)
    for event in (sine, cosine):
        waveform = np.asarray(event.waveform)
        assert abs(waveform.sum()) <= 1e-9 * np.abs(waveform).max() * waveform.size
        assert event.first == 0.0
        assert event.last == 0.0


def test_the_sine_leads_the_cosine_by_a_quarter_period():
    """Inside the envelope, each crosses zero where the other peaks."""
    cycles = 4
    sine, cosine = pp.make_wave_gradients(4e-3, cycles, 4e-3, system=SYSTEM)
    period = np.asarray(sine.waveform).size // cycles
    crest, node = period + period // 4, period
    assert abs(cosine.waveform[crest]) < 0.1 * abs(sine.waveform[crest])
    assert abs(sine.waveform[node]) < 0.1 * abs(cosine.waveform[node])


def test_a_gentle_wave_is_built_at_the_amplitude_asked_for():
    *_, built = pp.make_wave_gradients(
        8e-3, 2, 2e-3, return_amplitude=True, system=SYSTEM
    )
    assert built == pytest.approx(2e-3)


def test_a_fast_wave_is_capped_at_the_slew_limit():
    sine, cosine, built = pp.make_wave_gradients(
        2e-3, 8, 40e-3, return_amplitude=True, system=SYSTEM
    )
    assert built < 40e-3
    for event in (sine, cosine):
        # Samples sit at raster centres, so the steps in from and out to zero
        # cross half a raster.
        waveform = np.asarray(event.waveform)
        steps = np.concatenate(
            [[2 * waveform[0]], np.diff(waveform), [-2 * waveform[-1]]]
        )
        assert np.abs(steps).max() / SYSTEM.grad_raster_time <= SYSTEM.max_slew * (
            1 + 1e-9
        )


def test_a_single_axis_wave_leaves_the_other_channel_out():
    sine, cosine = pp.make_wave_gradients(4e-3, 4, 4e-3, cosine_channel=None)
    assert sine.channel == "y"
    assert cosine is None


def test_a_wave_played_twice_on_one_channel_is_refused():
    with pytest.raises(ValueError, match="channels of their own"):
        pp.make_wave_gradients(4e-3, 4, 4e-3, sine_channel="z")


def test_a_wave_on_no_channel_is_refused():
    with pytest.raises(ValueError, match="sine channel, a cosine channel"):
        pp.make_wave_gradients(4e-3, 4, 4e-3, sine_channel=None, cosine_channel=None)


# %% half passages


def test_the_half_passages_mirror_each_other():
    down, up = pp.make_half_passages(4e-3, system=SYSTEM)
    assert np.allclose(np.asarray(down.signal), np.conj(np.asarray(up.signal))[::-1])


def test_each_half_passage_lasts_the_duration_asked_for():
    down, up = pp.make_half_passages(4e-3, dwell=10e-6, system=SYSTEM)
    for pulse in (down, up):
        assert np.asarray(pulse.signal).size * 10e-6 == pytest.approx(4e-3)


def test_the_half_passage_that_tips_down_ends_on_resonance():
    """The sweep runs from far off resonance to on resonance, so the
    instantaneous frequency is smallest at the end."""
    down, _ = pp.make_half_passages(4e-3, system=SYSTEM)
    frequency = np.abs(np.diff(np.unwrap(np.angle(np.asarray(down.signal)))))
    assert frequency[-1] < frequency[0]


# %% extended trapezoid of an area


def test_an_extended_trapezoid_of_an_area_joins_the_amplitudes_it_is_given():
    """The readout modules bridge onto and off their plateaus with it."""
    grad, times, amplitudes = pp.make_extended_trapezoid_area(
        area=500.0, channel="x", grad_start=0.0, grad_end=2e5, system=SYSTEM
    )
    area = np.sum(0.5 * (amplitudes[1:] + amplitudes[:-1]) * np.diff(times))
    assert amplitudes[0] == 0.0
    assert amplitudes[-1] == pytest.approx(2e5)
    assert area == pytest.approx(500.0, rel=1e-6)
    assert grad.first == 0.0


def test_the_played_wave_never_exceeds_the_gradient_limit():
    # A steep slew limit leaves max_grad as the cap that binds; the balancing
    # offset lifts the cosine's peak above the sinusoid amplitude.
    system = pp.Opts(max_grad=20, grad_unit="mT/m", max_slew=10000, slew_unit="T/m/s")
    sine, cosine = pp.make_wave_gradients(3e-3, 4, 1.0, system=system)
    for event in (sine, cosine):
        assert np.abs(np.asarray(event.waveform)).max() <= system.max_grad * (1 + 1e-9)
