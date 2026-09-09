"""A prescription applied to a designed sequence.

Where the volume sits, which way it faces and how big it is are three things
done to a sequence that was designed without knowing any of them. A scale is a
multiplication on a gradient's amplitude, a rotation is four numbers attached
to a block, and a shift is a phase -- ``dr . k`` -- written onto every pulse
and every readout that follows.

The invariants below are the reference toolbox's, asked of this API. Two
answers differ from it on purpose and say so where they are asserted: a
rotation is never baked into new waveforms here, and a shift's residual phase
is held against ``dr . k(t)`` itself rather than against a wrapped ``k``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pypulseqpp as pp

TURN = 2.0 * math.pi


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=200,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
    )


def rot(axis: str, degrees: float) -> np.ndarray:
    return Rotation.from_euler(axis, degrees, degrees=True).as_matrix()


def trap(axis, area, system, duration=2e-3, **kwargs):
    return pp.make_trapezoid(
        axis, area=area, duration=duration, system=system, **kwargs
    )


def flat(axis, flat_area, flat_time, system):
    """A gradient that does not move while a pulse or a readout runs."""
    return pp.make_trapezoid(
        axis, flat_area=flat_area, flat_time=flat_time, system=system
    )


def sinc(system, delay=0.0):
    return pp.make_sinc_pulse(
        math.pi / 6,
        duration=1e-3,
        slice_thickness=5e-3,
        delay=delay,
        use="excitation",
        system=system,
    )


def one_block(system, *events):
    seq = pp.Sequence(system)
    seq.add_block(*events)
    return seq


def played_areas(seq) -> np.ndarray:
    """The area each axis is driven through, in 1/m, rotations included.

    Read off the waveforms rather than the stored rows, because a rotation
    here moves what a block plays without moving the row it plays it from.
    """
    waveforms = seq.waveforms_and_times()[0]
    return np.array(
        [
            float(np.trapezoid(channel[1], channel[0])) if channel.size else 0.0
            for channel in waveforms
        ]
    )


# %% the prescription itself


def test_a_transform_with_nothing_to_do_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        pp.TransformFOV()


def test_a_rotation_is_kept_as_a_quaternion():
    """Four numbers, which is what a block carries and what a scanner is told."""
    turn = Rotation.from_euler("z", 45, degrees=True)
    prescription = pp.TransformFOV(rotation=turn.as_matrix())
    np.testing.assert_allclose(
        prescription.quaternion, turn.as_quat(canonical=True, scalar_first=True)
    )
    assert prescription.translation is None
    assert prescription.scale is None


def test_a_rotation_may_arrive_as_a_matrix_or_as_a_rotation():
    turn = Rotation.from_euler("y", 30, degrees=True)
    np.testing.assert_allclose(
        pp.TransformFOV(rotation=turn).quaternion,
        pp.TransformFOV(rotation=turn.as_matrix()).quaternion,
    )


def test_a_rotation_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        pp.TransformFOV(rotation=np.eye(4))


def test_a_translation_is_kept_in_metres():
    prescription = pp.TransformFOV(translation=(0.01, 0.02, 0.03))
    assert prescription.translation == (0.01, 0.02, 0.03)
    assert prescription.quaternion is None


def test_a_scale_is_kept_per_axis():
    assert pp.TransformFOV(scale=(2.0, 0.5, 1.0)).scale == (2.0, 0.5, 1.0)


@pytest.mark.parametrize("name", ["translation", "scale"])
def test_an_offset_that_is_not_three_numbers_is_refused(name):
    with pytest.raises(ValueError, match="three numbers"):
        pp.TransformFOV(**{name: (1.0, 2.0)})


@pytest.mark.parametrize(
    "other", [{"rotation": np.eye(3)}, {"translation": (1.0, 1.0, 1.0)}]
)
def test_a_homogeneous_matrix_says_it_all_by_itself(other):
    with pytest.raises(ValueError, match="not both"):
        pp.TransformFOV(transform=np.eye(4), **other)


def test_a_homogeneous_matrix_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        pp.TransformFOV(transform=np.eye(3))


def test_baking_the_rotation_into_the_waveforms_is_refused():
    """One waveform per orientation, where an annotation costs four numbers.

    This is where the API parts from the reference toolbox, which rotates the
    gradients themselves unless asked not to.
    """
    with pytest.raises(NotImplementedError, match="annotation"):
        pp.TransformFOV(rotation=np.eye(3), use_rotation_extension=False)


def test_the_reference_toolbox_spelling_is_the_same_method():
    assert pp.TransformFOV.apply_to_seq is pp.TransformFOV.apply_to_sequence


# %% scale


def test_scaling_an_axis_scales_what_it_draws(system):
    seq = one_block(system, trap("x", 1000, system))
    moved = pp.TransformFOV(scale=(0.5, 1.0, 1.0)).apply_to_sequence(seq)
    assert played_areas(moved)[0] == pytest.approx(0.5 * played_areas(seq)[0], rel=1e-9)


def test_scaling_one_axis_leaves_the_others_where_they_were(system):
    seq = one_block(
        system,
        trap("x", 1000, system),
        trap("y", 2000, system),
        trap("z", 3000, system),
    )
    before = played_areas(seq)
    after = played_areas(pp.TransformFOV(scale=(1.0, 1.0, 0.5)).apply_to_sequence(seq))
    np.testing.assert_allclose(after, before * (1.0, 1.0, 0.5), rtol=1e-9)


def test_a_negative_scale_reverses_the_axis(system):
    seq = one_block(system, trap("x", 1000, system))
    after = pp.TransformFOV(scale=(-1.0, 1.0, 1.0)).apply_to_sequence(seq)
    assert played_areas(after)[0] == pytest.approx(-played_areas(seq)[0], rel=1e-9)


def test_a_zero_scale_silences_the_axis(system):
    seq = one_block(system, trap("x", 1000, system))
    after = pp.TransformFOV(scale=(0.0, 1.0, 1.0)).apply_to_sequence(seq)
    assert played_areas(after)[0] == pytest.approx(0.0, abs=1e-9)


def test_the_original_is_left_alone_unless_asked(system):
    seq = one_block(system, trap("x", 1000, system))
    before = played_areas(seq)
    pp.TransformFOV(scale=(0.5, 1.0, 1.0)).apply_to_sequence(seq)
    np.testing.assert_allclose(played_areas(seq), before, rtol=1e-12)


def test_in_place_transforms_the_sequence_it_was_given(system):
    seq = one_block(system, trap("x", 1000, system))
    before = played_areas(seq)
    same = pp.TransformFOV(scale=(0.5, 1.0, 1.0)).apply_to_sequence(seq, in_place=True)
    assert same is seq
    assert played_areas(seq)[0] == pytest.approx(0.5 * before[0], rel=1e-9)


# %% rotation


def test_the_identity_leaves_the_axes_where_they_were(system):
    seq = one_block(system, trap("x", 1000, system), trap("y", 2000, system))
    before = played_areas(seq)
    after = played_areas(pp.TransformFOV(rotation=np.eye(3)).apply_to_sequence(seq))
    np.testing.assert_allclose(after, before, atol=1e-9)


def test_a_quarter_turn_about_z_sends_x_onto_y(system):
    seq = one_block(system, trap("x", 1000, system))
    after = played_areas(pp.TransformFOV(rotation=rot("z", 90)).apply_to_sequence(seq))
    np.testing.assert_allclose(after, (0.0, played_areas(seq)[0], 0.0), atol=1e-6)


def test_a_quarter_turn_about_x_sends_y_onto_z(system):
    seq = one_block(system, trap("y", 2000, system))
    after = played_areas(pp.TransformFOV(rotation=rot("x", 90)).apply_to_sequence(seq))
    np.testing.assert_allclose(after, (0.0, 0.0, played_areas(seq)[1]), atol=1e-6)


def test_an_eighth_turn_splits_the_area_between_two_axes(system):
    seq = one_block(system, trap("x", 1000, system))
    drawn = played_areas(seq)[0]
    after = played_areas(pp.TransformFOV(rotation=rot("z", 45)).apply_to_sequence(seq))
    np.testing.assert_allclose(
        after[:2], (drawn / math.sqrt(2), drawn / math.sqrt(2)), rtol=1e-6
    )


def test_a_half_turn_inverts_the_axis(system):
    seq = one_block(system, trap("x", 4000, system))
    after = played_areas(pp.TransformFOV(rotation=rot("z", 180)).apply_to_sequence(seq))
    assert after[0] == pytest.approx(-played_areas(seq)[0], rel=1e-6)


def test_turning_the_volume_does_not_change_how_far_k_travels(system):
    """A rotation moves where the area is drawn, not how much of it there is."""
    seq = one_block(
        system,
        *(
            trap(axis, area, system, max_slew=system.max_slew / math.sqrt(3))
            for axis, area in zip("xyz", (1000, 2000, 1500), strict=True)
        ),
    )
    turn = rot("y", 20) @ rot("z", 30)
    after = played_areas(pp.TransformFOV(rotation=turn).apply_to_sequence(seq))
    assert np.linalg.norm(after) == pytest.approx(
        np.linalg.norm(played_areas(seq)), rel=1e-6
    )


def test_what_each_axis_draws_is_the_area_vector_turned(system):
    seq = one_block(
        system,
        *(
            trap(axis, area, system, max_slew=system.max_slew / math.sqrt(3))
            for axis, area in zip("xyz", (1000, -500, 2000), strict=True)
        ),
    )
    turn = rot("x", 53) @ rot("z", 37)
    after = played_areas(pp.TransformFOV(rotation=turn).apply_to_sequence(seq))
    np.testing.assert_allclose(after, turn @ played_areas(seq), rtol=1e-6)


def test_three_turns_are_one_turn(system):
    seq = one_block(
        system,
        *(
            trap(
                axis,
                area,
                system,
                duration=3e-3,
                max_slew=system.max_slew / math.sqrt(3),
            )
            for axis, area in zip("xyz", (1500, -800, 2200), strict=True)
        ),
    )
    turn = rot("z", 17) @ rot("y", 43) @ rot("x", -29)
    after = played_areas(pp.TransformFOV(rotation=turn).apply_to_sequence(seq))
    np.testing.assert_allclose(after, turn @ played_areas(seq), rtol=1e-6)


def test_a_second_prescription_turns_what_the_first_left(system):
    """Two prescriptions applied in turn are the product of the two."""
    seq = one_block(system, trap("x", 1000, system))
    first, second = rot("z", 30), rot("x", 45)
    at_once = pp.TransformFOV(rotation=second @ first).apply_to_sequence(seq)
    one_then_the_other = pp.TransformFOV(rotation=second).apply_to_sequence(
        pp.TransformFOV(rotation=first).apply_to_sequence(seq)
    )
    np.testing.assert_allclose(
        played_areas(one_then_the_other), played_areas(at_once), rtol=1e-6
    )


def test_a_rotation_is_an_annotation_and_not_a_new_waveform(system):
    """The stored rows are the ones the design wrote; the block carries four
    numbers saying how they are played."""
    seq = one_block(system, trap("x", 1000, system))
    turned = pp.TransformFOV(rotation=rot("z", 90)).apply_to_sequence(seq)
    stored, played = seq.get_block(1).gx, turned.get_block(1).gx
    assert float(played.amplitude) == float(stored.amplitude)
    for field in ("rise_time", "flat_time", "fall_time"):
        # The copy travels through the binary form, where a time is an
        # integer number of picoseconds.
        assert float(getattr(played, field)) == pytest.approx(
            float(getattr(stored, field)), abs=1e-12
        )
    assert seq.get_block(1).rotation is None
    np.testing.assert_allclose(
        turned.get_block(1).rotation.quaternion,
        Rotation.from_matrix(rot("z", 90)).as_quat(canonical=True, scalar_first=True),
    )


def test_a_homogeneous_identity_changes_nothing(system):
    seq = one_block(system, trap("x", 1000, system), trap("y", 2000, system))
    after = pp.TransformFOV(transform=np.eye(4)).apply_to_sequence(seq)
    np.testing.assert_allclose(played_areas(after), played_areas(seq), atol=1e-9)


# %% translation


def test_a_pulse_under_a_steady_gradient_is_moved_by_a_frequency(system):
    """A slab under a constant gradient moves by gamma * G * dr, and the
    envelope is untouched."""
    gx = flat("x", 5000, 1e-3, system)
    seq = one_block(system, sinc(system, delay=gx.rise_time), gx)
    moved = pp.TransformFOV(translation=(0.01, 0.0, 0.0)).apply_to_sequence(seq)
    assert float(moved.get_block(1).rf.freq_offset) == pytest.approx(
        float(seq.get_block(1).rf.freq_offset) + 0.01 * float(gx.amplitude), rel=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(moved.get_block(1).rf.signal),
        np.asarray(seq.get_block(1).rf.signal),
        atol=1e-12,
    )


def test_a_readout_under_a_steady_gradient_is_moved_by_a_frequency(system):
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=system)
    seq = one_block(system, adc, gx)
    moved = pp.TransformFOV(translation=(0.02, 0.0, 0.0)).apply_to_sequence(seq)
    assert float(moved.get_block(1).adc.freq_offset) == pytest.approx(
        0.02 * float(gx.amplitude), rel=1e-9
    )


def test_a_steady_gradient_needs_no_phase_shape(system):
    """Two numbers say the whole of it, which is why a Cartesian readout
    carries no per-sample phase at all."""
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=system)
    seq = one_block(system, adc, gx)
    moved = pp.TransformFOV(translation=(0.02, 0.0, 0.0)).apply_to_sequence(seq)
    modulation = np.asarray(moved.get_block(1).adc.phase_modulation)
    assert modulation.size == 0 or np.allclose(modulation, 0.0, atol=1e-12)


def test_a_shift_of_nothing_moves_nothing(system):
    gx = flat("x", 5000, 1e-3, system)
    adc = pp.make_adc(32, duration=1e-3, delay=gx.rise_time, system=system)
    seq = one_block(system, sinc(system, delay=gx.rise_time), adc, gx)
    moved = pp.TransformFOV(translation=(0.0, 0.0, 0.0)).apply_to_sequence(seq)
    for event in ("rf", "adc"):
        assert float(getattr(moved.get_block(1), event).freq_offset) == pytest.approx(
            float(getattr(seq.get_block(1), event).freq_offset), abs=1e-9
        )
        assert float(getattr(moved.get_block(1), event).phase_offset) == pytest.approx(
            float(getattr(seq.get_block(1), event).phase_offset), abs=1e-9
        )


def test_a_shift_leaves_the_gradients_alone(system):
    """It is a phase written onto the pulses and the readouts, nothing else."""
    seq = one_block(system, trap("x", 1000, system, duration=1e-3))
    moved = pp.TransformFOV(translation=(0.01, 0.0, 0.0)).apply_to_sequence(seq)
    np.testing.assert_allclose(played_areas(moved), played_areas(seq), rtol=1e-12)


def test_a_pulse_with_no_gradient_under_it_is_not_moved(system):
    """Nothing encodes position there, so there is no position to move."""
    rf = pp.make_block_pulse(
        math.pi / 2, duration=1e-3, use="excitation", system=system
    )
    seq = one_block(system, rf)
    moved = pp.TransformFOV(translation=(0.01, 0.02, 0.03)).apply_to_sequence(seq)
    assert float(moved.get_block(1).rf.freq_offset) == pytest.approx(
        float(rf.freq_offset), abs=1e-9
    )
    assert float(moved.get_block(1).rf.phase_offset) == pytest.approx(
        float(rf.phase_offset), abs=1e-9
    )


def test_every_readout_of_a_repeated_shot_is_moved(system):
    gx = flat("x", 1000, 1e-3, system)
    adc = pp.make_adc(16, duration=1e-3, delay=gx.rise_time, system=system)
    seq = pp.Sequence(system)
    for _ in range(3):
        seq.add_block(gx, adc)
    moved = pp.TransformFOV(translation=(0.01, 0.0, 0.0)).apply_to_sequence(seq)
    for block in range(1, 4):
        assert abs(float(moved.get_block(block).adc.freq_offset)) > 0.0


def receive_phase(block, samples=None):
    """The phase a readout is demodulated with, in turns, sample by sample.

    Three pieces say it: a constant, a frequency read from the readout's own
    start, and -- only where the gradient moves under the readout -- a phase
    per sample.
    """
    count = int(block.adc.num_samples)
    when = float(block.adc.dwell) * (np.arange(count) + 0.5)
    phase = float(block.adc.phase_offset) + TURN * float(block.adc.freq_offset) * when
    modulation = np.asarray(block.adc.phase_modulation)
    if modulation.size:
        phase = phase + modulation
    return phase / TURN


def wrapped(turns):
    return (np.asarray(turns) + 0.5) % 1.0 - 0.5


def test_the_phase_a_readout_is_demodulated_with_is_the_shift_along_its_trajectory(
    system,
):
    """``dr . k(t)`` at every sample, counted from the excitation.

    Absolutely, not up to a constant: the reference toolbox counts from an
    origin of its own and so answers this plus a global phase per readout,
    which is a choice rather than a difference. What is asked here is the
    stronger of the two, because the trajectory a reconstructor is handed is
    counted from the excitation.
    """
    shift = 0.011
    gx_pre = trap("x", -2500, system, duration=2e-3)
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=system)
    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(gx_pre)
    seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    k = np.asarray(seq.calculate_kspace()[0])[0]
    np.testing.assert_allclose(
        wrapped(receive_phase(moved.get_block(3)) - shift * k), 0.0, atol=1e-9
    )


def test_a_gradient_that_moves_under_the_readout_needs_a_phase_per_sample(system):
    """A frequency and a constant cannot follow a ramp, so what is left over
    is written into the readout's own phase modulation."""
    shift = 0.007
    gx = trap("x", 3000, system, duration=2e-3)
    adc = pp.make_adc(
        128,
        duration=float(gx.rise_time) + float(gx.flat_time),
        delay=0.0,
        system=system,
    )
    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    assert np.asarray(moved.get_block(2).adc.phase_modulation).size == 128
    k = np.asarray(seq.calculate_kspace()[0])[0]
    # The phase modulation is a shape, and a copy carries shape samples as
    # float32, so the whole readout is held to a float32 of a turn.
    np.testing.assert_allclose(
        wrapped(receive_phase(moved.get_block(2)) - shift * k), 0.0, atol=5e-6
    )


def test_a_pulse_under_a_gradient_that_moves_carries_a_phase_shape(system):
    """The same for the transmit side, referenced to the centre the pulse
    records so that what the pulse does is untouched."""
    shift = 0.009
    gz = trap("z", 3000, system, duration=2e-3)
    rf = pp.make_sinc_pulse(
        math.pi / 6,
        duration=1e-3,
        slice_thickness=5e-3,
        use="excitation",
        system=system,
    )
    seq = one_block(system, rf, gz)
    moved = pp.TransformFOV(translation=(0.0, 0.0, shift)).apply_to_sequence(seq)
    before = np.asarray(seq.get_block(1).rf.signal)
    after = np.asarray(moved.get_block(1).rf.signal)
    np.testing.assert_allclose(np.abs(after), np.abs(before), rtol=1e-6)
    assert np.ptp(np.angle(after) - np.angle(before)) > 1e-3


def test_a_readout_that_runs_off_the_flat_top_is_not_taken_for_a_steady_one(system):
    """Whether a gradient moves is asked of the whole window.

    A readout that begins on a flat top and ends on the ramp down is steady
    across its first half and not across its second, so a question asked of
    the first half alone answers that two numbers will do -- and the ramp's
    phase is then never written anywhere.
    """
    shift = 0.011
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(
        128,
        duration=float(gx.flat_time) + float(gx.fall_time),
        delay=float(gx.rise_time),
        system=system,
    )
    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    assert np.asarray(moved.get_block(2).adc.phase_modulation).size == 128
    k = np.asarray(seq.calculate_kspace()[0])[0]
    np.testing.assert_allclose(
        wrapped(receive_phase(moved.get_block(2)) - shift * k), 0.0, atol=5e-6
    )


def test_a_pulse_whose_gradient_ramps_under_its_end_is_not_taken_for_a_steady_one(
    system,
):
    """The transmit side of the same question, asked of the whole pulse."""
    shift = 0.009
    gz = flat("z", 3000, 1e-3, system)
    rf = pp.make_sinc_pulse(
        math.pi / 6,
        duration=float(gz.flat_time) + float(gz.fall_time),
        slice_thickness=5e-3,
        delay=float(gz.rise_time),
        use="excitation",
        system=system,
    )
    seq = one_block(system, rf, gz)
    moved = pp.TransformFOV(translation=(0.0, 0.0, shift)).apply_to_sequence(seq)
    added = np.angle(np.asarray(moved.get_block(1).rf.signal)) - np.angle(
        np.asarray(seq.get_block(1).rf.signal)
    )
    assert np.ptp(added) / TURN > 1e-3


def test_a_readout_is_referenced_to_its_echo_and_not_to_its_window(system):
    """Where the frequency and the phase are anchored.

    The two are the same instant only for a readout symmetric about the
    centre of k-space. Here the prewinder undoes a fraction of the readout,
    so the echo falls early -- on the ramp, where the gradient is still
    moving -- and anchoring at the middle of the window would leave the
    centre of k-space turns away from where the shift asks for it.

    What is held is that the scalars *alone* place the echo: a reader that
    drops the phase profile still gets the centre of k-space right, and the
    profile carries only the curvature around it.
    """
    shift = 0.011
    gx = flat("x", 5000, 1.4e-3, system)
    samples = 192
    adc = pp.make_adc(
        samples,
        duration=float(gx.rise_time) + float(gx.flat_time),
        delay=0.0,
        system=system,
    )
    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(trap("x", -200, system, duration=1e-3))
    seq.add_block(gx, adc)

    k = np.asarray(seq.calculate_kspace()[0])[0]
    echo = int(np.argmin(np.abs(k)))
    # The echo is early, and on the ramp rather than the flat top.
    assert echo < samples // 4
    assert echo * float(adc.dwell) < float(gx.rise_time)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    block = moved.get_block(3)
    when = float(block.adc.dwell) * (np.arange(samples) + 0.5)
    scalars = (
        float(block.adc.phase_offset) + TURN * float(block.adc.freq_offset) * when
    ) / TURN
    assert abs(float(wrapped(scalars[echo] - shift * k[echo]))) < 1e-3
    # Only because it is the echo: the same two numbers are turns out by the
    # end of the readout, and that is what the profile is for.
    assert np.abs(wrapped(scalars - shift * k)).max() > 0.1


def test_an_exempt_stretch_still_counts_towards_what_follows_it(system):
    """A module that placed itself keeps its phase; its area still happened.

    The gradients an exempt block plays move k like any others, so a shift
    walk that skipped the stretch would leave every phase after it short by
    what the stretch swept.
    """
    shift = 0.011
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=float(gx.rise_time), system=system)
    spoiler = trap("x", 1234, system, duration=1e-3)

    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(spoiler, pp.make_label("NOPOS", "SET", 1))
    seq.add_block(pp.make_delay(1e-4), pp.make_label("NOPOS", "SET", 0))
    seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    k = np.asarray(seq.calculate_kspace()[0])[0]
    np.testing.assert_allclose(
        wrapped(receive_phase(moved.get_block(4)) - shift * k), 0.0, atol=1e-9
    )


def test_an_exempt_block_keeps_the_phase_it_was_designed_with(system):
    shift = 0.011
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=float(gx.rise_time), system=system)

    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(gx, adc, pp.make_label("NOPOS", "SET", 1))
    seq.add_block(pp.make_delay(1e-4), pp.make_label("NOPOS", "SET", 0))
    seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    assert float(moved.get_block(2).adc.freq_offset) == pytest.approx(
        float(seq.get_block(2).adc.freq_offset), abs=1e-9
    )
    assert abs(float(moved.get_block(4).adc.freq_offset)) > 0.0


def acquired_against_its_excitation(moved, excitation, readout, shift, k):
    """A readout's phase measured from the phase its own excitation was given.

    The physical quantity. Which origin the two are counted from is a choice
    -- shift both and nothing observable moves -- so only their difference
    means anything, and it has to come out as ``dr . k`` with ``k`` counted
    from that excitation.
    """
    block = moved.get_block(readout)
    return wrapped(
        receive_phase(block)
        - float(moved.get_block(excitation).rf.phase_offset) / TURN
        - float(shift) * k
    )


def test_every_playout_of_a_readout_shares_one_reference(system):
    """The echo belongs to the readout, not to the playout.

    Not every playout passes the centre of k-space. Here each shot has its
    own prewinder, so each crosses the readout axis at its own instant and a
    pivot chosen per playout would be five different instants -- and five
    phase profiles registered for one readout. The playout that comes
    nearest the centre fixes the instant for all of them, so the table
    shares one shape, and every shot is still acquired at the phase the
    shift asks for.
    """
    shift = 0.011
    gx = flat("x", 5000, 1.4e-3, system)
    samples = 192
    adc = pp.make_adc(
        samples,
        duration=float(gx.rise_time) + float(gx.flat_time),
        delay=0.0,
        system=system,
    )
    prewinders = (-200, -600, -900, -1400, -1900)

    seq = pp.Sequence(system)
    for area in prewinders:
        seq.add_block(sinc(system))
        seq.add_block(trap("x", area, system, duration=1e-3))
        seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    k = np.asarray(seq.calculate_kspace()[0])[0]

    # Each shot's own nearest sample is somewhere else.
    nearest = [
        int(np.argmin(np.abs(k[shot * samples : (shot + 1) * samples])))
        for shot in range(len(prewinders))
    ]
    assert len(set(nearest)) == len(prewinders)

    # One profile between them, and every shot right.
    profiles = {
        np.asarray(moved.get_block(3 + 3 * shot).adc.phase_modulation).tobytes()
        for shot in range(len(prewinders))
    }
    assert len(profiles) == 1
    for shot in range(len(prewinders)):
        np.testing.assert_allclose(
            acquired_against_its_excitation(
                moved,
                1 + 3 * shot,
                3 + 3 * shot,
                shift,
                k[shot * samples : (shot + 1) * samples],
            ),
            0.0,
            atol=5e-6,
        )


def test_a_readout_is_acquired_against_the_phase_its_excitation_was_given(system):
    """Two repetitions of one thing are acquired at one phase.

    A readout and its excitation are counted from the same place, so the
    place itself is a global phase and cancels. Count them from different
    places -- restart at the excitation for one and not the other -- and two
    repetitions of the same thing come out different.
    """
    shift = 0.013
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=float(gx.rise_time), system=system)
    seq = pp.Sequence(system)
    for _ in range(3):
        seq.add_block(sinc(system))
        seq.add_block(trap("x", -2500, system, duration=2e-3))
        seq.add_block(gx, adc)

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    k = np.asarray(seq.calculate_kspace()[0])[0]
    for shot in range(3):
        np.testing.assert_allclose(
            acquired_against_its_excitation(
                moved, 1 + 3 * shot, 3 + 3 * shot, shift, k[shot * 64 : (shot + 1) * 64]
            ),
            0.0,
            atol=1e-9,
        )


def test_a_readout_of_many_corners_is_walked_once(system):
    """An arbitrary path costs its samples plus its corners, not their product.

    A spiral samples every raster tick of a waveform that turns at every one
    of them, so a sweep restarted at the first corner for each sample is
    quadratic in the readout. What is held here is the answer; the cost is
    `benchmarks/`.
    """
    shift, samples = 0.011, 4000
    raster = system.grad_raster_time
    along = np.arange(samples) * raster
    span = along[-1]
    turning = 2.0 * math.pi * 8.0 * along / span
    radius = along / span
    windowed = np.sin(math.pi * along / span) ** 2
    gx = np.gradient(radius * np.cos(turning), raster) * windowed
    gy = np.gradient(radius * np.sin(turning), raster) * windowed
    scale = min(
        0.9 * system.max_grad / max(np.abs(gx).max(), np.abs(gy).max()),
        0.9
        * system.max_slew
        / (max(np.abs(np.diff(gx)).max(), np.abs(np.diff(gy)).max()) / raster),
    )

    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(
        pp.make_arbitrary_grad("x", gx * scale, system=system, first=0.0, last=0.0),
        pp.make_arbitrary_grad("y", gy * scale, system=system, first=0.0, last=0.0),
        pp.make_adc(samples, dwell=raster, delay=0.0, system=system),
    )

    moved = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(seq)
    k = np.asarray(seq.calculate_kspace()[0])[0]
    np.testing.assert_allclose(
        acquired_against_its_excitation(moved, 1, 2, shift, k), 0.0, atol=1e-4
    )


# %% the three together


def test_a_turn_and_a_shift_both_land(system):
    gx = flat("x", 5000, 1e-3, system)
    seq = one_block(system, sinc(system, delay=gx.rise_time), gx)
    moved = pp.TransformFOV(
        rotation=rot("z", 90), translation=(0.01, 0.0, 0.0)
    ).apply_to_sequence(seq)
    after = played_areas(moved)
    assert after[0] == pytest.approx(0.0, abs=1e-6)
    assert abs(after[1]) == pytest.approx(abs(played_areas(seq)[0]), rel=1e-6)
    assert abs(float(moved.get_block(1).rf.freq_offset)) > 0.0


def test_a_scale_and_a_turn_compose_in_that_order(system):
    """The volume is resized in the frame it was designed in and then turned."""
    area = np.array((1000.0, 2000.0, -1500.0))
    scale = np.array((0.7, 0.5, 0.6))
    turn = rot("x", 20) @ rot("z", 30)
    seq = one_block(
        system,
        *(
            trap(axis, value, system, max_slew=system.max_slew / math.sqrt(3))
            for axis, value in zip("xyz", area, strict=True)
        ),
    )
    moved = pp.TransformFOV(
        scale=scale, rotation=turn, translation=(0.005, 0.01, -0.003)
    ).apply_to_sequence(seq)
    np.testing.assert_allclose(
        played_areas(moved), turn @ (scale * played_areas(seq)), rtol=1e-6
    )


def test_a_scale_and_a_turn_of_a_whole_readout_land_on_the_trajectory(system):
    area = np.array((3000.0, -1000.0, 2000.0))
    scale = np.array((0.5, 0.8, 0.7))
    turn = rot("y", 25) @ rot("z", 40)
    seq = one_block(
        system,
        *(
            trap(
                axis,
                value,
                system,
                duration=3e-3,
                max_slew=system.max_slew / math.sqrt(3),
            )
            for axis, value in zip("xyz", area, strict=True)
        ),
    )
    moved = pp.TransformFOV(scale=scale, rotation=turn).apply_to_sequence(seq)
    np.testing.assert_allclose(
        played_areas(moved), turn @ (scale * played_areas(seq)), rtol=1e-6
    )


# %% which blocks a prescription reaches


def three_shots(system):
    seq = pp.Sequence(system)
    gx = trap("x", 1000, system)
    for _ in range(3):
        seq.add_block(gx)
    return seq


def test_a_prescription_reaches_every_block_by_default(system):
    seq = three_shots(system)
    turned = pp.TransformFOV(rotation=rot("z", 90)).apply_to_sequence(seq)
    assert len(turned.block_durations) == 3
    for block in range(1, 4):
        assert turned.get_block(block).rotation is not None


def test_a_block_range_names_the_blocks_it_reaches(system):
    seq = three_shots(system)
    turned = pp.TransformFOV(rotation=rot("z", 90)).apply_to_sequence(
        seq, block_range=(2, 3)
    )
    assert len(turned.block_durations) == 3
    assert turned.get_block(1).rotation is None
    assert turned.get_block(2).rotation is not None
    assert turned.get_block(3).rotation is not None


def test_a_time_range_names_them_by_when_they_are_played(system):
    seq = three_shots(system)
    turned = pp.TransformFOV(rotation=rot("z", 90)).apply_to_sequence(
        seq, time_range=[0.0, 0.5 * float(seq.block_durations[1])]
    )
    assert turned.get_block(1).rotation is not None
    assert turned.get_block(3).rotation is None


def test_a_time_range_and_a_block_range_are_not_given_together(system):
    with pytest.raises(ValueError, match="not both"):
        pp.TransformFOV(rotation=rot("z", 90)).apply_to_sequence(
            three_shots(system), time_range=[0.0, 1.0], block_range=(1, 2)
        )


def test_a_scan_can_be_moved_a_repetition_at_a_time(system):
    """What a consumer too large to hold a scan at once does.

    The transform carries where both walks stand, so a range picked up where
    the last one left off gives what one call over the whole scan would have.
    Which ranges is the consumer's business -- `_detect_tr` says how long a
    repetition is and where the first one starts, and the pieces here are two
    of them at a time.
    """
    shift = 0.01
    rf = sinc(system)
    gx = flat("x", 2000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=float(gx.rise_time), system=system)
    step = pp.make_phase_encoding("y", 0.22 / 64, system=system)

    def scan():
        seq = pp.Sequence(system)
        for shot in range(6):
            seq.add_block(rf)
            seq.add_block(pp.scale_grad(step, -1.0 + shot / 3))
            seq.add_block(gx, adc)
        return seq

    at_once = pp.TransformFOV(translation=(shift, 0.0, 0.0)).apply_to_sequence(scan())

    in_pieces = scan()
    size, start = in_pieces._detect_tr()
    assert (size, start) == (3, 1)
    moving = pp.TransformFOV(translation=(shift, 0.0, 0.0))
    block = start
    while block <= len(in_pieces):
        stop = min(block + 2 * size - 1, len(in_pieces))
        moving.apply_to_sequence(in_pieces, block_range=(block, stop), in_place=True)
        block = stop + 1

    for readout in range(3, 19, 3):
        assert float(in_pieces.get_block(readout).adc.phase_offset) == pytest.approx(
            float(at_once.get_block(readout).adc.phase_offset), abs=1e-9
        )
        assert float(in_pieces.get_block(readout).adc.freq_offset) == pytest.approx(
            float(at_once.get_block(readout).adc.freq_offset), rel=1e-12
        )


# %% what a reconstructor is handed instead of a phase


def test_the_trajectory_is_given_for_the_blocks_that_acquire(system):
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=system)
    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(gx, adc)
    found = pp.TransformFOV(translation=(0.01, 0.0, 0.0)).trajectories(seq)
    assert [block for block, _ in found] == [2]
    assert found[0][1].shape == (3, 64)


def test_the_trajectory_is_the_one_the_sequence_says_it_samples(system):
    gx_pre = trap("x", -2500, system, duration=2e-3)
    gx = flat("x", 5000, 2e-3, system)
    adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=system)
    seq = pp.Sequence(system)
    seq.add_block(sinc(system))
    seq.add_block(gx_pre)
    seq.add_block(gx, adc)
    found = pp.TransformFOV(translation=(0.01, 0.0, 0.0)).trajectories(seq)
    np.testing.assert_allclose(
        found[0][1], np.asarray(seq.calculate_kspace()[0]), atol=1e-6
    )
