"""The gradient each RF pulse plays under, and whether it holds one value across the pulse."""

import math

import numpy as np
import pytest

import pypulseqpp as pp

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def selective(system, **kwargs):
    pulse, gz, _ = pp.make_sinc_pulse(
        math.pi / 2,
        duration=2e-3,
        slice_thickness=5e-3,
        return_gz=True,
        use="excitation",
        system=system,
        **kwargs,
    )
    return pulse, gz


def test_a_slice_selective_pulse_is_steady_at_its_plateau(system):
    seq = pp.Sequence(system)
    pulse, gz = selective(system)
    seq.add_block(pulse, gz)

    under = seq.rf_gradients()

    assert under.block.tolist() == [1]
    assert under.steady.tolist() == [[True, True, True]]
    np.testing.assert_allclose(under.gradient, [[0.0, 0.0, gz.amplitude]])


def test_a_nonselective_pulse_is_steady_at_zero(system):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"))

    under = seq.rf_gradients()

    assert under.steady.tolist() == [[True, True, True]]
    assert under.gradient.tolist() == [[0.0, 0.0, 0.0]]


def test_a_gradient_that_ends_under_the_pulse_is_not_steady(system):
    seq = pp.Sequence(system)
    pulse = pp.make_block_pulse(math.pi / 2, duration=2e-3, use="excitation")
    gz = pp.make_trapezoid(
        "z", amplitude=1000, flat_time=1e-3, rise_time=1e-4, system=system
    )
    seq.add_block(pulse, gz)

    under = seq.rf_gradients()

    assert under.steady.tolist() == [[True, True, False]]
    np.testing.assert_allclose(under.gradient, [[0.0, 0.0, 1000.0]])


def test_a_pulse_that_starts_on_the_ramp_is_not_steady(system):
    """The plateau is reached a raster after the pulse's first sample."""
    seq = pp.Sequence(system)
    pulse, gz = selective(system)
    pulse.delay = gz.rise_time - system.grad_raster_time
    seq.add_block(pulse, gz)

    assert seq.rf_gradients().steady.tolist() == [[True, True, False]]


def test_a_pulse_under_a_ramping_waveform_is_not_steady_and_takes_its_centres_value(
    system,
):
    seq = pp.Sequence(system)
    pulse = pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation")
    ramp = pp.make_extended_trapezoid(
        "y", amplitudes=[0, 2000, 0], times=[0, 1e-3, 2e-3], system=system
    )
    pulse.delay = 0.5e-3
    seq.add_block(pulse, ramp)

    under = seq.rf_gradients()

    assert under.steady.tolist() == [[True, False, True]]
    np.testing.assert_allclose(under.gradient[0, 1], 2000.0)


def test_the_gradient_is_along_the_channel_axes_as_stored(system):
    """A block's rotation does not move it: the axes are those before the rotation."""
    seq = pp.Sequence(system)
    pulse, gz = selective(system)
    seq.add_block(
        pulse, gz, pp.make_rotation(Rotation.from_euler("y", 90, degrees=True))
    )

    under = seq.rf_gradients()

    assert under.steady.tolist() == [[True, True, True]]
    np.testing.assert_allclose(under.gradient, [[0.0, 0.0, gz.amplitude]])


def test_a_ptx_pulse_is_steady_across_one_channels_span(system):
    """The plateau covers one channel's 2 ms, not the two channels' 4 ms."""
    seq = pp.Sequence(system)
    pulse = pp.make_ptx_pulse(
        100.0 * np.ones((2, 1000)), delay=1e-4, dwell=2e-6, use="excitation"
    )
    gz = pp.make_trapezoid(
        "z", amplitude=1000, flat_time=2.2e-3, rise_time=1e-4, system=system
    )
    seq.add_block(pulse, gz)

    assert seq.rf_gradients().steady.tolist() == [[True, True, True]]


def test_only_blocks_with_rf_are_listed_in_play_order(system):
    seq = pp.Sequence(system)
    pulse, gz = selective(system)
    seq.add_block(pp.make_delay(1e-3))
    seq.add_block(pulse, gz)
    seq.add_block(pp.make_trapezoid("x", area=100, duration=1e-3, system=system))
    seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3, use="refocusing"))

    under = seq.rf_gradients()

    assert under.block.tolist() == [2, 4]
    assert under.steady.shape == (2, 3)
    assert under.gradient.shape == (2, 3)


@pytest.mark.parametrize(
    "ends_under_the_pulse", [False, True], ids=["steady", "moving"]
)
def test_a_steady_pulse_is_moved_without_a_phase_shape(system, ends_under_the_pulse):
    """TransformFOV writes a phase shape exactly where the gradient under the pulse moves."""
    seq = pp.Sequence(system)
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=2e-3, delay=1e-4, use="excitation"
    )
    flat = 1e-3 if ends_under_the_pulse else 2.2e-3
    seq.add_block(
        pulse,
        pp.make_trapezoid(
            "z", amplitude=1000, flat_time=flat, rise_time=1e-4, system=system
        ),
    )
    steady = bool(seq.rf_gradients().steady.all())

    moved = pp.TransformFOV(translation=(0.0, 0.0, 0.01)).apply_to_sequence(seq)

    reshaped = not np.allclose(moved.get_block(1).rf.signal, seq.get_block(1).rf.signal)
    assert steady is not ends_under_the_pulse
    assert reshaped is ends_under_the_pulse


def test_a_sequence_without_rf_has_no_entries():
    under = pp.Sequence(pp.Opts()).rf_gradients()

    assert under.block.shape == (0,)
    assert under.steady.shape == (0, 3)
    assert under.gradient.shape == (0, 3)
