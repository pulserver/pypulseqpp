"""The isochromat Bloch engine against closed forms and the relaxation-free kernel."""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp

RNG = np.random.default_rng(11)
ORIGIN = [[0.0, 0.0, 0.0]]


def _hard(flip_deg, phase=0.0, step=1e-7):
    """A pulse of one short step turning by ``flip_deg`` about ``phase`` from +x."""
    amplitude = flip_deg / 360.0 / step
    return (0.0, step, np.array([amplitude * np.exp(1j * phase)]))


def test_a_pulse_turns_as_the_mirror_of_the_right_handed_kernel():
    """Clockwise about (Re b1, Im b1, bz) is counter-clockwise about its mirror image."""
    b1 = 150.0 * (RNG.normal(size=300) + 1j * RNG.normal(size=300))
    off_resonance = RNG.normal(scale=400.0, size=6)
    spins = pp.Isochromats(np.zeros((6, 3)), off_resonance=off_resonance)
    spins.play(300e-6, rf=(0.0, 1e-6, b1))
    mirrored = pp.sim_bloch(np.conj(b1), off_resonance[:, None], 1e-6)
    np.testing.assert_allclose(
        spins.magnetization, mirrored * [1.0, -1.0, 1.0], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("kind", ["none", "one direction", "crossing"])
def test_a_pulse_under_a_gradient_turns_each_isochromat_about_its_own_field(kind):
    """Each step's field is the mean of the isochromat's over the step."""
    steps, step = 200, 2e-6
    b1 = 200.0 * (RNG.normal(size=steps) + 1j * RNG.normal(size=steps))
    times = np.linspace(0.0, steps * step, 7)
    ramp = np.array([0.0, 4e4, 1e4, 3e4, -2e4, 1e4, 0.0])
    other = {"none": 0.0 * ramp, "one direction": -0.5 * ramp, "crossing": ramp[::-1]}[
        kind
    ]
    gradients = {"none": [None, None, None]}.get(
        kind, [np.array([times, ramp]), np.array([times, other]), None]
    )
    count = 12
    positions = np.column_stack([RNG.uniform(-0.05, 0.05, (count, 2)), np.zeros(count)])
    off_resonance = RNG.normal(scale=100.0, size=count)
    spins = pp.Isochromats(positions, off_resonance=off_resonance)
    spins.play(steps * step, gradients=gradients, rf=(0.0, step, b1))

    edges = np.arange(steps + 1) * step
    areas = np.zeros((steps + 1, 3))
    if kind != "none":
        areas[:, 0] = [_area(times, ramp, t) for t in edges]
        areas[:, 1] = [_area(times, other, t) for t in edges]
    bz = (np.diff(areas, axis=0) @ positions.T).T / step + off_resonance[:, None]
    mirrored = pp.sim_bloch(np.conj(b1), bz, step)
    np.testing.assert_allclose(
        spins.magnetization, mirrored * [1.0, -1.0, 1.0], rtol=0, atol=1e-11
    )


def test_ninety_degrees_about_x_takes_z_to_y():
    spins = pp.Isochromats(ORIGIN)
    spins.play(1e-3, rf=_hard(90.0))
    np.testing.assert_allclose(spins.magnetization, [[0.0, 1.0, 0.0]], atol=1e-12)


def test_free_precession_is_exact_under_a_piecewise_linear_gradient():
    x, t2, t1, off_resonance = 0.03, 0.05, 0.4, 37.0
    spins = pp.Isochromats([[x, 0.0, 0.0]], t1=t1, t2=t2, off_resonance=off_resonance)
    spins.magnetization = [0.0, 1.0, 0.0]
    ramp = np.array([[0.2e-3, 0.7e-3, 1.9e-3, 2.3e-3], [0.0, 3e4, -1e4, 0.0]])
    duration = 3e-3
    spins.play(duration, gradients=[ramp, None, None])
    times, values = ramp
    area = np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(times))
    phase = -2 * np.pi * (x * area + off_resonance * duration)
    expected = 1j * np.exp(1j * phase - duration / t2)
    m = spins.magnetization[0]
    assert m[0] + 1j * m[1] == pytest.approx(expected, abs=1e-12)
    assert m[2] == pytest.approx(1.0 - np.exp(-duration / t1), abs=1e-12)


def _area(times, values, at):
    """The integral of the piecewise-linear waveform through the corners, up to ``at``."""
    fine = np.union1d(times, [at])
    fine = fine[fine <= at]
    return np.sum(
        0.5
        * np.diff(fine)
        * (np.interp(fine[1:], times, values) + np.interp(fine[:-1], times, values))
    )


def test_samples_within_a_window_follow_the_gradient_between_them():
    x = np.array([-0.02, 0.01, 0.045])
    spins = pp.Isochromats(np.column_stack([x, np.zeros(3), np.zeros(3)]))
    spins.magnetization = [0.0, 1.0, 0.0]
    # A trapezoid read out on its ramps and plateau, which no single
    # increment per sample describes.
    trapezoid = np.array([[0.0, 0.4e-3, 1.6e-3, 2e-3], [0.0, 2e4, 2e4, 0.0]])
    adc = np.linspace(0.05e-3, 1.95e-3, 40)
    signal = spins.play(2e-3, gradients=[trapezoid, None, None], adc=adc)
    areas = np.array([_area(*trapezoid, t) for t in adc])
    expected = 1j * np.exp(-2j * np.pi * np.outer(areas, x)).sum(axis=1)
    np.testing.assert_allclose(signal[0], expected, rtol=0, atol=1e-11)


def test_a_spin_echo_refocuses_off_resonance_and_decays_with_t2():
    count, t2, echo_time = 64, 0.05, 20e-3
    off_resonance = np.linspace(-500.0, 500.0, count)
    spins = pp.Isochromats(np.zeros((count, 3)), t2=t2, off_resonance=off_resonance)
    dephased = spins.play(echo_time / 2, rf=_hard(90.0), adc=[echo_time / 2])
    assert abs(dephased[0, 0]) < 0.05 * count
    echo = spins.play(
        echo_time / 2, rf=_hard(180.0, phase=np.pi / 2), adc=[echo_time / 2]
    )
    assert echo[0, 0] == pytest.approx(1j * count * np.exp(-echo_time / t2), rel=1e-4)


def test_a_short_t2_steady_state_is_the_ernst_signal():
    t1, t2, repetition, echo, flip = 0.1, 2e-3, 30e-3, 1e-3, 30.0
    spins = pp.Isochromats(ORIGIN, t1=t1, t2=t2)
    for _ in range(60):
        signal = spins.play(repetition, rf=_hard(flip), adc=[echo])
    e1 = np.exp(-repetition / t1)
    alpha = np.deg2rad(flip)
    ernst = np.sin(alpha) * (1 - e1) / (1 - e1 * np.cos(alpha)) * np.exp(-echo / t2)
    assert signal[0, 0] == pytest.approx(1j * ernst, rel=1e-4)


def test_balanced_ssfp_reaches_its_steady_state():
    t1, t2, repetition, flip = 0.3, 0.1, 5e-3, 60.0
    spins = pp.Isochromats(ORIGIN, t1=t1, t2=t2)
    for n in range(4000):
        # Alternating the pulse's phase puts the isochromat at the centre of
        # the passband.
        signal = spins.play(
            repetition, rf=_hard(flip, phase=np.pi * (n % 2)), adc=[repetition / 2]
        )
    e1, e2 = np.exp(-repetition / t1), np.exp(-repetition / t2)
    alpha = np.deg2rad(flip)
    steady = (1 - e1) * np.sin(alpha) / (1 - (e1 - e2) * np.cos(alpha) - e1 * e2)
    assert abs(signal[0, 0]) == pytest.approx(steady * np.sqrt(e2), rel=1e-4)


def _gradient_echo(fov=0.128, samples=64, rotation=None):
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    gx = pp.make_trapezoid(
        "x", flat_area=samples / fov, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(samples, duration=gx.flat_time, delay=gx.rise_time, system=system)
    pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    extra = [] if rotation is None else [rotation]
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=0.1e-3, system=system))
    seq.add_block(pre, *extra)
    seq.add_block(gx, adc, *extra)
    return seq


def test_a_gradient_echo_samples_the_transform_of_the_isochromats_at_their_k():
    positions = np.array([[-0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    density = np.array([1.0, 0.5, 2.0])
    seq = _gradient_echo()
    signal = seq.simulate(pp.Isochromats(positions, proton_density=density))
    k = seq.calculate_kspace()[0]
    expected = 1j * (density * np.exp(-2j * np.pi * (k.T @ positions.T))).sum(axis=1)
    np.testing.assert_allclose(signal[0], expected, rtol=0, atol=1e-8)
    # The inverse transform puts each isochromat at its own pixel of 2 mm.
    image = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(signal[0]))))
    assert sorted(np.argsort(image)[-3:] - 32) == [-10, 0, 15]


def test_a_rotated_block_plays_its_gradients_along_the_rotated_axes():
    positions = np.array([[0.0, -0.02, 0.0], [0.0, 0.031, 0.0]])
    seq = _gradient_echo(rotation=pp.make_rotation(np.pi / 2))
    signal = seq.simulate(pp.Isochromats(positions))
    k = seq.calculate_kspace()[0]
    assert np.abs(k[1]).max() > 0.0 and np.abs(k[0]).max() < 1e-9
    expected = 1j * np.exp(-2j * np.pi * (k.T @ positions.T)).sum(axis=1)
    np.testing.assert_allclose(signal[0], expected, rtol=0, atol=1e-8)


def test_a_rotated_gradient_steps_where_it_starts_and_ends_away_from_zero():
    """A step on one axis stays a step in the sum a rotation plays."""
    step = pp.make_extended_trapezoid(
        "x", times=np.array([0.0, 1e-3]), amplitudes=np.array([2e4, 2e4])
    )
    step.delay = 0.5e-3
    ramp = pp.make_trapezoid("y", amplitude=1e4, rise_time=0.2e-3, flat_time=1.6e-3)
    turn = pp.make_rotation(np.pi / 3)
    seq = pp.Sequence(pp.Opts())
    seq.add_block(step, ramp, turn, pp.make_delay(2.5e-3))
    positions = RNG.uniform(-0.05, 0.05, size=(6, 3))
    spins = pp.Isochromats(positions)
    spins.magnetization = [0.0, 1.0, 0.0]
    seq.simulate(spins)
    area = np.array([2e4 * 1e-3, 1e4 * (0.2e-3 + 1.6e-3), 0.0])
    matrix = np.array(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0.0],
            [np.sin(np.pi / 3), np.cos(np.pi / 3), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = 1j * np.exp(-2j * np.pi * positions @ (matrix @ area))
    m = spins.magnetization
    np.testing.assert_allclose(m[:, 0] + 1j * m[:, 1], expected, rtol=0, atol=1e-9)


def test_consecutive_block_ranges_play_one_scan():
    seq = pp.Sequence(pp.Opts())
    for n in range(6):
        seq.add_block(
            pp.make_block_pulse(np.deg2rad(20), duration=0.1e-3, phase_offset=0.3 * n)
        )
        seq.add_block(pp.make_trapezoid("z", area=200.0, duration=1e-3))
        seq.add_block(pp.make_adc(8, dwell=20e-6, phase_offset=0.3 * n))
    positions = RNG.uniform(-0.01, 0.01, size=(50, 3))
    whole = seq.simulate(pp.Isochromats(positions, t1=0.2, t2=0.05))
    spins = pp.Isochromats(positions, t1=0.2, t2=0.05)
    parts = [seq.simulate(spins, block_range=r) for r in ((1, 5), (6, 6), (7, 18))]
    np.testing.assert_allclose(np.concatenate(parts, axis=1), whole, rtol=0, atol=1e-12)
    assert spins.elapsed == pytest.approx(seq.duration()[0])


def test_the_adc_phase_offset_cancels_the_same_rf_phase_offset():
    for phase in (0.0, 0.7, 2.9):
        seq = pp.Sequence(pp.Opts())
        seq.add_block(
            pp.make_block_pulse(np.pi / 2, duration=0.1e-3, phase_offset=phase)
        )
        seq.add_block(pp.make_adc(1, dwell=10e-6, phase_offset=phase))
        signal = seq.simulate(pp.Isochromats(ORIGIN))
        assert signal[0, 0] == pytest.approx(1j, abs=1e-12)


def test_the_adc_follows_its_frequency_offset_and_phase_modulation():
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    gx = pp.make_trapezoid("x", flat_area=64 / 0.128, flat_time=3.2e-3, system=system)
    x = 0.023
    modulation = np.linspace(0.0, 1.0, 64) ** 2
    adc = pp.make_adc(
        64,
        duration=gx.flat_time,
        delay=gx.rise_time,
        freq_offset=gx.amplitude * x,
        phase_modulation=modulation,
        system=system,
    )
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=0.1e-3, system=system))
    seq.add_block(gx, adc)
    signal = seq.simulate(pp.Isochromats([[x, 0.0, 0.0]]))[0]
    # Demodulated at the isochromat's own frequency, it keeps the phase it
    # had at the window's start, plus the modulation.
    np.testing.assert_allclose(
        signal * np.exp(-1j * modulation),
        signal[0] * np.exp(-1j * modulation[0]),
        atol=1e-9,
    )


def test_a_shim_weights_a_pulse_onto_the_transmit_channels():
    rf = pp.make_block_pulse(np.pi / 6, duration=0.2e-3)
    flips = []
    for shim in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
        seq = pp.Sequence(pp.Opts())
        seq.add_block(rf, pp.make_rf_shim(shim))
        spins = pp.Isochromats(ORIGIN, transmit=[[1.0, 2.0]])
        seq.simulate(spins)
        flips.append(np.degrees(np.arccos(spins.magnetization[0, 2])))
    np.testing.assert_allclose(flips, [30.0, 60.0, 45.0], atol=1e-9)
    # Without a shim, the pulse plays on every channel.
    seq = pp.Sequence(pp.Opts())
    seq.add_block(rf)
    spins = pp.Isochromats(ORIGIN, transmit=[[1.0, 2.0]])
    seq.simulate(spins)
    assert np.degrees(np.arccos(spins.magnetization[0, 2])) == pytest.approx(
        90.0, abs=1e-9
    )
    # Without transmit sensitivities the shim has nothing to weight.
    seq = pp.Sequence(pp.Opts())
    seq.add_block(rf, pp.make_rf_shim([0.0, 1.0]))
    spins = pp.Isochromats(ORIGIN)
    seq.simulate(spins)
    assert np.degrees(np.arccos(spins.magnetization[0, 2])) == pytest.approx(
        30.0, abs=1e-9
    )


def test_simulate_takes_isochromats_only():
    with pytest.raises(TypeError, match="Isochromats"):
        pp.Sequence(pp.Opts()).simulate(np.zeros((1, 3)))


def test_a_positive_frequency_offset_excites_isochromats_at_a_positive_frequency():
    rf = pp.make_sinc_pulse(
        np.pi / 2, duration=4e-3, time_bw_product=4, freq_offset=2000.0
    )
    seq = pp.Sequence(pp.Opts())
    seq.add_block(rf)
    spins = pp.Isochromats(np.zeros((2, 3)), off_resonance=[2000.0, -2000.0])
    seq.simulate(spins)
    transverse = np.hypot(*spins.magnetization[:, :2].T)
    assert transverse[0] > 0.99 and transverse[1] < 0.02


def test_a_frequency_offset_and_the_same_phase_ramp_play_one_pulse():
    envelope = np.sinc(np.linspace(-2, 2, 1000)) * 300.0
    offset = 1500.0
    shifted = pp.make_arbitrary_rf(
        envelope, 0.0, freq_offset=offset, no_signal_scaling=True
    )
    # The ramp over the samples' own times, from the pulse's start.
    ramp = np.exp(2j * np.pi * offset * np.asarray(shifted.t))
    ramped = pp.make_arbitrary_rf(envelope * ramp, 0.0, no_signal_scaling=True)
    off_resonance = np.linspace(-3000.0, 3000.0, 13)
    results = []
    for rf in (shifted, ramped):
        seq = pp.Sequence(pp.Opts())
        seq.add_block(rf)
        spins = pp.Isochromats(np.zeros((13, 3)), off_resonance=off_resonance)
        seq.simulate(spins)
        results.append(spins.magnetization)
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-9)


def test_a_time_shaped_block_pulse_flips_as_one_on_the_raster():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_block_pulse(np.pi / 3, duration=0.25e-3))
    spins = pp.Isochromats(ORIGIN)
    seq.simulate(spins)
    np.testing.assert_allclose(
        spins.magnetization, [[0.0, np.sin(np.pi / 3), np.cos(np.pi / 3)]], atol=1e-9
    )


def test_isochromats_sharing_a_pulse_computation_answer_as_separate_ones():
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    rf, gz, _ = pp.make_sinc_pulse(
        np.pi / 2, duration=2e-3, slice_thickness=5e-3, return_gz=True, system=system
    )
    seq = pp.Sequence(system)
    seq.add_block(rf, gz)
    z = np.repeat(np.linspace(-6e-3, 6e-3, 25), 4)
    positions = np.column_stack([RNG.uniform(-0.1, 0.1, (z.size, 2)), z])
    shared = pp.Isochromats(positions, t2=0.08)
    # A T2 of its own for each isochromat, too close to change the answer,
    # gives each its own computation of the pulse.
    alone = pp.Isochromats(positions, t2=0.08 * (1 + 1e-13 * np.arange(z.size)))
    seq.simulate(shared)
    seq.simulate(alone)
    np.testing.assert_allclose(
        shared.magnetization, alone.magnetization, rtol=0, atol=1e-10
    )
    inside = np.abs(z) < 2e-3
    assert np.all(np.hypot(*shared.magnetization[inside, :2].T) > 0.9)
    assert np.all(np.hypot(*shared.magnetization[np.abs(z) > 4.5e-3, :2].T) < 0.05)


def test_threads_do_not_change_the_answer():
    positions = RNG.uniform(-0.05, 0.05, size=(20000, 3))
    receive = np.exp(1j * RNG.uniform(0, 2 * np.pi, size=(20000, 3)))
    seq = _gradient_echo()
    answers = [
        seq.simulate(
            pp.Isochromats(
                positions, t2=0.07, off_resonance=12.0, receive=receive, threads=n
            )
        )
        for n in (1, 4)
    ]
    np.testing.assert_allclose(answers[0], answers[1], rtol=1e-11, atol=1e-9)


def test_each_coil_receives_its_sensitivity_times_the_magnetisation():
    positions = RNG.uniform(-0.05, 0.05, size=(30, 3))
    receive = RNG.normal(size=(30, 2)) + 1j * RNG.normal(size=(30, 2))
    spins = pp.Isochromats(positions, receive=receive)
    spins.play(1e-3, rf=_hard(40.0))
    gradient = np.array([[0.0, 1e-3], [5e3, 5e3]])
    signal = spins.play(1e-3, gradients=[gradient, gradient, None], adc=[1e-3])
    m = spins.magnetization
    np.testing.assert_allclose(
        signal[:, 0], receive.T @ (m[:, 0] + 1j * m[:, 1]), atol=1e-12
    )


def test_a_transmit_sensitivity_scales_the_flip():
    spins = pp.Isochromats(np.zeros((2, 3)), transmit=[1.0, 0.5])
    spins.play(1e-3, rf=_hard(90.0))
    np.testing.assert_allclose(
        spins.magnetization,
        [[0.0, 1.0, 0.0], [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]],
        atol=1e-12,
    )


def test_the_channels_of_a_ptx_pulse_are_summed_without_transmit_sensitivities():
    samples = np.full((2, 100), 250.0 + 0j)
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_ptx_pulse(samples * [[1.0], [0.0]]))
    one = pp.Isochromats(ORIGIN)
    seq.simulate(one)
    ptx = pp.Sequence(pp.Opts())
    ptx.add_block(pp.make_ptx_pulse(samples * 0.5))
    summed = pp.Isochromats(ORIGIN)
    ptx.simulate(summed)
    np.testing.assert_allclose(summed.magnetization, one.magnetization, atol=1e-12)
    mapped = pp.Isochromats(ORIGIN, transmit=[[2.0, 0.0]])
    ptx.simulate(mapped)
    np.testing.assert_allclose(mapped.magnetization, one.magnetization, atol=1e-12)


def test_reset_and_the_magnetisation_setter():
    spins = pp.Isochromats(np.zeros((3, 3)), proton_density=[1.0, 2.0, 0.5])
    spins.play(1e-3, rf=_hard(90.0))
    spins.magnetization = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(spins.magnetization, np.tile([0.1, 0.2, 0.3], (3, 1)))
    spins.reset()
    np.testing.assert_allclose(spins.magnetization[:, 2], [1.0, 2.0, 0.5])
    assert spins.elapsed == 0.0 and len(spins) == 3 and spins.coils == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"positions": np.zeros((2, 2))}, "positions"),
        ({"t1": -1.0}, "positive"),
        ({"t2": [0.1, 0.1, 0.1]}, "t2"),
        ({"off_resonance": np.nan}, "finite"),
        ({"receive": np.ones((3, 2))}, "receive"),
    ],
)
def test_malformed_isochromats_are_refused(arguments, message):
    given = {"positions": np.zeros((2, 3)), **arguments}
    with pytest.raises(ValueError, match=message):
        pp.Isochromats(given.pop("positions"), **given)


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ({"rf": (0.0, 1e-3, [100.0]), "adc": [0.5e-3]}, "inside the RF pulse"),
        ({"adc": [2e-3]}, "within their block"),
        ({"adc": [0.5e-3, 0.2e-3]}, "increasing order"),
        ({"rf": (0.5e-3, 1e-3, [100.0])}, "within its block"),
        ({"rf": (0.0, 1e-4, np.ones((2, 3)))}, "channels"),
        (
            {"gradients": [np.array([[1e-4, 0.0], [1.0, 1.0]]), None, None]},
            "increasing",
        ),
    ],
)
def test_events_that_do_not_fit_are_refused(events, message):
    spins = pp.Isochromats(ORIGIN, transmit=[1.0])
    with pytest.raises(ValueError, match=message):
        spins.play(1e-3, **events)
