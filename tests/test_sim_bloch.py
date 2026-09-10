"""The compiled Bloch simulation, held to the step-by-step algorithms it replaces.

``sim_bloch`` is held to a rotation of the magnetisation about each step's
field, and ``sim_rf`` to the quaternion composition MATLAB Pulseq's ``simRf``
performs, both written out here in NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp

RNG = np.random.default_rng(3)


def rotated_step_by_step(b1_hz, bz_hz, dt, initial):
    """Rodrigues rotation of the magnetisation about each step's field."""
    b1_hz = np.atleast_2d(np.asarray(b1_hz, dtype=complex))
    bz_hz = np.atleast_2d(np.asarray(bz_hz, dtype=float))
    m = np.array(np.broadcast_to(initial, (bz_hz.shape[0], 3)), dtype=float)
    for step in range(b1_hz.shape[1]):
        omega = (
            2
            * np.pi
            * dt
            * np.column_stack(
                [
                    np.broadcast_to(b1_hz[:, step].real, len(m)),
                    np.broadcast_to(b1_hz[:, step].imag, len(m)),
                    bz_hz[:, step if bz_hz.shape[1] > 1 else 0],
                ]
            )
        )
        angle = np.linalg.norm(omega, axis=1)
        axis = np.divide(
            omega,
            angle[:, None],
            out=np.zeros_like(omega),
            where=angle[:, None] > 1e-15,
        )
        cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
        dot = np.sum(axis * m, axis=1)[:, None]
        m = m * cosine + np.cross(axis, m) * sine + axis * dot * (1 - cosine)
    return m


def test_sim_bloch_is_the_step_by_step_rotation():
    steps, positions = 300, 7
    b1 = 300 * (RNG.normal(size=steps) + 1j * RNG.normal(size=steps))
    bz = RNG.normal(scale=400, size=(positions, steps))
    initial = RNG.normal(size=(positions, 3))
    expected = rotated_step_by_step(b1, bz, 5e-6, initial)
    assert np.allclose(
        pp.sim_bloch(b1, bz, 5e-6, initial=initial), expected, atol=1e-12
    )


def test_each_position_can_see_its_own_field():
    steps, positions = 200, 5
    b1 = 200 * (
        RNG.normal(size=(positions, steps)) + 1j * RNG.normal(size=(positions, steps))
    )
    bz = RNG.normal(scale=300, size=(positions, 1))
    expected = rotated_step_by_step(b1, bz, 5e-6, [0.0, 0.0, 1.0])
    assert np.allclose(pp.sim_bloch(b1, bz, 5e-6), expected, atol=1e-12)


def test_fields_for_different_positions_are_refused():
    with pytest.raises(ValueError, match="different number of positions"):
        pp.sim_bloch(np.ones((3, 10), dtype=complex), np.zeros((4, 1)), 1e-6)


def quaternion_sim_rf(rf, rephase_factor, prephase_factor, df=1.0, multiplier=4.0):
    """MATLAB Pulseq's simRf, as a quaternion composition, on sim_rf's own grid."""
    _, _, f_hz = pp.sim_rf(
        rf, rephase_factor, prephase_factor, df=df, bandwidth_multiplier=multiplier
    )[:3]
    f = 2 * np.pi * f_hz
    count = f.size
    bandwidth = abs(pp.calc_rf_bandwidth(rf, cutoff=0.5, dw=df * 10.0, dt=10e-6))
    dt = next(
        step
        for over, step in ((2e4, 1e-6), (1e4, 2e-6), (4e3, 5e-6), (0.0, 10e-6))
        if bandwidth > over
    )
    t = (np.arange(1, int(np.round(rf.shape_dur / dt)) + 1) - 0.5) * dt
    signal = np.asarray(rf.signal)
    envelope = (
        2
        * np.pi
        * (
            np.interp(t, rf.t, signal.real, left=0, right=0)
            + 1j * np.interp(t, rf.t, signal.imag, left=0, right=0)
        )
    )
    envelope = envelope * np.exp(
        1j * (rf.phase_offset + 2 * np.pi * rf.freq_offset * t)
    )

    def multiply(q, r):
        scalar = q[:, 0] * r[:, 0] - np.sum(q[:, 1:] * r[:, 1:], axis=1)
        vector = (
            q[:, :1] * r[:, 1:] + r[:, :1] * q[:, 1:] + np.cross(q[:, 1:], r[:, 1:])
        )
        return np.column_stack((scalar, vector))

    def precession(seconds):
        angle = -f * seconds
        return np.column_stack(
            (np.cos(angle / 2), np.zeros((count, 2)), np.sin(angle / 2))
        )

    def applied(q, vector):
        m = np.zeros((count, 4))
        m[:, 1:] = vector
        conjugate = q * np.array([1, -1, -1, -1])
        turned = multiply(conjugate, multiply(m, q))
        return turned[:, 1] + 1j * turned[:, 2], turned[:, 3]

    q = np.tile([1.0, 0.0, 0.0, 0.0], (count, 1))
    q = multiply(q, precession(dt * t.size * prephase_factor))
    for step in range(t.size):
        angle = -dt * np.sqrt(abs(envelope[step]) ** 2 + f**2)
        axis = np.column_stack(
            (
                np.full(count, envelope[step].real),
                np.full(count, envelope[step].imag),
                f,
            )
        )
        turning = np.abs(angle) > 0
        axis[turning] *= dt / np.abs(angle[turning, None])
        q = multiply(
            q, np.column_stack((np.cos(angle / 2), np.sin(angle / 2)[:, None] * axis))
        )
    q = multiply(q, precession(dt * t.size * rephase_factor))
    mz_xy, mz_z = applied(q, (0, 0, 1))
    mx_xy, _ = applied(q, (1, 0, 0))
    my_xy, _ = applied(q, (0, 1, 0))
    return mz_z, mz_xy, f_hz, (mx_xy + 1j * my_xy) / 2, mx_xy, my_xy


@pytest.mark.parametrize(
    "pulse, rephase, prephase",
    [
        ("block", 0.0, 0.0),
        ("sinc", -0.5, 0.0),
        ("sinc", -0.5, 0.25),
        ("slr", 0.0, 0.0),
    ],
)
def test_sim_rf_is_the_quaternion_composition_simrf_performs(pulse, rephase, prephase):
    if pulse == "block":
        rf = pp.make_block_pulse(np.pi / 2, duration=0.5e-3)
    elif pulse == "sinc":
        rf = pp.make_sinc_pulse(
            np.pi / 3, duration=2e-3, time_bw_product=4, phase_offset=0.4
        )
    else:
        rf = pp.make_slr_pulse(np.pi, duration=3e-3, pulse_type="se", use="refocusing")
    got = pp.sim_rf(rf, rephase, prephase)
    want = quaternion_sim_rf(rf, rephase, prephase)
    for mine, theirs in zip(got, want, strict=True):
        assert np.allclose(mine, theirs, atol=1e-10)
