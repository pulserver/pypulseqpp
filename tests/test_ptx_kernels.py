"""Compiled small-tip pTx kernels.

The forward model is checked against a NumPy transcription of its sum and
against ``pp.sim_bloch`` at small tip; the solvers against the property each
claims.
"""

from __future__ import annotations

import numpy as np
import pytest
from pypulseqpp._ext import ptx

import pypulseqpp as pp

RNG = np.random.default_rng(7)
DWELL = 10e-6


def problem(channels=3, positions=40, samples=60, dims=2, off_resonance=True):
    """A random pTx problem on a gentle trajectory, in the kernels' units."""
    sens = RNG.normal(size=(channels, positions)) + 1j * RNG.normal(
        size=(channels, positions)
    )
    x = RNG.uniform(-0.05, 0.05, size=(positions, dims))
    gradient = 1e4 * RNG.normal(size=(samples, dims))
    kspace = -np.cumsum(gradient[::-1], axis=0)[::-1] * DWELL
    times = (np.arange(samples) + 0.5) * DWELL
    df = RNG.uniform(-50, 50, positions) if off_resonance else np.zeros(0)
    return {
        "sens": sens,
        "positions": x,
        "kspace": kspace,
        "times": times,
        "end": samples * DWELL,
        "scale": 2 * np.pi * DWELL,
        "off_resonance": df,
    }


def reference(b, sens, positions, kspace, times, end, scale, off_resonance):
    cycles = positions @ kspace.T
    if off_resonance.size:
        cycles = cycles + np.outer(off_resonance, times - end)
    return -1j * scale * np.einsum("cs,st,ct->s", sens, np.exp(-2j * np.pi * cycles), b)


def waveforms(model):
    shape = (model["sens"].shape[0], model["kspace"].shape[0])
    return RNG.normal(size=shape) + 1j * RNG.normal(size=shape)


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_the_forward_model_is_the_sum_it_states(dims):
    model = problem(dims=dims)
    b = waveforms(model)
    assert np.allclose(
        ptx.forward(b, **model), reference(b, **model), rtol=1e-10, atol=0.0
    )


def test_the_adjoint_is_the_adjoint():
    model = problem()
    b = waveforms(model)
    m = RNG.normal(size=model["positions"].shape[0]) + 1j * RNG.normal(
        size=model["positions"].shape[0]
    )
    left = np.vdot(ptx.forward(b, **model), m)
    right = np.vdot(b, ptx.adjoint(m, **model))
    assert left == pytest.approx(right, rel=1e-10)


def test_the_model_is_the_bloch_equation_at_small_tip():
    """A spin's field is the channels summed through their B1 maps."""
    samples, positions = 400, 9
    x = np.linspace(-0.05, 0.05, positions)[:, None]
    gradient = 1.6e4 * np.sin(2 * np.pi * np.arange(samples) / samples) + 4e3
    kspace = (-np.cumsum(gradient[::-1])[::-1] * DWELL)[:, None]
    sens = np.stack([np.ones(positions), np.exp(1j * np.linspace(0, np.pi, positions))])
    b = 1.0 * np.exp(1j * RNG.uniform(0, 2 * np.pi, size=(2, samples)))
    df = np.linspace(-80.0, 80.0, positions)
    times = (np.arange(samples) + 0.5) * DWELL
    predicted = ptx.forward(
        b, sens, x, kspace, times, samples * DWELL, 2 * np.pi * DWELL, df
    )
    simulated = []
    for s in range(positions):
        field = sens[:, s] @ b
        m = pp.sim_bloch(field, (x[s, 0] * gradient + df[s])[None, :], DWELL)[0]
        simulated.append(m[0] + 1j * m[1])
    simulated = np.asarray(simulated)
    assert np.abs(predicted - simulated).max() < 0.03 * np.abs(simulated).max()


def test_a_reachable_target_is_reached():
    model = problem(off_resonance=False)
    target = ptx.forward(waveforms(model), **model)
    weight = np.ones(target.size)
    b = ptx.design(target, weight, **model, iterations=500, tolerance=1e-12)
    assert np.abs(ptx.forward(b, **model) - target).max() < 1e-6 * np.abs(target).max()


def magnitude_error(model, b, target):
    return np.linalg.norm(np.abs(ptx.forward(b, **model)) - np.abs(target))


def test_magnitude_least_squares_fits_magnitude_better():
    model = problem(channels=2, positions=80, samples=30)
    target = np.full(80, 0.05 + 0j)
    weight = np.ones(80)
    plain = ptx.design(target, weight, **model, iterations=100)
    free = ptx.design(target, weight, **model, iterations=100, phase_updates=10)
    assert magnitude_error(model, free, target) < 0.8 * magnitude_error(
        model, plain, target
    )


def test_regularisation_trades_error_for_power():
    model = problem()
    target = ptx.forward(waveforms(model), **model) + 0.01
    weight = np.ones(target.size)
    loose = ptx.design(target, weight, **model, iterations=200)
    tight = ptx.design(target, weight, **model, iterations=200, regularization=1e-3)
    assert np.linalg.norm(tight) < np.linalg.norm(loose)
    assert np.linalg.norm(ptx.forward(tight, **model) - target) > np.linalg.norm(
        ptx.forward(loose, **model) - target
    )


def spread(field):
    magnitude = np.abs(field)
    return magnitude.std() / magnitude.mean()


def test_a_shim_evens_the_field_out():
    """Eight loops around a disc, each bright on its own side."""
    angle = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    y, x = np.mgrid[-1:1:24j, -1:1:24j]
    inside = x**2 + y**2 <= 1
    xs, ys = x[inside], y[inside]
    distance = np.hypot(
        xs[None] - 1.5 * np.cos(angle)[:, None], ys[None] - 1.5 * np.sin(angle)[:, None]
    )
    sens = np.exp(-distance) * np.exp(1j * (angle[:, None] + 2.0 * xs[None] * ys[None]))
    weights = ptx.rf_shim(sens, np.ones(xs.size), np.ones(xs.size))
    unit = np.ones(8)
    assert spread(weights @ sens) < 0.5 * spread(unit @ sens)


def test_more_spokes_correct_more_of_a_b1_profile():
    """One channel, bright in the middle: a single spoke cannot flatten it."""
    y, x = np.mgrid[-0.1:0.1:20j, -0.1:0.1:20j]
    positions = np.column_stack([x.ravel(), y.ravel()])
    sens = (1.2 - 30 * (x**2 + y**2)).ravel()[None, :].astype(complex)
    kmax, fov = 1 / 0.05, 0.2
    axis = np.linspace(-kmax / 2, kmax / 2 - 1 / fov, int(fov * kmax))
    grid = np.array([(a, b) for a in axis for b in axis if a or b])
    target = np.full(positions.shape[0], 0.1)
    weight = np.ones(positions.shape[0])
    errors = []
    for count in (1, 3, 5):
        where, amounts = ptx.spokes(sens, positions, target, weight, grid, count)
        assert where.shape == (count, 2) and amounts.shape == (1, count)
        field = (amounts[0][None, :] * np.exp(-2j * np.pi * positions @ where.T)).sum(
            1
        ) * sens[0]
        errors.append(np.linalg.norm(np.abs(field) - target))
    assert errors[1] < errors[0] and errors[2] < errors[1]


def test_a_spoke_sits_at_the_centre_of_k_space():
    y, x = np.mgrid[-0.1:0.1:10j, -0.1:0.1:10j]
    positions = np.column_stack([x.ravel(), y.ravel()])
    sens = np.ones((2, positions.shape[0]), dtype=complex)
    grid = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
    where, _ = ptx.spokes(
        sens,
        positions,
        np.ones(positions.shape[0]),
        np.ones(positions.shape[0]),
        grid,
        3,
    )
    assert any(np.allclose(row, 0.0) for row in where)
