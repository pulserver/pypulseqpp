from __future__ import annotations

import math

import numpy as np
import pypulseq as pp
import pytest

from pypulseqpp import Sequence, calc_golden_angles, make_rotation
from pypulseqpp import _arbgrad as arbgrad

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # pragma: no cover - scipy is an existing pypulseqpp dependency
    Rotation = None

FOV_M = 0.256
N_PIX = 128
GAMMA_HZ_PER_T = 42.5756e6
SLEW_LIMIT_MT = 50.0  # T/m/s
GRAD_LIMIT_MT = 50e-3  # T/m
DT_S = 10e-6

SLEW_LIMIT_HZ_PIX_S = SLEW_LIMIT_MT * GAMMA_HZ_PER_T * FOV_M / N_PIX
GRAD_LIMIT_HZ_PIX = GRAD_LIMIT_MT * GAMMA_HZ_PER_T * FOV_M / N_PIX


def _assert_within_limits(waveform: arbgrad.BaseWaveform, tol: float = 1e-2) -> None:
    grad_si = arbgrad.to_gradient_tesla_per_meter(
        waveform, FOV_M, N_PIX, GAMMA_HZ_PER_T
    )
    grad_mag = np.linalg.norm(grad_si, axis=1)
    assert np.max(grad_mag) <= GRAD_LIMIT_MT * (1.0 + tol)

    slew_si = np.diff(grad_si, axis=0) / DT_S
    slew_mag = np.linalg.norm(slew_si, axis=1)
    assert np.max(slew_mag) <= SLEW_LIMIT_MT * (1.0 + tol)


def test_spiral_constant_pitch_special_case_shape_and_limits():
    """Equal k_rho_phi0/k_rho_phi1 is the constant-pitch (Archimedean) case."""
    k_rho_phi = 0.5 / (4.0 * math.pi)
    wf = arbgrad.spiral(
        FOV_M,
        N_PIX,
        SLEW_LIMIT_HZ_PIX_S,
        GRAD_LIMIT_HZ_PIX,
        DT_S,
        k_rho_phi0=k_rho_phi,
        k_rho_phi1=k_rho_phi,
    )

    assert wf.k0.shape == (3,)
    assert wf.gradient.ndim == 2
    assert wf.gradient.shape[1] == 3
    assert wf.n_shots >= 1
    # Spiral from k-space center: start offset is at the origin.
    assert np.allclose(wf.k0, 0.0, atol=1e-9)
    # Gradient must ramp from and back to zero (no residual moment discontinuity).
    assert np.allclose(wf.gradient[0], 0.0, atol=1e-6)
    assert np.allclose(wf.gradient[-1], 0.0, atol=1e-6)

    _assert_within_limits(wf)

    expected_n_shots = math.ceil(N_PIX * 2.0 * math.pi * k_rho_phi)
    assert wf.n_shots == expected_n_shots


def test_spiral_variable_density_shape_and_limits():
    wf = arbgrad.spiral(FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S)

    assert wf.gradient.shape[1] == 3
    assert wf.n_shots >= 1
    _assert_within_limits(wf)


def test_rosette_base_waveform_shape_and_limits():
    wf = arbgrad.rosette(FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S)

    assert wf.gradient.shape[1] == 3
    assert wf.n_shots >= 1
    _assert_within_limits(wf)


@pytest.mark.skipif(
    Rotation is None, reason="scipy is required to build rotation quaternions"
)
def test_base_waveform_consumable_by_fast_sequence_with_rotation():
    """The base arm, replayed under a rotation per shot, builds a sequence.

    Which is the whole division of labour: the solver designs one arm, the
    angle vocabulary says where each shot points, and a rotation extension
    carries it -- so a spiral of n_shots arms registers one waveform.
    """
    opts = pp.Opts()
    wf = arbgrad.spiral(FOV_M, N_PIX, SLEW_LIMIT_HZ_PIX_S, GRAD_LIMIT_HZ_PIX, DT_S)
    grad_si = arbgrad.to_gradient_tesla_per_meter(wf, FOV_M, N_PIX, GAMMA_HZ_PER_T)
    angles = calc_golden_angles(wf.n_shots)

    seq = Sequence(opts)
    for angle in angles[:3]:
        gx = pp.make_arbitrary_grad(
            channel="x", waveform=np.ascontiguousarray(grad_si[:, 0]), system=opts
        )
        gy = pp.make_arbitrary_grad(
            channel="y", waveform=np.ascontiguousarray(grad_si[:, 1]), system=opts
        )
        rotation = make_rotation(Rotation.from_euler("z", angle))
        seq.add_block(gx, gy, rotation)

    assert len(seq.block_events) == 3
