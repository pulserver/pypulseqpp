"""Arbitrary-gradient waveform design for non-Cartesian trajectories.

Thin Python wrapper around the MRArbGrad C++ solver, which this package
carries as a submodule under ``external/`` (see ``external/NOTICE.md``). Each
function below designs a
single slew/gradient-limited **base waveform** (one shot, standard
orientation) plus the number of shots required for full k-space coverage —
it does not enumerate shots or apply any rotation itself.

:func:`traj2grad` is the reusable exception: it accepts arbitrary 2D/3D
NumPy k-space samples in cycles/m and returns a rasterized gradient in Hz/m.

Where the shots go is not the solver's business and is not here either:
:func:`pypulseqpp.calc_golden_angles` and its family say what angle each shot
is played at, and :func:`pypulseqpp.make_rotation` turns one into a rotation
the base arm is replayed under.

Note there is a single :func:`spiral` entry point, not separate
constant-pitch/variable-density functions: MRArbGrad's variable-density
spiral trajectory function algebraically reduces to the exact constant-pitch
(Archimedean) formula when its two shape parameters are equal, so a separate
binding would be a redundant special case (see ``external/NOTICE.md`` for the
derivation).

Units
-----
:func:`spiral` and :func:`rosette` operate in MRArbGrad's native normalized
units:

- ``fov`` in meters, ``n_pix`` a plain pixel count.
- ``slew_limit`` in Hz/pix/s, ``grad_limit`` in Hz/pix (*not* T/m/s, T/m —
  these "per pixel" units fold the gyromagnetic ratio and FOV/matrix size
  into the limit so the solver never needs gamma explicitly).
- ``dt`` in seconds (gradient raster time).

Use :func:`to_gradient_tesla_per_meter` to convert a returned waveform to SI
units before handing it to :func:`pypulseqpp.make_arbitrary_grad`.

Examples
--------
>>> from pypulseqpp import _arbgrad as arbgrad
>>> wf = arbgrad.spiral(fov=0.256, n_pix=128,
...                     slew_limit=50 * 42.5756e6 * 0.256 / 128,
...                     grad_limit=50e-3 * 42.5756e6 * 0.256 / 128,
...                     dt=10e-6)
>>> wf.gradient.shape[1]
3
>>> wf.n_shots > 0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .._ext import arbgrad as _kernels

__all__ = [
    "BaseWaveform",
    "rosette",
    "spiral",
    "to_gradient_tesla_per_meter",
    "traj2grad",
]

DEFAULT_GAMMA_HZ_PER_T = 42.5756e6


@dataclass
class BaseWaveform:
    """Single-shot arbitrary-gradient waveform, in native Hz/pix units.

    Parameters
    ----------
    k0 : np.ndarray
        Shape ``(3,)`` k-space start offset of this shot (native units).
    gradient : np.ndarray
        Shape ``(n_samp, 3)`` gradient waveform samples, one row per
        ``dt``-spaced sample, columns are x/y/z (native Hz/pix units).
    n_shots : int
        Number of shots required for full (Nyquist) k-space coverage —
        callers replicate/rotate ``gradient`` this many times.
    """

    k0: np.ndarray
    gradient: np.ndarray
    n_shots: int


def traj2grad(
    trajectory: np.ndarray,
    max_slew: float,
    max_grad: float,
    dt: float,
    *,
    oversampling: int = 8,
    start_at_zero: bool = True,
    end_at_zero: bool = True,
) -> np.ndarray:
    """Generate a minimum-time gradient for sampled k-space coordinates.

    Unlike :func:`pypulseq.traj_to_grad`, this function reparameterizes the
    requested path with MRArbGrad, enforcing vector gradient and slew limits.
    It is therefore suitable for a NumPy-generated trajectory whose samples
    describe geometry, rather than already describing uniformly spaced time
    samples.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Shape ``(n, 2)`` or ``(n, 3)`` k-space path in cycles/m. Samples need
        not be uniformly spaced, but consecutive duplicates are not allowed.
    max_slew : float
        Vector slew limit in Hz/m/s.
    max_grad : float
        Vector gradient limit in Hz/m.
    dt : float
        Output gradient raster time in seconds.
    oversampling : int, optional
        Internal MRArbGrad integration oversampling, at least 2.
    start_at_zero, end_at_zero : bool, optional
        Constrain the gradient amplitude at the respective path endpoint to
        zero. Disable an endpoint when a continuous prewinder/rewinder will
        bridge directly to the readout gradient.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_grad, 3)`` gradient samples in Hz/m, one per ``dt``.
    """
    path = np.asarray(trajectory, dtype=float)
    if path.ndim != 2 or path.shape[1] not in (2, 3) or path.shape[0] < 4:
        raise ValueError("trajectory must have shape (n, 2) or (n, 3), with n >= 4")
    if not np.all(np.isfinite(path)):
        raise ValueError("trajectory samples must be finite")
    if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) == 0.0):
        raise ValueError("trajectory cannot contain consecutive duplicate samples")
    if max_slew <= 0 or max_grad <= 0 or dt <= 0:
        raise ValueError("max_slew, max_grad, and dt must be positive")
    if int(oversampling) < 2:
        raise ValueError("oversampling must be >= 2")

    return np.asarray(
        _kernels.sampled_trajectory_gradient(
            np.ascontiguousarray(path),
            float(max_slew),
            float(max_grad),
            float(dt),
            int(oversampling),
            bool(start_at_zero),
            bool(end_at_zero),
        )
    )


def spiral(
    fov: float,
    n_pix: int,
    slew_limit: float,
    grad_limit: float,
    dt: float,
    k_rho_phi0: float = 0.5 / (8.0 * pi),
    k_rho_phi1: float = 0.5 / (2.0 * pi),
) -> BaseWaveform:
    """Design a base spiral gradient waveform.

    Variable-density by default (``k_rho_phi0 != k_rho_phi1``); pass equal
    values for a plain constant-pitch (Archimedean) spiral.

    Parameters
    ----------
    fov : float
        Field of view, meters.
    n_pix : int
        Matrix size (pixels).
    slew_limit : float
        Slew-rate limit, Hz/pix/s.
    grad_limit : float
        Gradient-amplitude limit, Hz/pix.
    dt : float
        Gradient raster time, seconds.
    k_rho_phi0 : float, optional
        Inner (center) spiral shape parameter.
    k_rho_phi1 : float, optional
        Outer (edge) spiral shape parameter. Set equal to ``k_rho_phi0`` for
        a constant-pitch spiral.
    """
    k0, gradient, n_shots = _kernels.vdspiral_waveform(
        float(fov),
        int(n_pix),
        float(slew_limit),
        float(grad_limit),
        float(dt),
        float(k_rho_phi0),
        float(k_rho_phi1),
    )
    return BaseWaveform(k0=k0, gradient=gradient, n_shots=int(n_shots))


def rosette(
    fov: float,
    n_pix: int,
    slew_limit: float,
    grad_limit: float,
    dt: float,
    om1: float = 5.0 * pi,
    om2: float = 3.0 * pi,
    t_max: float = 1.0,
) -> BaseWaveform:
    """Design a base rosette gradient waveform.

    Parameters
    ----------
    fov, n_pix, slew_limit, grad_limit, dt
        See :func:`spiral`.
    om1, om2 : float, optional
        Rosette shape parameters (petal count/shape).
    t_max : float, optional
        Trajectory parameter upper bound.
    """
    k0, gradient, n_shots = _kernels.rosette_waveform(
        float(fov),
        int(n_pix),
        float(slew_limit),
        float(grad_limit),
        float(dt),
        float(om1),
        float(om2),
        float(t_max),
    )
    return BaseWaveform(k0=k0, gradient=gradient, n_shots=int(n_shots))


def to_gradient_tesla_per_meter(
    waveform: BaseWaveform | np.ndarray,
    fov: float,
    n_pix: int,
    gamma: float = DEFAULT_GAMMA_HZ_PER_T,
) -> np.ndarray:
    r"""Convert a native Hz/pix gradient array to SI units (T/m).

    Parameters
    ----------
    waveform : BaseWaveform or np.ndarray
        A :class:`BaseWaveform` (its ``gradient`` array is converted) or a
        bare ``(n_samp, 3)``/``(3,)`` array already in Hz/pix units (e.g.
        ``waveform.k0``, or a rotated copy of ``waveform.gradient``).
    fov : float
        Field of view used when designing the waveform, meters.
    n_pix : int
        Matrix size used when designing the waveform, pixels.
    gamma : float, optional
        Gyromagnetic ratio, Hz/T. Defaults to :sup:`1`\\ H (42.5756e6 Hz/T).

    Returns
    -------
    np.ndarray
        Same shape as the input array's gradient data, in T/m.
    """
    array = (
        waveform.gradient
        if isinstance(waveform, BaseWaveform)
        else np.asarray(waveform)
    )
    return array * (float(n_pix) / (float(gamma) * float(fov)))
