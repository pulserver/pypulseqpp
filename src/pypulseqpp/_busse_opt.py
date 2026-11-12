"""Forward-EPG optimization of fixed-prescription Busse angle controls."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay, cKDTree

from ._busse import _schedule_slopes, make_busse_schedule
from ._fse_contrast import _effective_te_slopes


@lru_cache(maxsize=4)
def _weights(shape, length, encoded):
    echo_map = np.frombuffer(encoded, dtype=np.int32).reshape(shape)
    points = np.argwhere(echo_map >= 0)
    if min(shape) < 3 or len(points) < 4:
        raise ValueError(
            "Busse PSF optimization needs a two-dimensional encoding plane"
        )
    y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    radius = np.hypot(
        (y - shape[0] // 2) / (shape[0] / 2), (z - shape[1] // 2) / (shape[1] / 2)
    )
    keep = radius < 1
    grid = np.c_[y[keep], z[keep]]
    triangulation = Delaunay(points)
    simplex = triangulation.find_simplex(grid)
    inside = simplex >= 0
    vertices = triangulation.simplices[simplex[inside]]
    transform = triangulation.transform[simplex[inside]]
    barycentric = np.einsum(
        "ijk,ik->ij", transform[:, :2], grid[inside] - transform[:, 2]
    )
    barycentric = np.c_[barycentric, 1 - barycentric.sum(1)]
    echoes = echo_map[points[:, 0], points[:, 1]]
    rows = np.repeat(np.flatnonzero(inside), 3)
    columns = echoes[vertices].ravel()
    values = barycentric.ravel()
    outside = np.flatnonzero(~inside)
    nearest = cKDTree(points).query(grid[outside])[1]
    rows = np.r_[rows, outside]
    columns = np.r_[columns, echoes[nearest]]
    values = np.r_[values, np.ones(len(outside))]
    projection = csr_matrix((values, (rows, columns)), shape=(len(grid), length))
    aperture = 0.5 * (1 + np.cos(np.pi * np.clip((radius[keep] - 0.95) / 0.05, 0, 1)))
    projection = projection.multiply(aperture[:, None]).tocsr()
    basis = np.zeros((*shape, length))
    basis[keep] = projection.toarray()
    psf = np.fft.ifft2(np.fft.ifftshift(basis, axes=(0, 1)), axes=(0, 1), norm="ortho")
    dy = np.fft.fftfreq(shape[0]) * shape[0]
    dz = np.fft.fftfreq(shape[1]) * shape[1]
    displacement = dy[:, None] ** 2 + dz[None, :] ** 2
    flat = psf.reshape(-1, length)
    energy = (projection.T @ projection).toarray()
    moment = (flat.conj().T @ (displacement.ravel()[:, None] * flat)).real
    energy.flags.writeable = moment.flags.writeable = False
    return energy, moment


def psf_weights(echo_map, length):
    """Return ideal-interpolation PSF energy and squared-radius forms.

    ``echo_map`` is a phase/partition plane of acquired echo indices; -1
    marks unsampled points. Distances are in native phase/partition voxels.
    A common 5% cosine aperture taper is included. This is an idealized
    interpolation model, not a model of a nonlinear reconstruction.
    """
    plane = np.ascontiguousarray(echo_map, dtype=np.int32)
    return _weights(plane.shape, length, plane.tobytes())


class _ForwardSignal:
    """Echo magnitudes of a train over a tissue/B1 ensemble, and their derivatives.

    The train is blochsim's FSE with the first echo at ``first_echo_time``,
    played ``repetitions`` times at ``tr`` so that the last playing starts from
    the steady state. Derivatives with respect to the flip angles are taken in
    forward mode, one pass per direction, through blochsim's native kernels.
    """

    def __init__(self, length, *, esp, first_echo_time, tr, t1_ms, t2_ms, b1_scales):
        from blochsim.simulators import FSESimulator

        class TimedFSE(FSESimulator):
            def layout(
                self, *, flip, ESP, phases=0.0, exc_flip=90.0, exc_phase=90.0, TR=1e6
            ):
                parts = super().layout(
                    flip=flip,
                    ESP=ESP,
                    phases=phases,
                    exc_flip=exc_flip,
                    exc_phase=exc_phase,
                    TR=TR,
                )
                # FSESimulator takes ESP/TR in ms but returns event times in s.
                extra = first_echo_time - esp
                return [
                    (time + (extra / 2 if i == 1 else extra if i > 1 else 0), op)
                    for i, (time, op) in enumerate(parts)
                ]

        self.model = TimedFSE(
            ESP=esp * 1e3,
            states=2 * length + 1,
            T1=list(t1_ms) * len(b1_scales),
            T2=list(t2_ms) * len(b1_scales),
            B1=np.repeat(b1_scales, len(t1_ms)).tolist(),
        )
        self.tr_ms = tr * 1e3

    def _simulate(self, flip):
        return self.model.simulate(flip=flip, TR=self.tr_ms, repetitions=8)

    def __call__(self, flips):
        """Magnitudes ``(tissues, echoes)`` of ``flips`` (rad)."""
        import torch

        signal = self._simulate(torch.as_tensor(np.rad2deg(flips), dtype=torch.float64))
        return signal.abs().numpy().astype(np.float64, copy=False)

    def slopes(self, flips, directions):
        """Return magnitudes and their derivatives along ``directions``.

        ``flips`` and ``directions`` ``(k, echoes)`` are in radians; returns
        ``(tissues, echoes)`` and ``(k, tissues, echoes)``.
        """
        import torch
        from blochsim._derivative import forward_mode

        forward_mode()
        flip = torch.as_tensor(np.rad2deg(flips), dtype=torch.float64)
        tangents = torch.as_tensor(np.rad2deg(directions), dtype=torch.float64)
        signal, changes = torch.func.vmap(
            lambda tangent: torch.func.jvp(self._simulate, (flip,), (tangent,))
        )(tangents)
        signal = signal[0]
        magnitude = signal.abs()
        slopes = (signal.conj() * changes).real / magnitude.clamp_min(1e-30)
        return (
            magnitude.numpy().astype(np.float64, copy=False),
            slopes.numpy().astype(np.float64, copy=False),
        )


def _psf_radius(signal, slopes, energy_form, moment_form):
    """Ideal-interpolation PSF RMS radius per tissue, and its derivatives.

    ``signal`` is ``(tissues, echoes)`` and ``slopes`` ``(k, tissues,
    echoes)``; returns ``(tissues,)`` and ``(k, tissues)``.
    """
    energy = np.einsum("ti,ij,tj->t", signal, energy_form, signal, optimize=True)
    moment = np.einsum("ti,ij,tj->t", signal, moment_form, signal, optimize=True)
    d_energy = 2 * np.einsum("ti,ij,ktj->kt", signal, energy_form, slopes)
    d_moment = 2 * np.einsum("ti,ij,ktj->kt", signal, moment_form, slopes)
    energy = np.maximum(energy, 1e-20)
    moment = np.maximum(moment, 0)
    radius = np.sqrt(moment / energy)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_radius = (d_moment / energy - moment * d_energy / energy**2) / (2 * radius)
    return radius, np.where(radius > 0, d_radius, 0.0)


def optimize_controls(
    length,
    *,
    minimum_flip_angle,
    centre_flip_angle,
    maximum_flip_angle,
    centre_echo,
    startup_echoes,
    initial_decay,
    esp,
    first_echo_time,
    tr,
    echo_map,
    t1_ms,
    t2_ms,
    b1_scales,
    contrast_pair,
    objective="psf",
    effective_te=None,
):
    """Fit minimum/centre angles within five degrees of their prescription.

    The maximum/first/final angle, centre pulse, startup count and initial
    decay remain fixed. The objective is mean ideal-interpolation PSF RMS
    radius over the supplied tissue/B1 ensemble. RF energy is bounded by the
    prescribed train. Each B1 case loses at most 4.5% of the magnitude of its
    prescribed signed (bright-minus-dark) contrast.
    The contrast objective maximizes mean contrast relative to the prescription.
    If effective_te is supplied (seconds), preserve this reference contrast
    time at the centre within min(0.1 ms, 5% ESP), using relaxed/relaxation-free
    CPMG EPG at T1/T2 = 1.0/0.1 s. This prevents fitting from undoing a UI
    effective-TE prescription.
    Returns angles (rad); an unconverged or infeasible fit raises ValueError.
    """
    if objective not in ("psf", "contrast"):
        raise ValueError("objective must be psf or contrast")
    try:
        from blochsim.simulators import FSESimulator  # noqa: F401
    except ImportError as error:
        raise ImportError("Busse optimization requires pypulseqpp[design]") from error
    from blochsim._execution import execution

    protocol = {
        "minimum_flip_angle": minimum_flip_angle,
        "centre_flip_angle": centre_flip_angle,
        "maximum_flip_angle": maximum_flip_angle,
        "centre_echo": centre_echo,
        "startup_echoes": startup_echoes,
        "initial_decay": initial_decay,
    }
    baseline = make_busse_schedule(length, **protocol)
    acquired = length - startup_echoes
    energy_form, moment_form = psf_weights(echo_map, acquired)
    requested = np.rad2deg([minimum_flip_angle, centre_flip_angle])
    maximum_deg = np.rad2deg(maximum_flip_angle)
    window = min(
        5.0,
        requested[0] / 2,
        (requested[1] - requested[0]) / 4,
        (maximum_deg - requested[1]) / 4,
    )
    if window <= 0:
        raise ValueError("Busse optimization needs separated angle controls")
    with execution("cpu"):
        forward = _ForwardSignal(
            length,
            esp=esp,
            first_echo_time=first_echo_time,
            tr=tr,
            t1_ms=t1_ms,
            t2_ms=t2_ms,
            b1_scales=b1_scales,
        )
        reference = forward(baseline).reshape(len(b1_scales), len(t1_ms), length)
        bright, dark = contrast_pair
        reference_contrast = (
            reference[:, bright, centre_echo] - reference[:, dark, centre_echo]
        )
        if not np.all(np.isfinite(reference_contrast)):
            raise ValueError("The prescribed model contrast must be finite")
        contrast_scale = np.maximum(abs(reference_contrast), 1e-8)
        budget = np.sum(baseline**2)
        cached = {}

        def evaluate(u):
            if np.array_equal(cached.get("u"), u):
                return cached["value"]
            controls = requested + window * u
            flips, jacobian = _schedule_slopes(
                length,
                *controls,
                maximum_deg,
                centre_echo,
                float(initial_decay),
            )
            # d flips / d u, one row per control, in radians.
            directions = np.deg2rad(window) * jacobian[:, :2].T
            signal, slopes = forward.slopes(flips, directions)
            radius, d_radius = _psf_radius(
                signal[:, startup_echoes:],
                slopes[:, :, startup_echoes:],
                energy_form,
                moment_form,
            )
            tissue = signal.reshape(len(b1_scales), len(t1_ms), length)
            d_tissue = slopes.reshape(2, len(b1_scales), len(t1_ms), length)
            contrast = tissue[:, bright, centre_echo] - tissue[:, dark, centre_echo]
            d_contrast = (
                d_tissue[:, :, bright, centre_echo] - d_tissue[:, :, dark, centre_echo]
            )
            if objective == "contrast":
                loss = -(contrast / contrast_scale).mean()
                gradient = -(d_contrast / contrast_scale).mean(1)
            else:
                loss = radius.mean()
                gradient = d_radius.mean(1)
            constraints = [
                (contrast - reference_contrast) / contrast_scale + 0.045,
                [0.999 - np.sum(flips**2) / budget],
            ]
            d_constraints = [
                d_contrast.T / contrast_scale[:, None],
                -2 * (directions @ flips)[None, :] / budget,
            ]
            if effective_te is not None:
                times, d_times = _effective_te_slopes(
                    flips[: centre_echo + 1],
                    directions[:, : centre_echo + 1],
                    esp,
                    first_echo_time,
                )
                deviation = (times[-1] - effective_te) / esp
                d_deviation = d_times[:, -1] / esp
                tolerance = min(1e-4, 0.05 * esp) / esp
                constraints.append([tolerance - deviation, tolerance + deviation])
                d_constraints.append(np.stack([-d_deviation, d_deviation]))
            value = (
                loss,
                gradient,
                np.concatenate(constraints),
                np.concatenate(d_constraints),
                flips,
            )
            cached.update(u=u.copy(), value=value)
            return value

        result = minimize(
            lambda u: evaluate(u)[:2],
            np.zeros(2),
            method="SLSQP",
            jac=True,
            bounds=((-1, 1), (-1, 1)),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda u: evaluate(u)[2],
                    "jac": lambda u: evaluate(u)[3],
                }
            ],
            options={"maxiter": 20, "ftol": 1e-4},
        )
        final = evaluate(result.x)
        if not result.success or min(final[2]) < -1e-4:
            raise ValueError(
                f"Busse control optimization did not converge: {result.message}"
            )
        return final[4].copy()
