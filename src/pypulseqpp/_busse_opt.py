"""Forward-EPG optimization of fixed-prescription Busse angle controls."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay, cKDTree

from ._busse import make_busse_schedule


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
):
    """Fit minimum/centre angles within five degrees of their prescription.

    The maximum/first/final angle, centre pulse, startup count and initial
    decay remain fixed. The objective is mean ideal-interpolation PSF RMS
    radius over the supplied tissue/B1 ensemble. RF energy is bounded by the
    prescribed train. Each B1 case loses at most 4.5% of the magnitude of its
    prescribed signed (bright-minus-dark) contrast.
    The contrast objective maximizes mean contrast relative to the prescription.
    Returns angles (rad); an unconverged or infeasible fit raises ValueError.
    """
    if objective not in ("psf", "contrast"):
        raise ValueError("objective must be psf or contrast")
    try:
        import torch
        from blochsim._execution import execution
        from blochsim.simulators import FSESimulator
    except ImportError as error:
        raise ImportError("Busse optimization requires pypulseqpp[design]") from error

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

    with execution("cpu"):
        model = TimedFSE(
            ESP=esp * 1e3,
            states=2 * length + 1,
            T1=list(t1_ms) * len(b1_scales),
            T2=list(t2_ms) * len(b1_scales),
            B1=np.repeat(b1_scales, len(t1_ms)).tolist(),
        )

        def simulate(flips):
            return (
                model.simulate(
                    flip=torch.as_tensor(np.rad2deg(flips), dtype=torch.float64),
                    TR=tr * 1e3,
                    repetitions=8,
                )
                .abs()
                .detach()
                .numpy()
                .astype(np.float64, copy=False)
            )

        reference = simulate(baseline).reshape(len(b1_scales), len(t1_ms), length)
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
            step = 0.02
            probes = np.r_[u[None, :], u + step * np.eye(2), u - step * np.eye(2)]
            flips = []
            for probe in probes:
                controls = np.deg2rad(requested + window * probe)
                flips.append(
                    make_busse_schedule(
                        length,
                        **{
                            **protocol,
                            "minimum_flip_angle": controls[0],
                            "centre_flip_angle": controls[1],
                        },
                    )
                )
            flips = np.asarray(flips)
            signal = simulate(flips)
            image_signal = signal[:, :, startup_echoes:]
            energy = np.einsum(
                "bti,ij,btj->bt", image_signal, energy_form, image_signal, optimize=True
            )
            moment = np.einsum(
                "bti,ij,btj->bt", image_signal, moment_form, image_signal, optimize=True
            )
            loss = np.sqrt(np.maximum(moment, 0) / np.maximum(energy, 1e-20)).mean(1)
            tissue = signal.reshape(5, len(b1_scales), len(t1_ms), length)
            contrast = (
                tissue[:, :, bright, centre_echo] - tissue[:, :, dark, centre_echo]
            )
            if objective == "contrast":
                loss = -(contrast / contrast_scale).mean(1)
            constraints = np.c_[
                (contrast - reference_contrast) / contrast_scale + 0.045,
                0.999 - np.sum(flips**2, axis=1) / budget,
            ]
            value = (
                loss[0],
                (loss[1:3] - loss[3:5]) / (2 * step),
                constraints[0],
                ((constraints[1:3] - constraints[3:5]) / (2 * step)).T,
                flips[0],
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
