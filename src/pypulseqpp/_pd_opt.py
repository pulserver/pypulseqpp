"""Small-control forward-EPG fitting of prescribed-signal PD designs."""

import numpy as np
from scipy.optimize import minimize

from ._busse_opt import psf_weights
from ._fse_contrast import calc_fse_effective_te
from ._pd import _make_pd_schedules, _pd_schedule


def optimize_pd_controls(
    length,
    controls,
    *,
    startup,
    esp,
    first_echo,
    tr,
    centre,
    echo_map,
    t1_ms,
    t2_ms,
    b1_scales,
    contrast_pair,
    objective="psf",
    effective_te=None,
    initial_decay=1.3,
):
    """Fit first angle and signal-tail onset; keep actual peak/final angles fixed.

    Controls are first/maximum/final degrees and plateau fraction. The first
    changes by at most five degrees; tail onset by at most 0.1 of the acquired
    train. Bound RF energy by the prescribed train, retain 95% of centre mean
    signal and signed contrast to 4.5% of its magnitude across the B1 ensemble.
    An explicit reference effective TE is preserved to min(0.1 ms, 5% ESP).
    Return fitted controls, radians and signal levels, or raise on failure.
    """
    import torch
    from blochsim._execution import execution
    from blochsim.simulators import FSESimulator

    if objective not in ("psf", "contrast"):
        raise ValueError("PD objective must be psf or contrast")
    controls = np.asarray(controls, dtype=float)
    baseline, levels = _pd_schedule(
        length, *controls, startup, esp, first_echo, 1.0, 0.1, initial_decay
    )
    energy_form, moment_form = psf_weights(echo_map, length - startup)
    widths = np.array(
        [
            min(5.0, controls[0] / 10, (controls[1] - controls[0]) / 2),
            min(0.1, controls[3] / 2, (1 - controls[3]) / 2),
        ]
    )

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
            extra = first_echo - esp
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
            signal = model.simulate(
                flip=torch.as_tensor(np.rad2deg(flips), dtype=torch.float64),
                TR=tr * 1e3,
                repetitions=8,
            )
            return signal.abs().detach().numpy().astype(np.float64, copy=False)

        reference = simulate(baseline).reshape(len(b1_scales), len(t1_ms), length)
        bright, dark = contrast_pair
        reference_contrast = reference[:, bright, centre] - reference[:, dark, centre]
        scale = np.maximum(abs(reference_contrast), 1e-3)
        reference_signal = reference[:, :, centre].mean(1)
        budget = np.sum(baseline**2)
        cache = {}

        def evaluate(u):
            if np.array_equal(cache.get("u"), u):
                return cache["value"]
            step = 0.005
            probes = np.r_[u[None, :], u + step * np.eye(2), u - step * np.eye(2)]
            prescribed = np.tile(controls, (5, 1))
            prescribed[:, [0, 3]] += widths * probes
            flips, fitted_levels = _make_pd_schedules(
                np.full(5, length),
                prescribed,
                startup,
                esp,
                first_echo,
                decay=initial_decay,
                seed=np.tile(levels, (5, 1)),
                return_levels=True,
            )
            signal = simulate(flips)
            image_signal = signal[:, :, startup:]
            energy = np.einsum(
                "bti,ij,btj->bt", image_signal, energy_form, image_signal, optimize=True
            )
            moment = np.einsum(
                "bti,ij,btj->bt", image_signal, moment_form, image_signal, optimize=True
            )
            loss = np.sqrt(np.maximum(moment, 0) / np.maximum(energy, 1e-20)).mean(1)
            tissue = signal.reshape(5, len(b1_scales), len(t1_ms), length)
            contrast = tissue[:, :, bright, centre] - tissue[:, :, dark, centre]
            centre_signal = tissue[:, :, :, centre].mean(2)
            if objective == "contrast":
                loss = -(contrast / scale).mean(1)
            constraints = np.c_[
                (contrast - reference_contrast) / scale + 0.045,
                centre_signal / np.maximum(reference_signal, 1e-8) - 0.95,
                1.0 - np.sum(flips**2, axis=1) / budget,
            ]
            if effective_te is not None:
                equivalent = calc_fse_effective_te(
                    flips[:, : centre + 1], esp, first_echo_time=first_echo
                )[:, -1]
                delta = (equivalent - effective_te) / esp
                tolerance = min(1e-4, 0.05 * esp) / esp
                constraints = np.c_[constraints, tolerance - delta, tolerance + delta]
            value = (
                loss[0],
                (loss[1:3] - loss[3:5]) / (2 * step),
                constraints[0],
                ((constraints[1:3] - constraints[3:5]) / (2 * step)).T,
                prescribed[0],
                flips[0],
                fitted_levels[0],
            )
            cache.update(u=u.copy(), value=value)
            return value

        result = minimize(
            lambda u: evaluate(u)[:2],
            np.zeros(2),
            jac=True,
            method="SLSQP",
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
                f"PD control optimization did not converge: {result.message}"
            )
        return final[4].copy(), final[5].copy(), final[6].copy()
