"""Small-control forward-EPG fitting of prescribed-signal PD designs."""

import numpy as np
from scipy.optimize import minimize

from ._busse_opt import _ForwardSignal, _psf_radius, psf_weights
from ._fse_contrast import _effective_te_slopes
from ._pd import _make_pd_schedules, _pd_fraction_slopes, _pd_schedule


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
    """Fit signal-tail onset; keep actual first/peak/final angles fixed.

    Controls are first/maximum/final degrees and plateau fraction. Tail onset
    changes by at most 0.1 of the acquired train. Bound RF energy by the
    prescribed train, retain 95% of centre mean
    signal and signed contrast to 4.5% of its magnitude across the B1 ensemble.
    An explicit reference effective TE is preserved to min(0.1 ms, 5% ESP).
    Return fitted controls, radians and signal levels, or raise on failure.
    """
    from blochsim._execution import execution

    if objective not in ("psf", "contrast"):
        raise ValueError("PD objective must be psf or contrast")
    controls = np.asarray(controls, dtype=float)
    baseline, levels = _pd_schedule(
        length, *controls, startup, esp, first_echo, 1.0, 0.1, initial_decay
    )
    energy_form, moment_form = psf_weights(echo_map, length - startup)
    width = min(0.1, controls[3] / 2, (1 - controls[3]) / 2)
    timing = (startup, esp, first_echo, 1.0, 0.1, initial_decay)

    with execution("cpu"):
        forward = _ForwardSignal(
            length,
            esp=esp,
            first_echo_time=first_echo,
            tr=tr,
            t1_ms=t1_ms,
            t2_ms=t2_ms,
            b1_scales=b1_scales,
        )
        reference = forward(baseline).reshape(len(b1_scales), len(t1_ms), length)
        bright, dark = contrast_pair
        reference_contrast = reference[:, bright, centre] - reference[:, dark, centre]
        scale = np.maximum(abs(reference_contrast), 1e-3)
        reference_signal = reference[:, :, centre].mean(1)
        budget = np.sum(baseline**2)
        cache = {}

        def evaluate(u):
            if np.array_equal(cache.get("u"), u):
                return cache["value"]
            prescribed = controls.copy()
            prescribed[3] += width * u[0]
            _, fitted_levels = _make_pd_schedules(
                [length],
                prescribed[None],
                startup,
                esp,
                first_echo,
                decay=initial_decay,
                seed=levels[None],
                return_levels=True,
            )
            flips, slope = _pd_fraction_slopes(
                length, prescribed, fitted_levels[0], *timing
            )
            # d flips / d u, in radians.
            directions = width * slope[None]
            signal, slopes = forward.slopes(flips, directions)
            radius, d_radius = _psf_radius(
                signal[:, startup:], slopes[:, :, startup:], energy_form, moment_form
            )
            tissue = signal.reshape(len(b1_scales), len(t1_ms), length)
            d_tissue = slopes.reshape(1, len(b1_scales), len(t1_ms), length)
            contrast = tissue[:, bright, centre] - tissue[:, dark, centre]
            d_contrast = d_tissue[:, :, bright, centre] - d_tissue[:, :, dark, centre]
            centre_signal = tissue[:, :, centre].mean(1)
            d_centre_signal = d_tissue[:, :, :, centre].mean(2)
            if objective == "contrast":
                loss = -(contrast / scale).mean()
                gradient = -(d_contrast / scale).mean(1)
            else:
                loss = radius.mean()
                gradient = d_radius.mean(1)
            signal_floor = np.maximum(reference_signal, 1e-8)
            constraints = [
                (contrast - reference_contrast) / scale + 0.045,
                centre_signal / signal_floor - 0.95,
                [1.0 - np.sum(flips**2) / budget],
            ]
            d_constraints = [
                d_contrast.T / scale[:, None],
                d_centre_signal.T / signal_floor[:, None],
                -2 * (directions @ flips)[None, :] / budget,
            ]
            if effective_te is not None:
                times, d_times = _effective_te_slopes(
                    flips[: centre + 1], directions[:, : centre + 1], esp, first_echo
                )
                delta = (times[-1] - effective_te) / esp
                d_delta = d_times[:, -1] / esp
                tolerance = min(1e-4, 0.05 * esp) / esp
                constraints.append([tolerance - delta, tolerance + delta])
                d_constraints.append(np.stack([-d_delta, d_delta]))
            value = (
                loss,
                gradient,
                np.concatenate(constraints),
                np.concatenate(d_constraints),
                prescribed,
                flips,
                fitted_levels[0],
            )
            cache.update(u=u.copy(), value=value)
            return value

        result = minimize(
            lambda u: evaluate(u)[:2],
            np.zeros(1),
            jac=True,
            method="SLSQP",
            bounds=((-1, 1),),
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
