"""A gradient reaching a given area between two amplitudes, in least time."""

from __future__ import annotations

__all__ = ["make_hexagon_gradient_area"]

import numpy as _np
import pypulseq as _pp
from pypulseq.make_extended_trapezoid import (
    make_extended_trapezoid as _make_extended_trapezoid,
)


def _to_raster(time_val: float, raster_time: float) -> float:
    return _np.ceil(time_val / raster_time) * raster_time


def _calc_ramp_time(g1: float, g2: float, max_slew: float, raster_time: float) -> float:
    return _to_raster(abs(g1 - g2) / max_slew, raster_time)


def _binary_search(fun, low: int, high: int):
    while low < high - 1:
        mid = (low + high) // 2
        times, _ = fun(mid)
        if len(times) != 0:
            high = mid
        else:
            low = mid
    return fun(high)


def _find_solution(
    duration: int,
    area: float,
    grad_start: float,
    grad_end: float,
    max_slew: float,
    max_grad: float,
    raster_time: float,
):
    sign_area = _np.sign(area)
    if sign_area == 0:
        sign_area = -_np.sign(grad_start + grad_end)
        if sign_area == 0:
            sign_area = 1.0
    grad_amp = sign_area * max_grad
    ramp_up_times = []
    ramp_down_times = []

    ru_min = abs(grad_amp - grad_start) / max_slew / raster_time
    rd_min = abs(grad_amp - grad_end) / max_slew / raster_time
    flat_time = max(duration - ru_min - rd_min, 0)
    area_check = (
        ru_min * (grad_amp + grad_start)
        + rd_min * (grad_amp + grad_end)
        + 2 * flat_time * grad_amp
    )

    if abs(2 * area / raster_time) > abs(area_check):
        return _np.array([]), _np.array([])

    ru = (duration * max_slew * raster_time + sign_area * (grad_end - grad_start)) / (
        2 * max_slew * raster_time
    )
    if sign_area * grad_start + ru * max_slew * raster_time > max_grad + 1e-5:
        ru_steps = round(
            abs(grad_start - sign_area * max_grad) / max_slew / raster_time
        )
        rd_steps = round(abs(grad_end - sign_area * max_grad) / max_slew / raster_time)
        flat_steps = duration - ru_steps - rd_steps
        if flat_steps > 0:
            grad_amp = -(
                ru_steps * raster_time * grad_start
                + rd_steps * raster_time * grad_end
                - 2 * area
            ) / ((ru_steps + 2 * flat_steps + rd_steps) * raster_time)
            amps = _np.array([grad_start, grad_amp, grad_amp, grad_end], dtype=float)
            t = _np.cumsum([0, ru_steps, flat_steps, rd_steps]) * raster_time
            slew = _np.diff(amps) / _np.diff(t)
            if (
                _np.max(_np.abs(slew)) < max_slew + 1e-5
                and _np.max(_np.abs(amps)) < max_grad
            ):
                return t, amps

    while abs(2 * area / raster_time) < abs(area_check):
        grad_amp = grad_amp / 2
        if abs(grad_amp) < abs(max_grad) / 10:
            ru_min = 0
            rd_min = 0
            flat_time = max(duration, 0)
            break
        ru_min = abs(grad_amp - grad_start) / max_slew / raster_time
        rd_min = abs(grad_amp - grad_end) / max_slew / raster_time
        flat_time = max(duration - ru_min - rd_min, 0)
        area_check = (
            ru_min * (grad_amp + grad_start)
            + rd_min * (grad_amp + grad_end)
            + 2 * flat_time * grad_amp
        )

    ru_min = _np.floor(ru_min)
    rd_min = _np.floor(rd_min)
    ru_limit = _np.ceil(abs(sign_area * max_grad - grad_start) / max_slew / raster_time)
    rd_limit = _np.ceil(abs(sign_area * max_grad - grad_end) / max_slew / raster_time)

    flat_time = duration - min(rd_min, rd_limit) - min(ru_min, ru_limit)
    flat_time_min = duration - rd_limit - ru_limit
    min_dif_area = area
    i = -1

    while flat_time > max(flat_time_min, -1):
        i += 1
        ru_max = ru_min + i
        rd_max = rd_min + i
        flat_time = duration - min(rd_min + i, rd_limit) - min(ru_min + i, ru_limit)

        if flat_time <= 0:
            denom_min = (
                grad_start - grad_end + sign_area * duration * max_slew * raster_time
            )
            denom_max = (
                -grad_start + grad_end + sign_area * duration * max_slew * raster_time
            )
            if denom_min != 0 and denom_max != 0:
                ru_0min = _np.floor(
                    (2 * area - duration * (grad_end + grad_start) * raster_time)
                    / denom_min
                    / raster_time
                )
                ru_0max = _np.ceil(
                    duration
                    - (2 * area - duration * (grad_end + grad_start) * raster_time)
                    / denom_max
                    / raster_time
                )
                for ru_try in range(int(ru_0min), int(ru_0max) + 1):
                    ramp_up_times.extend([ru_try, ru_try])
                    ramp_down_times.extend([duration - ru_try - 1, duration - ru_try])
            break

        grad_p0 = sign_area * max_slew * ru_max * raster_time + grad_start
        grad_p1 = sign_area * max_slew * rd_max * raster_time + grad_end

        if abs(grad_p0) >= max_grad:
            option1 = abs(
                sign_area * max_slew * (ru_limit - 1) * raster_time + grad_start
            )
            option2 = abs(
                ((ru_limit + flat_time) * sign_area * max_grad + grad_start - grad_p1)
                / (ru_limit + flat_time)
            )
            if option1 > option2:
                grad_p0 = (
                    sign_area * max_slew * (ru_limit - 1) * raster_time + grad_start
                )
                ru_max = _np.floor(
                    abs(sign_area * max_grad - grad_start) / max_slew / raster_time
                )
                ru_limit = ru_max
            else:
                grad_p0 = sign_area * max_grad
                ru_max = _np.ceil(
                    abs(sign_area * max_grad - grad_start) / max_slew / raster_time
                )
                ru_limit = ru_max
            flat_time = duration - rd_max - ru_max

        if abs(grad_p1) >= max_grad:
            option1 = abs(
                sign_area * max_slew * (rd_limit - 1) * raster_time + grad_end
            )
            option2 = abs(
                ((rd_limit + flat_time) * sign_area * max_grad + grad_end - grad_p0)
                / (rd_limit + flat_time)
            )
            if option1 > option2:
                grad_p1 = sign_area * max_slew * (rd_limit - 1) * raster_time + grad_end
                rd_max = _np.floor(
                    abs(sign_area * max_grad - grad_end) / max_slew / raster_time
                )
                rd_limit = rd_max
            else:
                grad_p1 = sign_area * max_grad
                rd_max = _np.ceil(
                    abs(sign_area * max_grad - grad_end) / max_slew / raster_time
                )
                rd_limit = rd_max
            flat_time = duration - rd_max - ru_max

        if abs(grad_p0 - grad_p1) / (flat_time * raster_time) > max_slew:
            if abs(grad_p0) < abs(grad_p1):
                grad_p1 = grad_p0 + flat_time * raster_time * max_slew * sign_area
                rd_max = _np.ceil(abs(grad_end - grad_p1) / max_slew / raster_time)
            else:
                grad_p0 = grad_p1 + flat_time * raster_time * max_slew * sign_area
                ru_max = _np.ceil(abs(grad_start - grad_p0) / max_slew / raster_time)
            flat_time = duration - rd_max - ru_max

        area_current = (
            raster_time
            / 2
            * (
                ru_max * (grad_p0 + grad_start)
                + flat_time * (grad_p0 + grad_p1)
                + rd_max * (grad_p1 + grad_end)
            )
        )

        if _np.sign(min_dif_area) != _np.sign(area - area_current):
            t = _np.round(_np.cumsum([0, ru_max, flat_time, rd_max]) * raster_time, 5)
            if abs(grad_p0) < abs(grad_p1):
                g_test = grad_p1
                corner_grad = -(
                    grad_start * ru_max
                    + grad_end * rd_max
                    + g_test * (flat_time + rd_max)
                    - 2 * area / raster_time
                ) / (flat_time + ru_max)
                amps = _np.array(
                    [grad_start, corner_grad, g_test, grad_end], dtype=float
                )
            else:
                g_test = grad_p0
                corner_grad = -(
                    grad_start * ru_max
                    + grad_end * rd_max
                    + g_test * (flat_time + ru_max)
                    - 2 * area / raster_time
                ) / (flat_time + rd_max)
                amps = _np.array(
                    [grad_start, g_test, corner_grad, grad_end], dtype=float
                )

            if _np.any(_np.round(_np.diff(t) / raster_time) < 1):
                continue
            slew = _np.diff(amps) / _np.diff(t)
            if _np.max(_np.abs(slew)) <= max_slew + 1e-5:
                return t, amps
            for ru_try in range(int(ru_max - 1), int(duration - rd_max) + 1):
                for rd_try in range(int(rd_max - 1), int(duration - ru_try) + 1):
                    flat = duration - ru_try - rd_try
                    grad_amp = -(
                        ru_try * raster_time * grad_start
                        + rd_try * raster_time * grad_end
                        - 2 * area
                    ) / ((ru_try + 2 * flat + rd_try) * raster_time)
                    amps = _np.array(
                        [grad_start, grad_amp, grad_amp, grad_end], dtype=float
                    )
                    t = _np.cumsum([0, ru_try, flat, rd_try]) * raster_time
                    # A candidate with a zero-length segment slews infinitely
                    # fast and is rejected by the test below rather than here.
                    with _np.errstate(divide="ignore", invalid="ignore"):
                        slew = _np.diff(amps) / _np.diff(t)
                    if (
                        _np.max(_np.abs(slew)) < max_slew + 1e-5
                        and _np.max(_np.abs(amps)) < max_grad
                    ):
                        return t, amps
        if abs(area_current - area) < abs(min_dif_area):
            min_dif_area = area - area_current

    ru_vec = _np.asarray(ramp_up_times, dtype=float)
    rd_vec = _np.asarray(ramp_down_times, dtype=float)
    valid = ru_vec * rd_vec > 0
    ru_vec = ru_vec[valid]
    rd_vec = rd_vec[valid]
    flat = duration - ru_vec - rd_vec
    valid = flat >= 0
    ru_vec = ru_vec[valid]
    rd_vec = rd_vec[valid]
    flat = flat[valid]
    if ru_vec.size == 0:
        return _np.array([]), _np.array([])

    grad_amp = -(
        ru_vec * raster_time * grad_start + rd_vec * raster_time * grad_end - 2 * area
    ) / ((ru_vec + 2 * flat + rd_vec) * raster_time)
    slew1 = abs(grad_start - grad_amp) / (ru_vec * raster_time)
    slew2 = abs(grad_end - grad_amp) / (rd_vec * raster_time)
    valid = (
        (abs(grad_amp) <= max_grad + 1e-5)
        & (slew1 <= max_slew + 1e-5)
        & (slew2 <= max_slew + 1e-5)
    )
    indices = _np.flatnonzero(valid)
    if indices.size == 0:
        return _np.array([]), _np.array([])

    idx = indices[0]
    t = _np.cumsum([0, ru_vec[idx], flat[idx], rd_vec[idx]]) * raster_time
    amps = _np.array([grad_start, grad_amp[idx], grad_amp[idx], grad_end], dtype=float)
    return t, amps


def make_hexagon_gradient_area(
    channel: str,
    grad_start: float,
    grad_end: float,
    area: float,
    system=None,
):
    """Design the shortest gradient carrying ``area`` between two amplitudes.

    A trapezoid starts and ends at zero. This does not: it is handed the
    amplitude the axis is already at and the amplitude the next event needs
    it at, and finds the shortest waveform between them enclosing the area
    asked for -- which is a hexagon rather than a trapezoid whenever the two
    ends differ. What a continuous readout needs between two arms, and what a
    spiral or a radial trajectory joins its shots with.

    The duration is searched for rather than solved: the shortest one that
    admits a solution is taken, from the ramp between the two end amplitudes
    upwards, and a doubling-then-bisection search takes over where that range
    holds none.

    Parameters
    ----------
    channel : str
        One of ``x``, ``y``, ``z``.
    grad_start, grad_end : float
        Amplitude at each end, in Hz/m.
    area : float
        The area to enclose, in 1/m.
    system : Opts, optional
        System limits. The gradients are designed against 99 percent of them,
        so the waveform that comes back is inside them rather than on them.

    Returns
    -------
    grad : SimpleNamespace
        The gradient event.
    times : numpy.ndarray
        The corner times, in seconds.
    amplitudes : numpy.ndarray
        The corner amplitudes, in Hz/m.

    Raises
    ------
    ValueError
        When either end amplitude is already over the limit, or when no
        waveform found encloses the area.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> grad, times, amplitudes = pp.make_hexagon_gradient_area(
    ...     "x", grad_start=0.0, grad_end=0.0, area=1000.0, system=system
    ... )
    >>> round(float(grad.area))
    1000

    Joining two arms of a readout means starting where the last one left off:

    >>> grad, _, amplitudes = pp.make_hexagon_gradient_area(
    ...     "x", grad_start=1e5, grad_end=-1e5, area=500.0, system=system
    ... )
    >>> float(amplitudes[0]), float(amplitudes[-1])
    (100000.0, -100000.0)
    """
    if system is None:
        system = _pp.Opts.default

    if abs(grad_start) > system.max_grad:
        raise ValueError(
            f"grad_start amplitude violation ({abs(grad_start) / system.max_grad * 100:.0f}%)"
        )
    if abs(grad_end) > system.max_grad:
        raise ValueError(
            f"grad_end amplitude violation ({abs(grad_end) / system.max_grad * 100:.0f}%)"
        )

    max_slew = system.max_slew * 0.99
    max_grad = system.max_grad * 0.99
    raster_time = system.grad_raster_time

    min_duration = max(
        round(
            _calc_ramp_time(grad_end, grad_start, max_slew, raster_time) / raster_time
        ),
        2,
    )
    max_duration = max(
        round(_calc_ramp_time(0, grad_start, max_slew, raster_time) / raster_time),
        round(_calc_ramp_time(0, grad_end, max_slew, raster_time) / raster_time),
        min_duration,
    )

    def solve(duration):
        return _find_solution(
            duration, area, grad_start, grad_end, max_slew, max_grad, raster_time
        )

    times = _np.array([])
    amplitudes = _np.array([])
    for duration in range(int(min_duration), int(max_duration) + 1):
        times, amplitudes = solve(duration)
        if len(times) != 0:
            break

    if len(times) == 0:
        duration = int(max_duration)
        while len(times) == 0:
            duration *= 2
            times, _ = solve(duration)
        times, amplitudes = _binary_search(solve, duration // 2, duration)

    keep = _np.concatenate(([True], _np.diff(times) > 0))
    times = times[keep]
    amplitudes = amplitudes[keep]

    grad = _make_extended_trapezoid(
        channel=channel, system=system, times=times, amplitudes=amplitudes
    )
    if abs(grad.area - area) >= 1e-3:
        raise ValueError(f"no waveform found enclosing area={area:.6f}")
    return grad, times, amplitudes
