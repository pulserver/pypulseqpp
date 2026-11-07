"""Prescribed-PSS refocusing schedules with inverse EPG."""

from __future__ import annotations

from functools import lru_cache
from operator import index

import numpy as np
from scipy.optimize import root
from scipy.special import eval_legendre


def _pss(angle_deg):
    angle = np.deg2rad(angle_deg)
    sine = np.sin(angle / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sine) * eval_legendre(
            -0.5, sine * (1 + np.sin(angle) ** 2 / (8 * sine**4))
        )


def _shift(state):
    state[0, ..., 1:] = state[0, ..., :-1]
    state[1, ..., :-1] = state[1, ..., 1:]
    state[1, ..., -1] = 0
    state[0, ..., 0] = state[1, ..., 0].conjugate()


def _rotate(state, angle):
    plus, minus, longitudinal = state
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        (
            0.5 * (1 + cosine) * plus
            + 0.5 * (1 - cosine) * minus
            - 1j * sine * longitudinal,
            0.5 * (1 - cosine) * plus
            + 0.5 * (1 + cosine) * minus
            + 1j * sine * longitudinal,
            -0.5j * sine * plus + 0.5j * sine * minus + cosine * longitudinal,
        )
    )


def _inverse(target, first_deg):
    state = np.zeros((3, len(target) + 1), dtype=complex)
    state[:2, 0] = 1
    angles = np.empty(len(target))
    achieved = np.empty(len(target))
    for echo, desired in enumerate(target):
        _shift(state)
        if echo == 0:
            angle = np.deg2rad(first_deg)
        else:
            plus, minus, longitudinal = state[:, 1]
            a = 0.5 * (plus.real + minus.real)
            b = 0.5 * (minus.real - plus.real)
            c = -longitudinal.imag
            delta = np.arctan2(c, b)
            radius = np.hypot(b, c)
            turn = np.arccos(np.clip((desired - a) / max(radius, 1e-16), -1, 1))
            candidates = np.array(
                [
                    0,
                    np.pi,
                    delta,
                    delta + np.pi,
                    delta - turn,
                    delta + turn,
                    delta - turn + 2 * np.pi,
                    delta + turn + 2 * np.pi,
                ]
            )
            candidates = candidates[(candidates >= 0) & (candidates <= np.pi)]
            errors = abs(a + b * np.cos(candidates) + c * np.sin(candidates) - desired)
            # Equivalent echo solutions use the smaller RF rotation. Bounds
            # and extrema diagnose infeasible trials; accepted targets must
            # subsequently agree with the complete forward state evolution.
            angle = candidates[np.argmin(errors + 1e-12 * candidates)]
        state = _rotate(state, angle)
        _shift(state)
        achieved[echo] = state[0, 0].real
        angles[echo] = np.rad2deg(angle)
    return angles, achieved


def _target(controls, length, centre, first_signal, decay):
    low, middle, final = _pss(controls)
    echo = np.arange(length)
    signal = low + (middle - low) * np.clip((echo - 5) / (centre - 5), 0, 1) ** 1.5
    after = echo > centre
    signal[after] = middle + (final - middle) * (echo[after] - centre) / (
        length - 1 - centre
    )
    signal += (first_signal - low) * np.exp(-echo / decay)
    return signal


@lru_cache(maxsize=128)
def _schedule(length, minimum_deg, centre_deg, maximum_deg, centre, decay):
    requested = np.array([minimum_deg, centre_deg, maximum_deg])
    first_signal = np.sin(np.deg2rad(maximum_deg / 2)) ** 2
    last = {}

    def evaluate(controls):
        if np.array_equal(last.get("controls"), controls):
            return last["value"]
        target = _target(controls, length, centre, first_signal, decay)
        angles, achieved = _inverse(target, maximum_deg)
        residual = np.array([angles[1:centre].min(), angles[centre], angles[-1]])
        value = residual - requested, angles, achieved, target
        last.update(controls=np.array(controls), value=value)
        return value

    solution = root(lambda controls: evaluate(controls)[0], requested, tol=1e-9)
    residual, angles, achieved, target = evaluate(solution.x)
    if (
        not np.all(np.isfinite(angles))
        or np.max(abs(residual)) > 1e-6
        or np.max(abs(achieved - target)) > 1e-10
        or angles[1:].min() < minimum_deg - 1e-6
        or angles.max() > maximum_deg + 1e-6
        or not (0 < solution.x[0] < solution.x[1] < solution.x[2] < 180)
    ):
        raise ValueError(
            "The prescribed Busse controls do not give a feasible refocusing train"
        )
    result = np.deg2rad(angles)
    result.flags.writeable = False
    return result


def _integer(value, name):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def make_busse_schedule(
    length: int,
    *,
    minimum_flip_angle: float = np.deg2rad(25.0),
    centre_flip_angle: float = np.deg2rad(70.0),
    maximum_flip_angle: float = np.deg2rad(120.0),
    centre_echo: int | None = None,
    startup_echoes: int = 6,
    initial_decay: float = 1.3,
) -> np.ndarray:
    """Construct a prescribed-PSS refocusing train with inverse EPG.

    The first and final pulses use the maximum angle. An initial asymptotic
    approach establishes the low-angle state; PSS coherence then increases
    through the centre control toward the final control. A shooting correction
    enforces the actual minimum, centre and final RF angles. The output is
    independent of echo spacing: relaxation belongs to the forward EPG model.

    Parameters
    ----------
    length : int
        Total number of refocusing pulses, including startup pulses.
    minimum_flip_angle : float, default=np.deg2rad(25.0)
        Minimum actual refocusing angle (rad).
    centre_flip_angle : float, default=np.deg2rad(70.0)
        Actual refocusing angle at ``centre_echo`` (rad).
    maximum_flip_angle : float, default=np.deg2rad(120.0)
        Actual first and final refocusing angle (rad).
    centre_echo : int | None, default=None
        Zero-based RF pulse index, including startup pulses. ``None`` selects
        ``startup_echoes + (length - startup_echoes - 1) // 2``.
    startup_echoes : int, default=6
        Leading unacquired pulses, used to locate the default centre. The
        caller excludes their ADC events; this function emits angles only.
    initial_decay : float, default=1.3
        Initial exponential time constant (echoes). This reconstruction
        parameter is not specified numerically in the reference paper.

    Returns
    -------
    numpy.ndarray
        Actual RF flip angles (rad), shape ``(length,)``, including startup.
        Each call returns an independent array.

    Raises
    ------
    ValueError
        If counts or controls are invalid, the centre is not after RF6 and
        before the final pulse, or the prescribed train is infeasible.

    Notes
    -----
    Controls satisfy ``0 < minimum < centre < maximum < pi``. The signal
    segments use powers 1.5 before the centre and 1 afterward [1]_. No
    relaxation, transmit inhomogeneity or slice profile enters the inversion.
    These effects can be evaluated while optimizing the angle controls with
    an outer forward-EPG objective; the constructor remains the same.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> angles = pp.make_busse_schedule(106)
    >>> angles.shape
    (106,)
    >>> np.allclose(np.rad2deg(angles[[np.argmin(angles), 55, -1]]), [25, 70, 120])
    True

    References
    ----------
    .. [1] Busse RF et al. Effects of refocusing flip angle modulation and view
       ordering in 3D fast spin echo. *Magnetic Resonance in Medicine*.
       2008;60:640-649. https://doi.org/10.1002/mrm.21680
    """
    length = _integer(length, "length")
    startup = _integer(startup_echoes, "startup_echoes")
    if length < 1 or not 0 <= startup < length:
        raise ValueError("length must be positive and 0 <= startup_echoes < length")
    centre = (
        startup + (length - startup - 1) // 2
        if centre_echo is None
        else _integer(centre_echo, "centre_echo")
    )
    if not 5 < centre < length - 1:
        raise ValueError("centre_echo must be after RF6 and before the final pulse")
    controls = np.asarray([minimum_flip_angle, centre_flip_angle, maximum_flip_angle])
    if (
        not np.all(np.isfinite(controls))
        or not 0 < controls[0] < controls[1] < controls[2] < np.pi
    ):
        raise ValueError("Angles must satisfy 0 < minimum < centre < maximum < pi")
    if not np.isfinite(initial_decay) or initial_decay <= 0:
        raise ValueError("initial_decay must be positive and finite")
    return _schedule(length, *np.rad2deg(controls), centre, float(initial_decay)).copy()


def _inverse_many(target, first_deg):
    count, length = target.shape
    state = np.zeros((3, count, length + 1), dtype=complex)
    state[:2, :, 0] = 1
    angles = np.empty_like(target)
    achieved = np.empty_like(target)
    rows = np.arange(count)
    for echo in range(length):
        _shift(state)
        if echo == 0:
            angle = np.deg2rad(first_deg)
        else:
            plus, minus, longitudinal = state[:, :, 1]
            a = 0.5 * (plus.real + minus.real)
            b = 0.5 * (minus.real - plus.real)
            c = -longitudinal.imag
            delta = np.arctan2(c, b)
            turn = np.arccos(
                np.clip(
                    (target[:, echo] - a) / np.maximum(np.hypot(b, c), 1e-16), -1, 1
                )
            )
            candidates = np.column_stack(
                (
                    np.zeros(count),
                    np.full(count, np.pi),
                    delta,
                    delta + np.pi,
                    delta - turn,
                    delta + turn,
                    delta - turn + 2 * np.pi,
                    delta + turn + 2 * np.pi,
                )
            )
            errors = abs(
                a[:, None]
                + b[:, None] * np.cos(candidates)
                + c[:, None] * np.sin(candidates)
                - target[:, echo, None]
            )
            errors[(candidates < 0) | (candidates > np.pi)] = np.inf
            angle = candidates[rows, np.argmin(errors + 1e-12 * candidates, axis=1)]
        state = _rotate(state, angle[:, None])
        _shift(state)
        achieved[:, echo] = state[0, :, 0].real
        angles[:, echo] = np.rad2deg(angle)
    return angles, achieved


def _make_busse_schedules(lengths, controls_deg, centre, decay):
    """Batch shooting for variable trains; controls are actual RF degrees.

    Every row uses the same prescribed-PSS family as make_busse_schedule.
    Lengths include startup. Padding after each row's final pulse is zero.
    Batched finite-difference Newton steps avoid a scalar EPG solve per shot.
    """
    lengths = np.asarray(lengths, dtype=int)
    requested = np.asarray(controls_deg, dtype=float)
    maximum = int(lengths.max())
    if np.any(lengths <= centre + 1) or centre <= 5:
        raise ValueError("The Busse centre must precede the final pulse of every train")
    count = len(lengths)
    echo = np.arange(maximum)[None, :]

    def evaluate(controls, repeats=1):
        active = np.tile(lengths, repeats)
        wanted = np.tile(requested, (repeats, 1))
        low, middle, final = _pss(controls.T)[:, :, None]
        signal = low + (middle - low) * np.clip((echo - 5) / (centre - 5), 0, 1) ** 1.5
        signal = np.where(
            echo > centre,
            middle
            + (final - middle) * (echo - centre) / (active[:, None] - 1 - centre),
            signal,
        )
        first = np.sin(np.deg2rad(wanted[:, 2, None] / 2)) ** 2
        signal += (first - low) * np.exp(-echo / decay)
        angles, achieved = _inverse_many(signal, wanted[:, 2])
        actual = np.column_stack(
            (
                angles[:, 1:centre].min(1),
                angles[:, centre],
                angles[np.arange(len(active)), active - 1],
            )
        )
        return actual - wanted, angles, achieved, signal

    controls = requested.copy()
    step = 1e-3
    for _ in range(12):
        probes = np.concatenate(
            [controls, *(controls + step * axis for axis in np.eye(3))]
        )
        residual, angles, achieved, target = evaluate(probes, 4)
        base = residual[:count]
        if np.max(abs(base)) < 1e-7:
            break
        jacobian = np.stack(
            [
                (residual[(i + 1) * count : (i + 2) * count] - base) / step
                for i in range(3)
            ],
            axis=-1,
        )
        try:
            correction = np.linalg.solve(jacobian, base[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        controls -= np.clip(correction, -10, 10)
    residual, angles, achieved, target = evaluate(controls)
    played = echo < lengths[:, None]
    valid = (
        (abs(residual).max(1) < 1e-6)
        & np.all(np.where(played, abs(achieved - target), 0) < 1e-10, axis=1)
        & np.all(np.where(played, angles >= requested[:, :1] - 1e-6, True), axis=1)
        & np.all(np.where(played, angles <= requested[:, 2:] + 1e-6, True), axis=1)
        & (controls[:, 0] > 0)
        & np.all(np.diff(controls, axis=1) > 0, axis=1)
        & (controls[:, 2] < 180)
    )
    angles[~played] = 0
    # Exceptional nonsmooth roots use the canonical scalar shooting solver.
    for row in np.flatnonzero(~valid):
        angles[row, : lengths[row]] = np.rad2deg(
            make_busse_schedule(
                int(lengths[row]),
                minimum_flip_angle=np.deg2rad(requested[row, 0]),
                centre_flip_angle=np.deg2rad(requested[row, 1]),
                maximum_flip_angle=np.deg2rad(requested[row, 2]),
                centre_echo=centre,
                startup_echoes=0,
                initial_decay=decay,
            )
        )
    return np.deg2rad(angles)
