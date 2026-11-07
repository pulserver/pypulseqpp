"""Relaxation-aware inverse-EPG PD trains with prescribed signal envelopes."""

from functools import lru_cache
from operator import index

import numpy as np

from pypulseqpp._busse import _rotate, _shift


def _inverse_pd(target, first, esp, first_echo, t1=1.0, t2=0.1):
    count, n = target.shape
    s = np.zeros((3, count, n + 1), complex)
    s[:2, :, 0] = 1
    angles = np.empty_like(target)
    signal = np.empty_like(target)
    rows = np.arange(count)
    for i in range(n):
        half = (first_echo if i == 0 else esp) / 2
        e1 = np.exp(-half / t1)
        e2 = np.exp(-half / t2)
        s[:2] *= e2
        s[2] *= e1
        s[2, :, 0] += 1 - e1
        _shift(s)
        if i == 0:
            a = np.deg2rad(first)
        else:
            plus, minus, z = s[:, :, 1]
            v = 0.5 * (plus.real + minus.real)
            b = 0.5 * (minus.real - plus.real)
            c = -z.imag
            phase = np.arctan2(c, b)
            r = np.hypot(b, c)
            turn = np.arccos(
                np.clip((target[:, i] / e2 - v) / np.maximum(r, 1e-16), -1, 1)
            )
            cs = np.column_stack(
                (
                    np.zeros(count),
                    np.full(count, np.pi),
                    phase,
                    phase + np.pi,
                    phase - turn,
                    phase + turn,
                    phase - turn + 2 * np.pi,
                    phase + turn + 2 * np.pi,
                )
            )
            err = abs(
                (v[:, None] + b[:, None] * np.cos(cs) + c[:, None] * np.sin(cs)) * e2
                - target[:, i, None]
            )
            err[(cs < 0) | (cs > np.pi)] = np.inf
            a = cs[rows, np.argmin(err + 1e-12 * cs, axis=1)]
        s = _rotate(s, a[:, None])
        s[:2] *= e2
        s[2] *= e1
        s[2, :, 0] += 1 - e1
        _shift(s)
        angles[:, i] = np.rad2deg(a)
        signal[:, i] = s[0, :, 0].real
    return angles, signal


def _solve_pd(lengths, controls, startup, esp, first_echo, t1, t2, decay, seed=None):
    lengths = np.asarray(lengths)
    controls = np.asarray(controls)
    count = len(lengths)
    n = lengths.max()
    x = np.arange(n)[None, :]
    active = x < lengths[:, None]
    first = controls[:, 0]
    initial = np.sin(np.deg2rad(first) / 2) ** 2 * np.exp(-first_echo / t2)

    def evaluate(level, ratio, repeat=1):
        nrows = len(level)
        lens = np.tile(lengths, repeat)
        c = np.tile(controls, (repeat, 1))
        init = np.tile(initial, repeat)
        cut = startup + c[:, 3, None] * (lens[:, None] - 1 - startup)
        z = np.clip((x - cut) / (lens[:, None] - 1 - cut), 0, 1)
        z = z * z
        target = level[:, None] * (1 + (ratio[:, None] - 1) * z) + (
            init[:, None] - level[:, None]
        ) * np.exp(-x / decay)
        a, s = _inverse_pd(target, c[:, 0], esp, first_echo, t1, t2)
        # The final pulse has its own constraint. Keeping it out of the peak
        # equation avoids identical Jacobian rows for a trial peaking at its end.
        mask = (x < lens[:, None] - 1) & (x >= startup)
        highest = np.where(mask, a, -np.inf).max(1)
        last = a[np.arange(nrows), lens - 1]
        return np.column_stack((highest - c[:, 1], last - c[:, 2])), a, s, target

    ratio = 0.65 * np.exp(-0.007 * (lengths - startup))
    if seed is None:
        lo = np.full(count, 0.001)
        hi = np.full(count, 0.99)
        for _ in range(34):
            mid = (lo + hi) / 2
            r, *_ = evaluate(mid, ratio)
            lo = np.where(r[:, 0] < 0, mid, lo)
            hi = np.where(r[:, 0] >= 0, mid, hi)
        level = (lo + hi) / 2
        p = np.column_stack((level, ratio))
    else:
        p = np.asarray(seed, dtype=float).copy()
    step = 1e-5
    for _iteration in range(16):
        v = np.concatenate(
            (p, p + step * np.array([1, 0]), p + step * np.array([0, 1]))
        )
        residual, *_ = evaluate(v[:, 0], v[:, 1], 3)
        r = residual[:count]
        if abs(r).max(initial=0) < 1e-6:
            break
        j = np.stack(
            (
                (residual[count : 2 * count] - r) / step,
                (residual[2 * count :] - r) / step,
            ),
            axis=-1,
        )
        det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
        safe = np.where(abs(det) > 1e-12, det, np.inf)
        delta = np.column_stack(
            (
                (j[:, 1, 1] * r[:, 0] - j[:, 0, 1] * r[:, 1]) / safe,
                (-j[:, 1, 0] * r[:, 0] + j[:, 0, 0] * r[:, 1]) / safe,
            )
        )
        delta = np.clip(delta, [-0.03, -0.1], [0.03, 0.1])
        best = p.copy()
        score = (r * r).sum(1)
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = np.clip(p - scale * delta, [0.0001, 0.05], [0.99, 1.0])
            err, *_ = evaluate(trial[:, 0], trial[:, 1])
            new = (err * err).sum(1)
            improve = new < score
            best[improve] = trial[improve]
            score = np.minimum(score, new)
        p = best
    r, a, s, t = evaluate(p[:, 0], p[:, 1])
    err = np.where(active, abs(s - t), 0).max(1)
    return p, r, a, err, _iteration


def _make_pd_schedules(
    lengths,
    controls,
    startup,
    esp,
    first_echo,
    *,
    t1=1.0,
    t2=0.1,
    decay=1.3,
    seed=None,
    return_levels=False,
):
    """Batch inverse-EPG shooting; columns are first/max/final degrees and plateau fraction."""
    lengths = np.asarray(lengths, dtype=int)
    controls = np.asarray(controls, dtype=float)
    if lengths.ndim != 1 or not len(lengths) or controls.shape != (len(lengths), 4):
        raise ValueError(
            "PD lengths and controls must have shapes (shots,) and (shots, 4)"
        )
    if np.any(lengths < startup + 2) or startup < 0:
        raise ValueError("PD trains need at least two acquired echoes after startup")
    if (
        not np.all(np.isfinite(controls))
        or np.any(controls[:, :3] <= 0)
        or np.any(controls[:, 1] >= 180)
        or np.any(controls[:, 0] > controls[:, 1])
        or np.any(controls[:, 2] > controls[:, 1])
        or np.any((controls[:, 3] <= 0) | (controls[:, 3] >= 1))
    ):
        raise ValueError(
            "PD controls require 0 < first/final <= maximum < 180 degrees and 0 < plateau_fraction < 1"
        )
    times = np.asarray([esp, first_echo, t1, t2, decay])
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError(
            "PD timing, reference T1/T2 and initial decay must be positive and finite"
        )
    unique_lengths, inverse = np.unique(lengths, return_inverse=True)
    if len(lengths) >= max(32, 4 * len(unique_lengths)):
        # A long parameter transition has few distinct integer ETLs. Solving
        # one representative per ETL gives nearby starts without a per-shot fit.
        representative = np.stack(
            [np.median(controls[lengths == n], axis=0) for n in unique_lengths]
        )
        levels, residual, _, error, _ = _solve_pd(
            unique_lengths, representative, startup, esp, first_echo, t1, t2, decay
        )
        if np.all(abs(residual) < 1e-5) and np.all(error < 1e-10):
            seed = levels[inverse]
    latent, residual, angles, error, _ = _solve_pd(
        lengths, controls, startup, esp, first_echo, t1, t2, decay, seed=seed
    )
    active = np.arange(angles.shape[1])[None, :] < lengths[:, None]
    first_signal = np.sin(np.deg2rad(controls[:, 0]) / 2) ** 2 * np.exp(
        -first_echo / t2
    )
    valid = (
        (abs(residual).max(1) < 1e-5)
        & (error < 1e-10)
        & (latent[:, 0] > 0)
        & (latent[:, 0] <= first_signal + 1e-10)
        & (latent[:, 1] > 0)
        & (latent[:, 1] <= 1 + 1e-10)
        & np.all(np.where(active, angles > 0, True), axis=1)
        & np.all(np.where(active, angles <= controls[:, 1, None] + 1e-5, True), axis=1)
    )
    result = np.deg2rad(np.where(active, angles, 0))
    if not np.all(valid):
        if seed is None:
            raise ValueError(
                "The prescribed PD controls do not give a feasible signal envelope and refocusing train"
            )
        # A remote warm start retries the same equations with a bracketed seed.
        rows = np.flatnonzero(~valid)
        fresh, levels = _make_pd_schedules(
            lengths[rows],
            controls[rows],
            startup,
            esp,
            first_echo,
            t1=t1,
            t2=t2,
            decay=decay,
            return_levels=True,
        )
        result[rows] = 0
        result[rows, : fresh.shape[1]] = fresh
        latent[rows] = levels
    return (result, latent) if return_levels else result


@lru_cache(maxsize=128)
def _pd_schedule(
    length, first, maximum, final, fraction, startup, esp, first_echo, t1, t2, decay
):
    result, levels = _make_pd_schedules(
        [length],
        [[first, maximum, final, fraction]],
        startup,
        esp,
        first_echo,
        t1=t1,
        t2=t2,
        decay=decay,
        return_levels=True,
    )
    result = result[0]
    levels = levels[0]
    result.flags.writeable = levels.flags.writeable = False
    return result, levels


def make_pd_schedule(
    length: int,
    echo_spacing: float,
    *,
    first_echo_time: float | None = None,
    first_flip_angle: float = np.deg2rad(120.0),
    maximum_flip_angle: float = np.deg2rad(140.0),
    final_flip_angle: float = np.deg2rad(100.0),
    plateau_fraction: float = 0.7,
    startup_echoes: int = 6,
    initial_decay: float = 1.3,
    reference_t1: float = 1.0,
    reference_t2: float = 0.1,
) -> np.ndarray:
    """Design a PD refocusing train by relaxation-aware inverse EPG.

    Parameters
    ----------
    length : int
        Total RF count, including startup; at least ``startup_echoes + 2``.
    echo_spacing : float
        Echo spacing (s), positive; shared by all trains of an acquisition.
    first_echo_time : float | None, default=None
        Excitation-to-first-echo time (s). ``None`` uses ``echo_spacing``.
    first_flip_angle : float, default=np.deg2rad(120.0)
        First refocusing angle (rad), independent of the peak and final angle.
    maximum_flip_angle : float, default=np.deg2rad(140.0)
        Maximum actual refocusing angle (rad), positive and below pi.
    final_flip_angle : float, default=np.deg2rad(100.0)
        Final actual refocusing angle (rad), positive and at most the maximum.
        Equality allows the RF maximum to extend to the final pulse.
    plateau_fraction : float, default=0.7
        Fraction of the acquired train before the gently declining signal
        tail begins, strictly between 0 and 1. This sets signal-tail onset;
        the actual RF peak position follows from EPG rather than interpolation.
    startup_echoes : int, default=6
        Leading unacquired pulses. This function emits angles, not ADC events.
    initial_decay : float, default=1.3
        Initial signal-transition exponential time constant (echoes).
    reference_t1 : float, default=1.0
        Reference T1 for inverse EPG (s), positive and finite.
    reference_t2 : float, default=0.1
        Reference T2 for inverse EPG (s), positive and finite.

    Returns
    -------
    numpy.ndarray
        Actual refocusing angles (rad), shape ``(length,)``, including startup.
        Each call returns an independent array.

    Raises
    ------
    ValueError
        If counts, controls or timing are invalid, or no feasible envelope
        realizes the prescribed maximum/final angles and initial signal drop.

    Notes
    -----
    The prescribed signal has an initial exponential drop, a nearly constant
    middle and a quadratic declining tail with zero slope at its onset,
    following the PD signal-evolution
    design principle of Mugler et al. [1]_. Two scalar signal levels are solved
    to enforce the actual maximum and final RF angles. Minimum and centre
    angles are outputs, without an imposed strict inequality. The peak may
    precede the final pulse. Accepted trains reproduce their target in-phase
    signal without pulse-by-pulse clipping. This is an independent design,
    not a numerical reconstruction of Siemens PDvar*.

    EPG assumes a fully relaxed ideal 90-degree excitation and instantaneous
    CPMG pulses with longitudinal recovery. Finite RF profiles, actual B1 and
    inter-shot steady state require separate forward evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> angles = pp.make_pd_schedule(26, 0.003416, first_echo_time=0.004162)
    >>> np.allclose(np.rad2deg(angles[[0, -1]]), [120, 100], atol=1e-4)
    True
    >>> bool(np.isclose(np.rad2deg(angles.max()), 140, atol=1e-4))
    True

    References
    ----------
    .. [1] Mugler JP III et al. Three-dimensional spin-echo-train
       proton-density-weighted imaging using shaped signal evolutions.
       Proceedings ISMRM. 1999; abstract 1631.
       https://cds.ismrm.org/ismrm-1999/PDF6/1631.pdf
    """
    try:
        if isinstance(length, (bool, np.bool_)) or isinstance(
            startup_echoes, (bool, np.bool_)
        ):
            raise TypeError
        length, startup = index(length), index(startup_echoes)
    except TypeError as error:
        raise ValueError("PD length and startup_echoes must be integers") from error
    first_echo = echo_spacing if first_echo_time is None else first_echo_time
    angles = np.rad2deg([first_flip_angle, maximum_flip_angle, final_flip_angle])
    return _pd_schedule(
        length,
        *angles,
        float(plateau_fraction),
        startup,
        echo_spacing,
        first_echo,
        reference_t1,
        reference_t2,
        initial_decay,
    )[0].copy()
