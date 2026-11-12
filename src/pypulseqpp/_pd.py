"""Relaxation-aware inverse-EPG PD trains with prescribed signal envelopes."""

from functools import lru_cache
from operator import index

import numpy as np

from pypulseqpp._busse import _trains


def _inverse_pd(target, first, esp, first_echo, t1=1.0, t2=0.1, tangents=None):
    """Flip angles (deg) and achieved echoes of prescribed relaxed echo signals.

    With ``tangents``, ``(directions, trains, echoes)`` changes of the target,
    the exact derivatives of the angles (deg) and echoes come back as well.
    """
    return _trains().prescribed_flips(
        target,
        first_flip=first,
        ESP=1e3 * esp,
        first_ESP=1e3 * first_echo,
        T1=1e3 * t1,
        T2=1e3 * t2,
        signal_tangents=tangents,
    )


class _Envelope:
    """PD signal envelopes of a batch of trains, and their derivatives.

    Each train's target is ``level * (1 + (ratio - 1) * z) + (initial - level)
    * exp(-echo / decay)``, with ``z`` the squared ramp of the tail that
    starts at the plateau fraction of the acquired train. The residuals are
    the highest acquired angle before the last less the maximum angle, and the
    last angle less the final angle, in degrees.
    """

    def __init__(self, lengths, controls, startup, esp, first_echo, t1, t2, decay):
        self.lengths = np.asarray(lengths)
        self.controls = np.asarray(controls)
        self.startup = startup
        self.timing = (esp, first_echo, t1, t2)
        self.echo = np.arange(self.lengths.max())[None, :]
        self.transient = np.exp(-self.echo / decay)
        first = self.controls[:, 0]
        self.initial = np.sin(np.deg2rad(first) / 2) ** 2 * np.exp(-first_echo / t2)

    def _tail(self, rows, fraction):
        """Return the tail ramp ``z`` and its derivative per unit plateau fraction."""
        lengths = self.lengths[rows, None]
        span = lengths - 1 - self.startup
        cut = self.startup + fraction[:, None] * span
        rest = lengths - 1 - cut
        ramp = (self.echo - cut) / rest
        inside = (ramp > 0) & (ramp < 1)
        clipped = np.clip(ramp, 0, 1)
        d_ramp = (self.echo - lengths + 1) / rest**2 * span
        return clipped * clipped, np.where(inside, 2 * clipped * d_ramp, 0.0)

    def target(self, rows, level, ratio, fraction=None):
        """Return targets and their derivatives.

        The targets are ``(trains, echoes)``; their derivatives along level,
        ratio and plateau fraction are ``(3, trains, echoes)``.
        """
        fraction = self.controls[rows, 3] if fraction is None else fraction
        z, d_z = self._tail(rows, fraction)
        level, ratio = level[:, None], ratio[:, None]
        target = (
            level * (1 + (ratio - 1) * z)
            + (self.initial[rows, None] - level) * self.transient
        )
        slopes = np.stack(
            (
                1 + (ratio - 1) * z - self.transient,
                level * z,
                level * (ratio - 1) * d_z,
            )
        )
        return target, slopes

    def residual(self, rows, angles):
        """Return the two residuals, and which echo each of them reads."""
        lengths = self.lengths[rows]
        each = np.arange(len(rows))
        mask = (self.echo < lengths[:, None] - 1) & (self.echo >= self.startup)
        highest = np.where(mask, angles, -np.inf).argmax(1)
        read = np.column_stack((highest, lengths - 1))
        controls = self.controls[rows]
        residual = np.column_stack(
            (
                angles[each, read[:, 0]] - controls[:, 1],
                angles[each, read[:, 1]] - controls[:, 2],
            )
        )
        return residual, read

    def solve(self, rows, level, ratio, directions=2):
        """Solve the trains ``rows`` with derivatives.

        Returns angles (deg), achieved echoes, targets, residuals, the
        residuals' derivatives ``(trains, 2, directions)`` and the angles'
        ``(directions, trains, echoes)``, along the first ``directions`` of
        level, ratio and plateau fraction.
        """
        target, slopes = self.target(rows, level, ratio)
        angles, achieved, d_angles, _ = _inverse_pd(
            target,
            self.controls[rows, 0],
            *self.timing,
            tangents=slopes[:directions],
        )
        residual, read = self.residual(rows, angles)
        each = np.arange(len(rows))
        jacobian = np.stack(
            (d_angles[:, each, read[:, 0]].T, d_angles[:, each, read[:, 1]].T),
            axis=1,
        )
        return angles, achieved, target, residual, jacobian, d_angles

    def values(self, rows, level, ratio):
        """Return residuals, angles (deg), achieved echoes and targets."""
        target, _ = self.target(rows, level, ratio)
        angles, achieved = _inverse_pd(target, self.controls[rows, 0], *self.timing)
        return self.residual(rows, angles)[0], angles, achieved, target


def _solve_pd(lengths, controls, startup, esp, first_echo, t1, t2, decay, seed=None):
    lengths = np.asarray(lengths)
    count = len(lengths)
    rows = np.arange(count)
    envelope = _Envelope(lengths, controls, startup, esp, first_echo, t1, t2, decay)
    active = envelope.echo < lengths[:, None]
    ratio = 0.65 * np.exp(-0.007 * (lengths - startup))
    if seed is None:
        lo = np.full(count, 0.001)
        hi = np.full(count, 0.99)
        for _ in range(34):
            mid = (lo + hi) / 2
            r, *_ = envelope.values(rows, mid, ratio)
            lo = np.where(r[:, 0] < 0, mid, lo)
            hi = np.where(r[:, 0] >= 0, mid, hi)
        level = (lo + hi) / 2
        p = np.column_stack((level, ratio))
    else:
        p = np.asarray(seed, dtype=float).copy()
    for _iteration in range(16):
        *_, r, j, _ = envelope.solve(rows, p[:, 0], p[:, 1])
        if abs(r).max(initial=0) < 1e-6:
            break
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
            err, *_ = envelope.values(rows, trial[:, 0], trial[:, 1])
            new = (err * err).sum(1)
            improve = new < score
            best[improve] = trial[improve]
            score = np.minimum(score, new)
        p = best
    r, a, s, t = envelope.values(rows, p[:, 0], p[:, 1])
    err = np.where(active, abs(s - t), 0).max(1)
    return p, r, a, err, _iteration


def _pd_fraction_slopes(
    length, controls, levels, startup, esp, first_echo, t1, t2, decay
):
    """Angles (rad) of a solved PD train and their derivative per unit plateau fraction.

    The two signal levels keep the maximum and final angles as the fraction
    moves, so by the implicit function theorem they move by ``-R_p^-1 R_f``,
    with ``R`` the two residuals; the angles move by what the fraction does
    directly plus what the levels do.
    """
    envelope = _Envelope([length], [controls], startup, esp, first_echo, t1, t2, decay)
    rows = np.array([0])
    angles, _, _, _, jacobian, d_angles = envelope.solve(
        rows, levels[:1], levels[1:], directions=3
    )
    moved = -np.linalg.solve(jacobian[0, :, :2], jacobian[0, :, 2])
    slope = d_angles[2, 0] + moved @ d_angles[:2, 0]
    played = envelope.echo[0] < length
    return np.deg2rad(np.where(played, angles[0], 0.0)), np.deg2rad(
        np.where(played, slope, 0.0)
    )


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
    ImportError
        If blochsim, which inverts the phase graph, is not installed.

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


#: The parameters of a PD target signal, in the order a control row holds them.
PD_TARGET_PARAMETERS = (
    "first_angle_deg",
    "plateau_signal",
    "plateau_slope",
    "plateau_fraction",
    "tail_signal",
    "tail_exponent",
    "initial_signal",
    "initial_decay",
)

#: Target-signal parameters of the central and the peripheral train, fitted
#: (``benchmarks/pd_buonincontri_fit.py``) so that inverse EPG at 3.416 ms
#: spacing, 4.162 ms to the first echo, reproduces the trains read off Fig. 1
#: of Buonincontri et al. (Proc. ISMRM 2025, abstract 566-05-007), the only
#: reference they were fitted to (``tests/fixtures/buonincontri_figure1.json``).
PD_CENTRE = (91.012, 0.371, 0.354, 0.662, 0.666, 0.842, 0.663, 0.861)
PD_PERIPHERY = (97.163, 0.508, 0.091, 0.666, 0.69, 0.907, 0.78, 0.425)

#: Default upper bound on every refocusing angle of a target-signal train
#: (degrees).
PD_MAXIMUM_ANGLE_DEG = 150.0


def _pd_targets(lengths, controls, jacobian=False):
    """Target in-phase signal of each train, ``(trains, max(lengths))``.

    Echo ``x`` of a train of ``n`` pulses, at ``u = x / (n - 1)`` along it,
    has the plateau ``plateau_signal * (1 + plateau_slope * u)``, lowered
    past ``u = plateau_fraction`` by ``(1 - tail_signal) * v ** tail_exponent``
    of itself, ``v`` the fraction of the tail gone, plus an initial excess
    ``initial_signal - plateau_signal`` decaying with ``initial_decay`` echoes
    from the second echo. With ``jacobian``, the derivatives of the targets
    along the seven signal parameters come back too, ``(7, trains, echoes)``.
    """
    lengths = np.asarray(lengths, dtype=int)
    c = np.asarray(controls, dtype=float)
    x = np.arange(lengths.max())[None, :]
    u = np.minimum(x / (lengths[:, None] - 1), 1.0)
    level, slope, onset, tail, power, start, decay = (
        c[:, i, None] for i in range(1, 8)
    )
    v = np.clip((u - onset) / (1 - onset), 0.0, 1.0)
    inside = (u > onset) & (v < 1.0)
    vp = v**power
    shape = 1 - (1 - tail) * vp
    ramp = 1 + slope * u
    initial = np.exp(-(x - 1) / decay)
    target = level * ramp * shape + (start - level) * initial
    if not jacobian:
        return target
    with np.errstate(divide="ignore", invalid="ignore"):
        log_v = np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), 0.0)
        d_v_onset = np.where(inside, (u - 1) / (1 - onset) ** 2, 0.0)
        d_shape_v = np.where(v > 0, -(1 - tail) * power * v ** (power - 1), 0.0)
    plateau = level * ramp
    slopes = np.stack(
        np.broadcast_arrays(
            ramp * shape - initial,
            level * u * shape,
            plateau * d_shape_v * d_v_onset,
            plateau * vp,
            -plateau * (1 - tail) * vp * log_v,
            initial,
            (start - level) * initial * (x - 1) / decay**2,
        )
    )
    return target, slopes


def _check_pd_targets(lengths, controls, maximum, times):
    """Refuse counts, target parameters, bounds or timing out of range."""
    if lengths.ndim != 1 or not len(lengths) or controls.shape != (len(lengths), 8):
        raise ValueError(
            "PD lengths and controls must have shapes (trains,) and (trains, 8)"
        )
    if np.any(lengths < 2):
        raise ValueError("PD trains need at least two pulses")
    first, level, _, onset, tail, power, start, decay = controls.T
    if (
        not np.all(np.isfinite(controls))
        or np.any((maximum <= 0) | (maximum >= 180))
        or np.any((first <= 0) | (first > maximum))
        or np.any((level <= 0) | (level > 1) | (start <= 0) | (start > 1))
        or np.any((onset <= 0) | (onset >= 1))
        or np.any((tail < 0) | (power <= 0) | (decay <= 0))
    ):
        raise ValueError(
            "PD controls require 0 < first angle <= maximum < 180 degrees, plateau "
            "and initial signals in (0, 1], a tail onset in (0, 1), and a "
            "nonnegative tail signal with positive exponent and decay"
        )
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("PD timing and reference T1/T2 must be positive and finite")


def _pd_target_trains(
    lengths, controls, maximum_deg, esp, first_echo, t1=1.0, t2=0.1, jacobian=False
):
    """Design one target-signal PD train per row of ``controls``, in one pass.

    Returns the refocusing angles (rad), zero past each train's own length,
    and the in-phase signal they reach, both ``(trains, max(lengths))``. A
    pulse the target would push past ``maximum_deg`` plays that angle. With
    ``jacobian``, the exact derivatives of the angles (rad) along the eight
    target parameters come back as well, ``(8, trains, max(lengths))``.
    """
    lengths = np.asarray(lengths, dtype=int)
    controls = np.asarray(controls, dtype=float)
    maximum = np.broadcast_to(np.asarray(maximum_deg, dtype=float), lengths.shape)
    times = np.asarray([esp, first_echo, t1, t2], dtype=float)
    _check_pd_targets(lengths, controls, maximum, times)
    arguments = {
        "first_flip": controls[:, 0],
        "maximum_flip": maximum,
        "ESP": 1e3 * esp,
        "first_ESP": 1e3 * first_echo,
        "T1": 1e3 * t1,
        "T2": 1e3 * t2,
        "lengths": lengths,
    }
    if not jacobian:
        angles, signal = _trains().prescribed_flips(
            _pd_targets(lengths, controls), **arguments
        )
    else:
        target, slopes = _pd_targets(lengths, controls, jacobian=True)
        count = len(lengths)
        signal_tangents = np.concatenate([np.zeros((1, *target.shape)), slopes])
        first_tangents = np.zeros((8, count))
        first_tangents[0] = 1.0
        angles, signal, d_angles, _ = _trains().prescribed_flips(
            target,
            signal_tangents=signal_tangents,
            first_flip_tangents=first_tangents,
            **arguments,
        )
    active = np.arange(angles.shape[1])[None, :] < lengths[:, None]
    result = np.deg2rad(np.where(active, angles, 0.0)), np.where(active, signal, 0.0)
    if jacobian:
        return *result, np.deg2rad(np.where(active, d_angles, 0.0))
    return result


def make_pd_target_schedule(
    length: int,
    echo_spacing: float,
    *,
    first_echo_time: float | None = None,
    first_flip_angle: float = np.deg2rad(PD_CENTRE[0]),
    maximum_flip_angle: float = np.deg2rad(PD_MAXIMUM_ANGLE_DEG),
    plateau_signal: float = PD_CENTRE[1],
    plateau_slope: float = PD_CENTRE[2],
    plateau_fraction: float = PD_CENTRE[3],
    tail_signal: float = PD_CENTRE[4],
    tail_exponent: float = PD_CENTRE[5],
    initial_signal: float = PD_CENTRE[6],
    initial_decay: float = PD_CENTRE[7],
    reference_t1: float = 1.0,
    reference_t2: float = 0.1,
) -> np.ndarray:
    """Design a PD refocusing train by relaxation-aware inverse EPG of a target signal.

    Parameters
    ----------
    length : int
        RF count, startup pulses included; at least 2.
    echo_spacing : float
        Echo spacing (s), positive.
    first_echo_time : float | None, default=None
        Excitation-to-first-echo time (s). ``None`` uses ``echo_spacing``.
    first_flip_angle : float, default=np.deg2rad(PD_CENTRE[0])
        First refocusing angle (rad), at most ``maximum_flip_angle``.
    maximum_flip_angle : float, default=np.deg2rad(PD_MAXIMUM_ANGLE_DEG)
        Upper bound on every angle (rad), below pi. A pulse the target would
        push past it plays it, and its echo falls below the target.
    plateau_signal : float, default=PD_CENTRE[1]
        Plateau of the target signal, over the equilibrium magnetization, in
        (0, 1].
    plateau_slope : float, default=PD_CENTRE[2]
        Relative change of the plateau from the first pulse to the last.
    plateau_fraction : float, default=PD_CENTRE[3]
        Fraction of the train, from its first pulse, before the signal tail
        begins, in (0, 1).
    tail_signal : float, default=PD_CENTRE[4]
        Signal at the last pulse over the plateau there, nonnegative.
    tail_exponent : float, default=PD_CENTRE[5]
        Power of the tail's decline in its own fraction of the train,
        positive: 2 starts it with zero slope.
    initial_signal : float, default=PD_CENTRE[6]
        Target at the second echo, in (0, 1], from which the signal decays to
        the plateau.
    initial_decay : float, default=PD_CENTRE[7]
        Time constant of that decay (echoes), positive.
    reference_t1 : float, default=1.0
        Reference T1 for inverse EPG (s), positive and finite.
    reference_t2 : float, default=0.1
        Reference T2 for inverse EPG (s), positive and finite.

    Returns
    -------
    numpy.ndarray
        Refocusing angles (rad), shape ``(length,)``.

    Raises
    ------
    ValueError
        If the count, the timing or a parameter is out of range.
    ImportError
        If blochsim, which inverts the phase graph, is not installed.

    Notes
    -----
    The target signal decreases from the second echo to a plateau and
    declines over a late tail, after the prescribed-signal design of
    Mugler et al. [1]_. Each angle is the one in ``[0, maximum_flip_angle]``
    whose echo comes nearest the target, given the train so far
    (:func:`blochsim.trains.prescribed_flips`). The default parameters are
    fitted to the central train of Fig. 1 of Buonincontri et al. [2]_, read
    off the published figure; ``PD_PERIPHERY`` holds those of its peripheral
    train. They are the only reference the defaults were fitted to, and this
    is not a numerical reconstruction of the trains of that figure.

    :func:`make_pd_schedule` designs a PD train from its first, maximum and
    final angles instead, solving the signal levels that reach them.

    EPG assumes a fully relaxed ideal 90-degree excitation and instantaneous
    CPMG pulses with longitudinal recovery. Finite RF profiles, actual B1 and
    inter-shot steady state require separate forward evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> angles = np.rad2deg(
    ...     pp.make_pd_target_schedule(40, 0.003416, first_echo_time=0.004162)
    ... )
    >>> bool(angles.max() <= 150.0), bool(angles[4] < angles[-1])
    (True, True)

    References
    ----------
    .. [1] Mugler JP III et al. Three-dimensional spin-echo-train
       proton-density-weighted imaging using shaped signal evolutions.
       Proceedings ISMRM. 1999; abstract 1631.
       https://cds.ismrm.org/ismrm-1999/PDF6/1631.pdf
    .. [2] Buonincontri G et al. Doubling the repetition time without paying
       the price: 3D Turbo Spin Echo with individually parameterized echo
       trains. Proceedings ISMRM. 2025; abstract 566-05-007.
    """
    try:
        if isinstance(length, (bool, np.bool_)):
            raise TypeError
        length = index(length)
    except TypeError as error:
        raise ValueError("PD length must be an integer") from error
    first_echo = echo_spacing if first_echo_time is None else first_echo_time
    controls = [
        np.rad2deg(first_flip_angle),
        plateau_signal,
        plateau_slope,
        plateau_fraction,
        tail_signal,
        tail_exponent,
        initial_signal,
        initial_decay,
    ]
    angles, _ = _pd_target_trains(
        [length],
        [controls],
        np.rad2deg(maximum_flip_angle),
        echo_spacing,
        first_echo,
        reference_t1,
        reference_t2,
    )
    return angles[0]
