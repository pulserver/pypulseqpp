"""Relaxation-aware inverse-EPG PD trains with prescribed signal envelopes."""

from operator import index

import numpy as np

from pypulseqpp._busse import _rotate, _shift


def _inverse_pd(target, first, esp, first_echo, t1=1.0, t2=0.1, maximum=180.0):
    count, n = target.shape
    maximum = np.broadcast_to(np.asarray(maximum, dtype=float), (count,))
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
                    np.minimum(np.deg2rad(maximum), np.pi),
                )
            )
            err = abs(
                (v[:, None] + b[:, None] * np.cos(cs) + c[:, None] * np.sin(cs)) * e2
                - target[:, i, None]
            )
            err[(cs < 0) | (cs > np.minimum(np.deg2rad(maximum), np.pi)[:, None])] = (
                np.inf
            )
            a = cs[rows, np.argmin(err + 1e-12 * cs, axis=1)]
        s = _rotate(s, a[:, None])
        s[:2] *= e2
        s[2] *= e1
        s[2, :, 0] += 1 - e1
        _shift(s)
        angles[:, i] = np.rad2deg(a)
        signal[:, i] = s[0, :, 0].real
    return angles, signal


#: The parameters of a PD target signal, in the order a control row holds them.
PD_PARAMETERS = (
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

#: Default upper bound on every refocusing angle (degrees).
PD_MAXIMUM_ANGLE_DEG = 150.0


def _pd_targets(lengths, controls):
    """Return the target in-phase signal of each train, ``(trains, max(lengths))``.

    Echo ``x`` of a train of ``n`` pulses, at ``u = x / (n - 1)`` along it,
    has the plateau ``plateau_signal * (1 + plateau_slope * u)``, lowered
    past ``u = plateau_fraction`` by ``(1 - tail_signal) * v ** tail_exponent``
    of itself, ``v`` the fraction of the tail gone, plus an initial excess
    ``initial_signal - plateau`` decaying with ``initial_decay`` echoes from
    the second echo. The first echo is the first angle's own.
    """
    lengths = np.asarray(lengths, dtype=int)
    c = np.asarray(controls, dtype=float)
    x = np.arange(lengths.max())[None, :]
    u = np.minimum(x / (lengths[:, None] - 1), 1.0)
    level, slope, onset, tail, power, start, decay = (
        c[:, i, None] for i in range(1, 8)
    )
    v = np.clip((u - onset) / (1 - onset), 0.0, 1.0)
    plateau = level * (1 + slope * u) * (1 - (1 - tail) * v**power)
    return plateau + (start - level) * np.exp(-(x - 1) / decay)


def _pd_trains(lengths, controls, maximum_deg, esp, first_echo, t1=1.0, t2=0.1):
    """Design one PD train per row of ``controls`` by inverse EPG, in one pass.

    Returns the refocusing angles (rad), zero past each train's own length,
    and the in-phase signal they reach, both ``(trains, max(lengths))``. A
    pulse the target would push past ``maximum_deg`` plays that angle.
    """
    lengths = np.asarray(lengths, dtype=int)
    controls = np.asarray(controls, dtype=float)
    maximum = np.broadcast_to(np.asarray(maximum_deg, dtype=float), lengths.shape)
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
    times = np.asarray([esp, first_echo, t1, t2])
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("PD timing and reference T1/T2 must be positive and finite")
    angles, signal = _inverse_pd(
        _pd_targets(lengths, controls), first, esp, first_echo, t1, t2, maximum
    )
    active = np.arange(angles.shape[1])[None, :] < lengths[:, None]
    return np.deg2rad(np.where(active, angles, 0.0)), np.where(active, signal, 0.0)


def make_pd_schedule(
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

    Notes
    -----
    The target signal decreases from the second echo to a plateau and
    declines over a late tail, after the prescribed-signal design of
    Mugler et al. [1]_. Each angle is the one in ``[0, maximum_flip_angle]``
    whose echo comes nearest the target, given the train so far. The default
    parameters are fitted to the central train of Fig. 1 of Buonincontri et
    al. [2]_, read off the published figure; ``PD_PERIPHERY`` holds those of
    its peripheral train. They are the only reference the defaults were
    fitted to, and this is not a numerical reconstruction of the PDvar* trains
    of that figure.

    EPG assumes a fully relaxed ideal 90-degree excitation and instantaneous
    CPMG pulses with longitudinal recovery. Finite RF profiles, actual B1 and
    inter-shot steady state require separate forward evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> angles = np.rad2deg(pp.make_pd_schedule(40, 0.003416, first_echo_time=0.004162))
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
    angles, _ = _pd_trains(
        [length],
        [controls],
        np.rad2deg(maximum_flip_angle),
        echo_spacing,
        first_echo,
        reference_t1,
        reference_t2,
    )
    return angles[0]
