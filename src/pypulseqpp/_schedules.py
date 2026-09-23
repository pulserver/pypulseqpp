"""Per-repetition RF phase and refocusing flip-angle schedules.

Each routine returns one value per repetition (or per echo), which the scan
loop applies to the RF and ADC events of that repetition. They select no
views and emit no labels.
"""

from __future__ import annotations

__all__ = [
    "make_phase_cycling_schedule",
    "make_rf_spoiling_schedule",
    "make_traps_schedule",
]

from collections.abc import Sequence

import numpy as np


def make_rf_spoiling_schedule(
    length: int,
    *,
    increment: float = np.deg2rad(117.0),
    initial_phase: float = 0.0,
    initial_increment: float = 0.0,
) -> np.ndarray:
    """Return quadratic RF-spoiling phases in radians.

    Apply each phase to both excitation and ADC offsets. With the default
    initial increment, the first two phases are equal.

    Parameters
    ----------
    length : int
        Number of repetitions.
    increment : float, default=np.deg2rad(117.0)
        Quadratic phase increment (rad); 117 degrees by default.
    initial_phase : float, default=0.0
        Phase of the first repetition (rad).
    initial_increment : float, default=0.0
        Linear increment at the first repetition (rad).

    Returns
    -------
    numpy.ndarray
        Phases (rad) in ``[0, 2 pi)``, length ``length``.

    Raises
    ------
    ValueError
        If ``length`` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> np.rad2deg(pp.make_rf_spoiling_schedule(4)).round(1)
    array([  0.,   0., 117., 351.])

    See Also
    --------
    make_phase_cycling_schedule : the balanced-SSFP alternative.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    phases = np.empty(length, dtype=float)
    phase = float(initial_phase)
    phase_increment = float(initial_increment)
    for index in range(length):
        phases[index] = phase % (2.0 * np.pi)
        phase = (phase + phase_increment) % (2.0 * np.pi)
        phase_increment = (phase_increment + increment) % (2.0 * np.pi)
    return phases


def make_phase_cycling_schedule(
    length: int,
    phases: Sequence[float] = (0.0, np.pi),
) -> np.ndarray:
    """Repeat a phase cycle, reducing each entry modulo 2*pi.

    Parameters
    ----------
    length : int
        Number of repetitions to fill.
    phases : sequence of float, default=(0.0, np.pi)
        The cycle to repeat (rad); ``(0, pi)`` by default.

    Returns
    -------
    numpy.ndarray
        Phases (rad) in ``[0, 2 pi)``, length ``length``.

    Raises
    ------
    ValueError
        If ``length`` is negative, or ``phases`` is not a non-empty
        one-dimensional finite sequence.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> np.rad2deg(pp.make_phase_cycling_schedule(5))
    array([  0., 180.,   0., 180.,   0.])
    >>> np.rad2deg(pp.make_phase_cycling_schedule(4, (0.0, np.pi / 2)))
    array([ 0., 90.,  0., 90.])

    See Also
    --------
    make_rf_spoiling_schedule : for spoiled, non-steady-state sequences.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    cycle = np.asarray(phases, dtype=float)
    if cycle.ndim != 1 or cycle.size == 0 or not np.all(np.isfinite(cycle)):
        raise ValueError("phases must be a non-empty one-dimensional finite sequence")
    return np.resize(cycle, length) % (2.0 * np.pi)


def make_traps_schedule(
    length: int,
    target_flip_angle: float,
    *,
    variable: bool = True,
) -> np.ndarray:
    """Return a variable refocusing flip-angle schedule in radians.

    ``variable=False`` gives a constant target angle. Otherwise the first
    refocusing angle is raised above the target, approximately
    ``pi / 2 + target / 2`` as proposed for pseudo-steady-state
    stabilisation [1]_, and the following angles decay towards the target by
    a factor of two per echo. Flip-angle trains that vary smoothly along the
    echo train are the subject of TRAPS [2]_; this routine implements only the
    initial stabilising transition.

    Parameters
    ----------
    length : int
        Echo train length (>= 1).
    target_flip_angle : float
        Asymptotic refocusing flip angle (rad), positive.
    variable : bool, default=True
        Sweep down to the target (default) instead of holding it constant.

    Returns
    -------
    numpy.ndarray
        Refocusing flip angles (rad), length ``length``, approaching
        ``target_flip_angle``.

    Raises
    ------
    ValueError
        If ``length`` is negative, or a flip angle is outside the range a
        refocusing train can use.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> flips = pp.make_traps_schedule(8, np.deg2rad(120))
    >>> np.rad2deg(flips)[[0, -1]].round(1)
    array([153. , 120.2])

    References
    ----------
    .. [1] Alsop DC. The sensitivity of low flip angle RARE imaging.
       *Magnetic Resonance in Medicine*. 1997;37(2):176-184.
       https://doi.org/10.1002/mrm.1910370206
    .. [2] Hennig J, Weigel M, Scheffler K. Multiecho sequences with variable
       refocusing flip angles: optimization of signal behavior using smooth
       transitions between pseudo steady states (TRAPS). *Magnetic Resonance
       in Medicine*. 2003;49(3):527-535. https://doi.org/10.1002/mrm.10391
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if not np.isfinite(target_flip_angle) or target_flip_angle <= 0:
        raise ValueError("target_flip_angle must be a positive finite angle")
    if not variable or length == 1:
        return np.full(length, target_flip_angle, dtype=float)

    first = (
        np.pi / 2.0
        + target_flip_angle / 2.0
        + 0.4 * ((2.0 - 1.0) / 2.0) ** 2.0 * (np.pi / 2.0 - target_flip_angle / 2.0)
    )
    delta = first - target_flip_angle
    echo = np.arange(2, length + 1, dtype=float)
    return np.concatenate(([first], target_flip_angle + delta / (2.0 ** (echo - 0.5))))
