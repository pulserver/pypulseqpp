"""Prescribed-PSS refocusing schedules, designed by blochsim's train kernels."""

from __future__ import annotations

from functools import lru_cache
from operator import index

import numpy as np


def _trains():
    """``blochsim.trains``, imported when a schedule is designed.

    The module is NumPy over a compiled kernel; importing it does not import
    PyTorch.
    """
    try:
        from blochsim import trains
    except ImportError as error:
        raise ImportError(
            "Busse and PD refocusing schedules and the FSE effective TE are "
            "computed by blochsim>=0.0.18: pip install 'pypulseqpp[design]'"
        ) from error
    return trains


@lru_cache(maxsize=128)
def _schedule(length, minimum_deg, centre_deg, maximum_deg, centre, decay):
    try:
        angles = _trains().busse_flips(
            length,
            min_flip=minimum_deg,
            centre_flip=centre_deg,
            max_flip=maximum_deg,
            centre_echo=centre,
            startup_echoes=0,
            initial_decay=decay,
        )
    except ValueError as error:
        raise ValueError(
            "The prescribed Busse controls do not give a feasible refocusing train"
        ) from error
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

    The train is designed by ``blochsim.trains.busse_flips``, which the
    ``design`` extra installs. The first and final pulses use the maximum angle. An initial asymptotic
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
    ImportError
        If blochsim is not installed.

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


def _make_busse_schedules(lengths, controls_deg, centre, decay):
    """Design variable trains in one batch; controls are actual RF degrees.

    Every row uses the same prescribed-PSS family as make_busse_schedule.
    Lengths include startup. Padding after each row's final pulse is zero.
    """
    lengths = np.asarray(lengths, dtype=int)
    requested = np.asarray(controls_deg, dtype=float)
    if np.any(lengths <= centre + 1) or centre <= 5:
        raise ValueError("The Busse centre must precede the final pulse of every train")
    try:
        angles = _trains().busse_flips(
            lengths,
            min_flip=requested[:, 0],
            centre_flip=requested[:, 1],
            max_flip=requested[:, 2],
            centre_echo=int(centre),
            startup_echoes=0,
            initial_decay=decay,
        )
    except ValueError as error:
        raise ValueError(
            "The prescribed Busse controls do not give a feasible refocusing train"
        ) from error
    return np.deg2rad(angles)
