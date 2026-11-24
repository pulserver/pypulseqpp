"""How many repetitions bring a spoiled or balanced steady state within a tolerance."""

from __future__ import annotations

__all__ = [
    "STEADY_STATE_B1",
    "STEADY_STATE_BALANCED_T1_S",
    "STEADY_STATE_T1_S",
    "STEADY_STATE_T2_S",
    "steady_state_dummies",
]

import math

#: T1 a spoiled steady state's dummies are counted for (s): CSF at 7 T
#: (Rooney et al., Magn Reson Med 57:308, 2007), the longest of the head's
#: tissues at the highest field the sequences target.
STEADY_STATE_T1_S = 4.4

#: Fraction of the nominal flip a spoiled steady state's dummies are counted
#: at: the transmit field of a CP-mode head coil at 7 T in the temporal lobes
#: and cerebellum, where a lower flip approaches the steady state more slowly.
STEADY_STATE_B1 = 0.5

#: T1 a balanced steady state's dummies are counted for (s): grey matter at
#: 3 T.
STEADY_STATE_BALANCED_T1_S = 1.5

#: T2 a balanced steady state's dummies are counted for (s): grey matter at
#: 3 T. The approach is slower the longer the T2.
STEADY_STATE_T2_S = 0.1


def steady_state_dummies(
    repetition_s: float,
    flip_angle_deg: float,
    *,
    t1_s: float | None = None,
    t2_s: float | None = None,
    tolerance: float = 0.01,
    b1: float | None = None,
) -> int:
    """Return the dummy repetitions that bring a spoiled or balanced steady state within ``tolerance``.

    The count is of repetitions played before acquiring. From equilibrium,
    the distance of the longitudinal magnetisation to a spoiled steady state
    shrinks by ``E1 cos(b1 flip)`` each repetition, ``E1 = exp(-repetition_s /
    t1_s)``; the count leaves at most ``tolerance`` of the steady-state
    magnetisation ``(1 - E1) / (1 - E1 cos(b1 flip))``, so of the signal, and
    a flip of 90 degrees or more reaches it after one. ``b1`` scales the flip
    to where the transmit field is weakest. The transverse
    coherences RF spoiling leaves in a tissue of long T2 are not counted.
    Given ``t2_s``, the count is for a balanced (bSSFP) steady state reached
    from an alpha/2 catalyst, whose distance shrinks by ``E1 cos^2(flip / 2)
    + E2 sin^2(flip / 2)``, ``E2 = exp(-repetition_s / t2_s)`` (Scheffler,
    Magn Reson Med 2003), to ``tolerance`` of the distance from the
    catalyst. At least one is always played.

    Parameters
    ----------
    repetition_s : float
        Time between two excitations of the same magnetisation (s).
    flip_angle_deg : float
        Excitation flip angle (degrees).
    t1_s : float | None, default=None
        T1 the steady state is reached for (s); :data:`STEADY_STATE_T1_S`
        for a spoiled one and :data:`STEADY_STATE_BALANCED_T1_S` for a
        balanced one without one.
    t2_s : float | None, default=None
        T2 a balanced steady state is reached for (s), as
        :data:`STEADY_STATE_T2_S`; ``None`` counts for a spoiled one.
    tolerance : float, default=0.01
        Distance to the steady state left: a fraction of the steady-state
        magnetisation for a spoiled one, of the distance from the catalyst
        for a balanced one.
    b1 : float | None, default=None
        Transmit field, as a fraction of the nominal flip, a spoiled steady
        state is counted at; :data:`STEADY_STATE_B1` without one. Not used
        for a balanced one.

    Raises
    ------
    ValueError
        If a time, ``b1`` or the tolerance is not positive, or the tolerance
        is not below one.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> sequences.steady_state_dummies(5e-3, 12.0)
    932
    >>> sequences.steady_state_dummies(0.5, 90.0, b1=1.0)
    1
    >>> sequences.steady_state_dummies(3.5e-3, 60.0, t2_s=0.1)
    443
    """
    if t1_s is None:
        t1_s = STEADY_STATE_T1_S if t2_s is None else STEADY_STATE_BALANCED_T1_S
    if b1 is None:
        b1 = STEADY_STATE_B1
    if (
        b1 <= 0
        or repetition_s <= 0
        or t1_s <= 0
        or (t2_s is not None and t2_s <= 0)
        or not 0 < tolerance < 1
    ):
        raise ValueError(
            "repetition_s, t1_s, t2_s and b1 must be positive and tolerance in (0, 1)"
        )
    e1 = math.exp(-repetition_s / t1_s)
    half = math.radians(flip_angle_deg) / 2
    if t2_s is None:
        factor = e1 * math.cos(2 * b1 * half)
        steady = (1 - e1) / (1 - factor)
        left = tolerance * steady / (1 - steady)
    else:
        e2 = math.exp(-repetition_s / t2_s)
        factor = e1 * math.cos(half) ** 2 + e2 * math.sin(half) ** 2
        left = tolerance
    if factor <= left:
        return 1
    return max(1, math.ceil(math.log(left) / math.log(factor)))
