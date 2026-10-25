"""How many repetitions bring a spoiled or balanced steady state within a tolerance."""

from __future__ import annotations

__all__ = ["STEADY_STATE_T1_S", "STEADY_STATE_T2_S", "steady_state_dummies"]

import math

#: T1 the dummies are counted for (s): grey matter at 3 T, among the longest
#: of brain tissue but CSF's.
STEADY_STATE_T1_S = 1.5

#: T2 a balanced steady state's dummies are counted for (s): grey matter at
#: 3 T. The approach is slower the longer the T2.
STEADY_STATE_T2_S = 0.1


def steady_state_dummies(
    repetition_s: float,
    flip_angle_deg: float,
    *,
    t1_s: float = STEADY_STATE_T1_S,
    t2_s: float | None = None,
    tolerance: float = 0.01,
) -> int:
    """Return the dummy repetitions that bring a spoiled or balanced steady state within ``tolerance``.

    The count is of repetitions played before acquiring, so that the
    magnetisation is within ``tolerance`` of its steady state. From
    equilibrium, the distance to a spoiled steady state shrinks by
    ``exp(-repetition_s / t1_s) * cos(flip)`` each repetition. A flip of 90
    degrees or more reaches it after one. Given ``t2_s``, the count is for a
    balanced (bSSFP) steady state reached from an alpha/2 catalyst, whose
    distance shrinks by ``E1 cos^2(flip / 2) + E2 sin^2(flip / 2)``, with
    ``E1, E2 = exp(-repetition_s / t1_s), exp(-repetition_s / t2_s)``
    (Scheffler, Magn Reson Med 2003). At least one is always played.

    Parameters
    ----------
    repetition_s : float
        Time between two excitations of the same magnetisation (s).
    flip_angle_deg : float
        Excitation flip angle (degrees).
    t1_s : float, default=STEADY_STATE_T1_S
        T1 the steady state is reached for (s).
    t2_s : float | None, default=None
        T2 a balanced steady state is reached for (s), as
        :data:`STEADY_STATE_T2_S`; ``None`` counts for a spoiled one.
    tolerance : float, default=0.01
        Fraction of the distance from equilibrium to the steady state left.

    Raises
    ------
    ValueError
        If a time or the tolerance is not positive, or the tolerance is not
        below one.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> sequences.steady_state_dummies(5e-3, 12.0)
    182
    >>> sequences.steady_state_dummies(0.5, 90.0)
    1
    >>> sequences.steady_state_dummies(3.5e-3, 60.0, t2_s=0.1)
    443
    """
    if (
        repetition_s <= 0
        or t1_s <= 0
        or (t2_s is not None and t2_s <= 0)
        or not 0 < tolerance < 1
    ):
        raise ValueError(
            "repetition_s, t1_s and t2_s must be positive and tolerance in (0, 1)"
        )
    e1 = math.exp(-repetition_s / t1_s)
    half = math.radians(flip_angle_deg) / 2
    if t2_s is None:
        factor = e1 * math.cos(2 * half)
    else:
        e2 = math.exp(-repetition_s / t2_s)
        factor = e1 * math.cos(half) ** 2 + e2 * math.sin(half) ** 2
    if factor <= tolerance:
        return 1
    return max(1, math.ceil(math.log(tolerance) / math.log(factor)))
