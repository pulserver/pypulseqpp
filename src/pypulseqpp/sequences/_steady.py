"""How many repetitions bring a spoiled steady state within a tolerance."""

from __future__ import annotations

__all__ = ["STEADY_STATE_T1_S", "steady_state_dummies"]

import math

#: T1 the dummies are counted for (s): grey matter at 3 T, among the longest
#: of brain tissue but CSF's.
STEADY_STATE_T1_S = 1.5


def steady_state_dummies(
    repetition_s: float,
    flip_angle_deg: float,
    *,
    t1_s: float = STEADY_STATE_T1_S,
    tolerance: float = 0.01,
) -> int:
    """Return the repetitions to play before acquiring, so the longitudinal magnetization is within ``tolerance`` of its spoiled steady state.

    From equilibrium, the distance to the steady state shrinks by
    ``exp(-repetition_s / t1_s) * cos(flip)`` each repetition. A flip of 90
    degrees or more reaches it after one; at least one is always played.

    Parameters
    ----------
    repetition_s : float
        Time between two excitations of the same magnetization (s).
    flip_angle_deg : float
        Excitation flip angle (degrees).
    t1_s : float, default=STEADY_STATE_T1_S
        T1 the steady state is reached for (s).
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
    """
    if repetition_s <= 0 or t1_s <= 0 or not 0 < tolerance < 1:
        raise ValueError(
            "repetition_s and t1_s must be positive and tolerance in (0, 1)"
        )
    factor = math.exp(-repetition_s / t1_s) * math.cos(math.radians(flip_angle_deg))
    if factor <= tolerance:
        return 1
    return max(1, math.ceil(math.log(tolerance) / math.log(factor)))
