"""Units and frame shared by the checks that read the physical-axis gradient."""

from __future__ import annotations

import numpy as np


def _gamma(seq, system) -> float:
    for chosen in (system, getattr(seq, "system", None)):
        gamma = getattr(chosen, "gamma", None)
        if gamma:
            return float(gamma)
    from .. import Opts

    return float(Opts().gamma)


def _prescription(rotation) -> np.ndarray:
    turn = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    if turn.shape != (3, 3) or not np.allclose(turn @ turn.T, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be a 3x3 orthonormal matrix")
    return turn
