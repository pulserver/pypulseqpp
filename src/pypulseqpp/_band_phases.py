"""Per-band phase schedules that lower a multiband pulse's peak amplitude.

Bands in phase peak together at the pulse's centre, so peak amplitude grows
with the band count. The schedules:

- ``"quadratic"``: ``3.4 / N`` times the squared distance from the centre
  band, Grissom's closed form.
- ``"wong"``: numerically optimised phases for 3 to 16 bands (E C Wong, ISMRM
  2012, p. 2209).
- ``"malik"``: Hermitian phases for 4 to 12 bands (S J Malik, ISMRM 2015,
  p. 2398). Opposite bands carry opposite phases, so with the bands placed
  symmetrically the modulation is real and rides on the amplitude alone.

The two tables are transcribed from SigPy's
``sigpy.mri.rf.multiband.mb_phs_tab`` (Copyright (c) 2016, Frank Ong and The
Regents of the University of California; BSD 3-Clause, see
``LICENSES/SigPy-BSD-3-Clause.txt``).
"""

from __future__ import annotations

__all__ = ["BAND_PHASE_SCHEDULES", "band_phases"]

import numpy as np

BAND_PHASE_SCHEDULES = ("quadratic", "wong", "malik")

# fmt: off
_WONG = {
    3: (0.0, 0.73, 4.602),
    4: (0.0, 3.875, 5.94, 6.197),
    5: (0.0, 3.778, 5.335, 0.872, 0.471),
    6: (0.0, 2.005, 1.674, 5.012, 5.736, 4.123),
    7: (0.0, 3.002, 5.998, 5.909, 2.624, 2.528, 2.44),
    8: (0.0, 1.036, 3.414, 3.778, 3.215, 1.756, 4.555, 2.467),
    9: (0.0, 1.25, 1.783, 3.558, 0.739, 3.319, 1.296, 0.521, 5.332),
    10: (0.0, 4.418, 2.36, 0.677, 2.253, 3.472, 3.04, 3.974, 1.192, 2.51),
    11: (0.0, 5.041, 4.285, 3.001, 5.765, 4.295, 0.056, 4.213, 6.04, 1.078, 2.759),
    12: (0.0, 2.755, 5.491, 4.447, 0.231, 2.499, 3.539, 2.931, 2.759, 5.376, 4.554,
         3.479),
    13: (0.0, 0.603, 0.009, 4.179, 4.361, 4.837, 0.816, 5.995, 4.15, 0.417, 1.52, 4.517,
         1.729),
    14: (0.0, 3.997, 0.83, 5.712, 3.838, 0.084, 1.685, 5.328, 0.237, 0.506, 1.356, 4.025,
         4.483, 4.084),
    15: (0.0, 4.126, 2.266, 0.957, 4.603, 0.815, 3.475, 0.977, 1.449, 1.192, 0.148,
         0.939, 2.531, 3.612, 4.801),
    16: (0.0, 4.359, 3.51, 4.41, 1.75, 3.357, 2.061, 5.948, 3.0, 2.822, 0.627, 2.768,
         3.875, 4.173, 4.224, 5.941),
}

_MALIK = {
    4: (0.0, np.pi, np.pi, 0.0),
    5: (0.0, 0.0, np.pi, 0.0, 0.0),
    6: (1.691, 2.812, 1.157, -1.157, -2.812, -1.691),
    7: (2.582, -0.562, 0.102, 0.0, -0.102, 0.562, -2.582),
    8: (2.112, 0.22, 1.464, 1.992, -1.992, -1.464, -0.22, -2.112),
    9: (0.479, -2.667, -0.646, -0.419, 0.0, 0.419, 0.646, 2.667, -0.479),
    10: (1.683, -2.395, 2.913, 0.304, 0.737, -0.737, -0.304, -2.913, 2.395, -1.683),
    11: (1.405, 0.887, -1.854, 0.07, -1.494, 0.0, 1.494, -0.07, 1.854, -0.887, -1.405),
    12: (1.729, 0.444, 0.722, 2.19, -2.196, 0.984, -0.984, 2.196, -2.19, -0.722, -0.444,
         -1.729),
}
# fmt: on

_TABLES = {"wong": _WONG, "malik": _MALIK}


def band_phases(num_bands: int, schedule: str) -> np.ndarray:
    """Return one phase per band, in radians, lowest frequency first.

    Parameters
    ----------
    num_bands : int
        Bands, counting the on-resonance one.
    schedule : {"quadratic", "wong", "malik"}
        Which schedule; see the module docstring.

    Returns
    -------
    numpy.ndarray
        ``(num_bands,)`` phases.

    Raises
    ------
    ValueError
        If the schedule is unknown, or a table has no entry for this many
        bands.
    """
    if schedule == "quadratic":
        position = np.arange(num_bands) - (num_bands - 1) / 2.0
        return (3.4 / num_bands) * position**2
    table = _TABLES.get(schedule)
    if table is None:
        raise ValueError(
            f"phases must be one of {BAND_PHASE_SCHEDULES} or one value per band, "
            f"got {schedule!r}"
        )
    if num_bands not in table:
        raise ValueError(
            f"{schedule!r} phases are tabulated for {min(table)} to {max(table)} "
            f"bands, not {num_bands}"
        )
    return np.asarray(table[num_bands], dtype=float)
