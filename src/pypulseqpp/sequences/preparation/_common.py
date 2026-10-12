"""Three-axis spoilers shared by the preparation modules."""

from __future__ import annotations

__all__ = ["AXES", "spoiler_gradients"]

import pypulseqpp as pp

AXES = ("x", "y", "z")


def spoiler_gradients(
    system: pp.Opts, spoiling_cycles: float, voxel_size_m: float
) -> tuple:
    """Return the x, y and z crushers of equal area.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits the crushers are designed against.
    spoiling_cycles : float
        Cycles of dephasing across ``voxel_size_m``.
    voxel_size_m : float
        Length the dephasing is counted over (m).

    Returns
    -------
    tuple
        One trapezoid per axis, in the order of ``AXES``.
    """
    return tuple(
        pp.make_crusher(spoiling_cycles, voxel_size_m, axis, system=system)[0]
        for axis in AXES
    )
