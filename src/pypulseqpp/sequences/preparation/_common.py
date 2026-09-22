"""Three-axis spoilers shared by the preparation modules."""

from __future__ import annotations

__all__ = ["AXES", "spoiler_gradients"]

import pypulseqpp as pp

AXES = ("x", "y", "z")


def spoiler_gradients(
    system: pp.Opts, spoiling_cycles: float, voxel_size_m: float
) -> tuple:
    """Return x, y and z crushers with equal area in cycles per voxel length.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits the crushers are designed against.
    spoiling_cycles : float
        Phase the crusher wraps across one voxel, in cycles.
    voxel_size_m : float
        The voxel length that phase is wrapped across, in m.

    Returns
    -------
    tuple
        One trapezoid per axis, in the order of `AXES`.
    """
    return tuple(
        pp.make_crusher(spoiling_cycles, voxel_size_m, axis, system=system)[0]
        for axis in AXES
    )
