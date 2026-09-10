"""The two things a magnetization preparation almost always needs."""

from __future__ import annotations

__all__ = ["AXES", "spoiler_gradients"]

import pypulseqpp as pp

AXES = ("x", "y", "z")


def spoiler_gradients(
    system: pp.Opts, spoiling_cycles: float, voxel_size_m: float
) -> tuple:
    """Crusher on each axis, all winding the same dephasing.

    Three axes rather than one because what a preparation has to destroy is a
    whole excitation, not a residue: a single axis leaves everything with no
    variation along it untouched.

    Returns
    -------
    tuple
        ``(gx_spoil, gy_spoil, gz_spoil)``.
    """
    return tuple(
        pp.make_crusher(spoiling_cycles, voxel_size_m, axis, system=system)[0]
        for axis in AXES
    )
