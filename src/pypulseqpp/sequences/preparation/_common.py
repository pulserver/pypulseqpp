"""Three-axis spoilers shared by the preparation modules."""

from __future__ import annotations

__all__ = ["AXES", "spoiler_gradients"]

import pypulseqpp as pp

AXES = ("x", "y", "z")


def spoiler_gradients(
    system: pp.Opts, spoiling_cycles: float, voxel_size_m: float
) -> tuple:
    """Return x, y and z crushers with equal area in cycles per voxel length."""
    return tuple(
        pp.make_crusher(spoiling_cycles, voxel_size_m, axis, system=system)[0]
        for axis in AXES
    )
