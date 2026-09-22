"""Within-shot phase-encoding offsets of EPI echo trains.

An EPI shot acquires one echo per phase-encoding blip. The routine here
returns the offsets of the shot's echoes from the shot's first echo, one row
per echo. It does not select the acquisition support and does not place the
shot: the scan loop chooses each shot's origin, and the views a shot
acquires are that origin plus the offsets.

References
----------
Segmented blipped-CAIPI (``'caipi'``) follows Stirnberg R, Stöcker T.
Segmented k-space blipped-controlled aliasing in parallel imaging for high
spatiotemporal resolution EPI. Magn Reson Med. 2021;85(3):1540-1551.
doi:10.1002/mrm.28486. The ``'zigzag'`` traversal follows Dong Z, Wald LL,
Polimeni JR, Wang F. Single-shot echo planar time-resolved imaging for
multi-echo functional MRI and distortion-free diffusion imaging. Magn Reson
Med. 2025;93(3):993-1013. doi:10.1002/mrm.30327.
"""

from __future__ import annotations

__all__ = ["make_epi_shot_offsets"]

import numpy as np

from ._masks import make_caipirinha_mask

_SCHEMES = ("linear", "caipi", "zigzag")


def make_epi_shot_offsets(
    etl: int,
    *,
    scheme: str = "linear",
    acceleration: int = 1,
    segments: int = 1,
    partition_acceleration: int = 1,
    caipi_shift: int = 1,
    extent: int | None = None,
) -> np.ndarray:
    """Return the phase-encoding offsets of one EPI shot, relative to its first echo.

    Row ``e`` is the offset ``(Δky, Δkz)``, in encoded lines and partitions,
    of the view acquired at echo ``e`` from the view acquired at echo 0.
    The first row is always ``(0, 0)``. For a shot whose first echo acquires
    the view ``(y0, z0)``, echo ``e`` acquires ``(y0, z0) + offsets[e]``;
    choosing ``(y0, z0)`` for each shot is the scan loop's role, so this
    routine does not determine the global acquisition support.

    With ``R_y = acceleration`` and ``S = segments``, the line offset of
    ``'linear'`` and ``'caipi'`` is ``Δky_e = e S R_y``. ``S`` shots with
    origins ``y0, y0 + R_y, ..., y0 + (S - 1) R_y`` therefore interleave to
    acquire every ``R_y``-th line. ``'caipi'`` adds the partition offset of
    the CAIPIRINHA lattice with ``R_z = partition_acceleration`` and shift
    ``Δz = caipi_shift``, reduced modulo ``R_z``, so the partition blips
    cycle through one lattice period; the offsets tile
    :func:`make_caipirinha_mask` exactly. ``'zigzag'`` alternates between an
    outward and a return pass across ``extent`` lines, the return pass
    displaced by half a blip.

    Parameters
    ----------
    etl : int
        Echo-train length: the number of echoes, and of rows returned.
    scheme : {'linear', 'caipi', 'zigzag'}, default='linear'
        ``'linear'``: segmented or single-shot EPI, ``Δkz = 0``.
        ``'caipi'``: segmented blipped-CAIPI. ``'zigzag'``: repeated
        up-and-down traversal of one phase-encoding segment, which acquires
        the same lines at several echo times; ``Δkz = 0``.
    acceleration : int, default=1
        In-plane undersampling factor ``R_y``.
    segments : int, default=1
        Number of shots ``S`` among which the lines are interleaved. The blip
        is ``S R_y`` lines.
    partition_acceleration : int, default=1
        Partition undersampling factor ``R_z``, the period of the partition
        offsets. ``'caipi'`` only.
    caipi_shift : int, default=1
        CAIPIRINHA shift ``Δz``: partitions the lattice is displaced by per
        acquired line. ``'caipi'`` only.
    extent : int or None, default=None
        Phase-encoding lines spanned by one pass. Required by ``'zigzag'``
        and rejected by the other schemes.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(etl, 2)``: ``[Δky, Δkz]`` of each echo,
        relative to echo 0.

    Raises
    ------
    ValueError
        If a count is out of range, ``scheme`` is unknown, or ``extent`` is
        given for a scheme other than ``'zigzag'`` or omitted for it.

    See Also
    --------
    make_caipirinha_mask : the lattice the ``'caipi'`` offsets tile.
    make_cartesian_plane_sampling : acquisition support of a Cartesian plane.
    make_traversal_order : loop order over shots or slices.

    Examples
    --------
    Four echoes at twofold acceleration step two lines per echo:

    >>> import pypulseqpp as pp
    >>> offsets = pp.make_epi_shot_offsets(4, acceleration=2)
    >>> offsets.tolist()
    [[0, 0], [2, 0], [4, 0], [6, 0]]

    The offsets are relative. With two segments the blip is four lines, and
    shots with origins at lines 3 and 5 acquire interleaved lines:

    >>> offsets = pp.make_epi_shot_offsets(4, acceleration=2, segments=2)
    >>> (offsets + [3, 0])[:, 0].tolist()
    [3, 7, 11, 15]
    >>> (offsets + [5, 0])[:, 0].tolist()
    [5, 9, 13, 17]

    Segmented blipped-CAIPI with ``R_z = 3`` cycles the partition offset
    through one lattice period:

    >>> pp.make_epi_shot_offsets(
    ...     6, scheme="caipi", acceleration=2, partition_acceleration=3
    ... ).tolist()
    [[0, 0], [2, 1], [4, 2], [6, 0], [8, 1], [10, 2]]

    A zigzag pass turns at ``extent``, and the return pass is displaced by
    half a blip:

    >>> offsets = pp.make_epi_shot_offsets(
    ...     9, scheme="zigzag", acceleration=4, extent=12
    ... )
    >>> offsets[:, 0].tolist()
    [0, 4, 8, 12, 10, 6, 2, 0, 4]
    """
    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be positive")
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
    acceleration, segments = int(acceleration), int(segments)
    if acceleration < 1 or segments < 1:
        raise ValueError("acceleration and segments must be positive")
    if (extent is None) != (scheme != "zigzag"):
        raise ValueError(
            "extent is what a 'zigzag' turns around inside and means nothing to the "
            "other schemes; give it for 'zigzag' and leave it unset otherwise"
        )

    if scheme == "zigzag":
        line = _zigzag(etl, acceleration * segments, int(extent))
        partition = np.zeros(etl, dtype=np.intp)
    else:
        line = np.arange(etl, dtype=np.intp) * (acceleration * segments)
        partition = (
            _caipi_partitions(
                line, acceleration, int(partition_acceleration), int(caipi_shift)
            )
            if scheme == "caipi"
            else np.zeros(etl, dtype=np.intp)
        )
    return np.column_stack((line, partition))


def _caipi_partitions(line: np.ndarray, ry: int, rz: int, shift: int) -> np.ndarray:
    """Return the partition index of each phase-encode line.

    Taken from :func:`make_caipirinha_mask` rather than restated, so a train
    and the mask it is meant to tile cannot drift apart. One period of the
    lattice is enough: every sampled row carries exactly one partition.
    """
    if rz < 1:
        raise ValueError("partition_acceleration must be positive")
    mask = make_caipirinha_mask((int(line[-1]) + 1, rz), ry, rz, delta=shift)
    return np.array([np.flatnonzero(mask[index])[0] for index in line], dtype=np.intp)


def _zigzag(etl: int, step: int, extent: int) -> np.ndarray:
    """Return a zigzag traversal of one phase-encode segment, alternating two lattices.

    The outward pass visits ``0, step, 2 * step, ...`` as far as ``extent``
    allows; the return pass visits the same lattice offset by ``step // 2``, so
    consecutive passes interleave and every blip stays within one ``step``. The two passes together are the cycle, repeated until the
    train runs out.
    """
    if extent < 0:
        raise ValueError("extent must be nonnegative")
    outward = np.arange(0, extent + 1, step, dtype=np.intp)
    inward = outward[:-1] + step // 2 if step > 1 else outward[:-1]
    cycle = np.concatenate((outward, inward[::-1]))
    repeats = -(-etl // len(cycle))
    return np.tile(cycle, repeats)[:etl]
