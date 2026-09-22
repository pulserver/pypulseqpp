"""Cartesian sampling support: encoded views and boolean masks.

Support functions select which Cartesian views an acquisition encodes. They
return either explicit zero-based encoded coordinates or a boolean mask whose
``True`` entries are the acquired views. They assign no temporal order, no
shot or echo index and no label; :mod:`pypulseqpp._ordering` orders a
selected set, and the sequence application scales its phase-encoding
gradients and writes its labels from the coordinates.

References
----------
Poisson-disc sampling derives from SigPy (BSD 3-Clause).
"""

from __future__ import annotations

__all__ = [
    "make_caipirinha_mask",
    "make_cartesian_axis_sampling",
    "make_cartesian_plane_sampling",
    "make_poisson_disc_mask",
    "make_random_mask",
]

import numpy as np

#: The support schemes :func:`make_cartesian_plane_sampling` draws from.
_PLANE_SCHEMES = ("lattice", "poisson")


def make_cartesian_axis_sampling(
    n: int,
    acceleration: int = 1,
    n_acs: int = 0,
    *,
    partial_fourier: float = 1.0,
) -> tuple[list[int], list[int]]:
    r"""Select the acquired views of one Cartesian encoding axis.

    The views are the zero-based indices ``0 .. n - 1`` of the axis, with the
    k-space centre at index ``n // 2``. With ``R = acceleration``, the
    undersampled lattice is

    .. math::

        \{\, i : (i - \lfloor n/2 \rfloor) \bmod R = 0 \,\},

    so the centre view is acquired for every ``R``. Partial Fourier removes
    the indices below ``n - round(partial_fourier * n)``; the centre and the
    far edge are retained. The calibration region is the ``n_acs`` contiguous
    views centred on ``n // 2``, fully sampled and clipped by partial Fourier.

    The routine applies to any Cartesian encoding axis: the phase-encoding
    lines of a 2D acquisition, or the partitions of a stack-of-stars,
    stack-of-spirals or stack-of-blades acquisition.

    Parameters
    ----------
    n : int
        Number of views on the axis (the encoding matrix size).
    acceleration : int, default=1
        Undersampling factor ``R`` of the lattice.
    n_acs : int, default=0
        Number of fully sampled calibration (ACS) views centred on
        ``n // 2``. Ignored when ``acceleration`` is 1.
    partial_fourier : float, default=1.0
        Fraction of the axis acquired, in ``(0.5, 1]``.

    Returns
    -------
    calibration : list of int
        Calibration views, ascending. Empty when ``acceleration`` is 1 or
        ``n_acs`` is 0.
    imaging : list of int
        Lattice views not in ``calibration``, ascending. The two lists are
        disjoint; the acquired support is their union.

    Raises
    ------
    ValueError
        If ``acceleration`` is below one, ``n_acs`` is negative, or
        ``partial_fourier`` is outside ``(0.5, 1]``.

    See Also
    --------
    make_cartesian_plane_sampling : the same selection over two encoding axes.
    make_traversal_order : a loop order over the selected views.

    Notes
    -----
    The two groups are returned separately so that a sequence can acquire
    and label them separately; the shipped sequences acquire the
    calibration views first and mark them with the ``IMA`` label.
    ``[*calibration, *imaging]`` is that acquisition order and
    ``sorted(calibration + imaging)`` the ascending one.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> pp.make_cartesian_axis_sampling(8)
    ([], [0, 1, 2, 3, 4, 5, 6, 7])

    Twofold undersampling about the centre view ``4``, with four calibration
    views. The imaging views exclude the calibration views:

    >>> calibration, imaging = pp.make_cartesian_axis_sampling(8, 2, 4)
    >>> calibration
    [2, 3, 4, 5]
    >>> imaging
    [0, 6]

    Partial Fourier with fraction 0.75 removes the first two views:

    >>> pp.make_cartesian_axis_sampling(8, partial_fourier=0.75)
    ([], [2, 3, 4, 5, 6, 7])
    """
    if acceleration < 1:
        raise ValueError(f"acceleration must be at least 1, got {acceleration}")
    if n_acs < 0:
        raise ValueError(f"n_acs must be nonnegative, got {n_acs}")
    first = n - round(_checked_partial_fourier(partial_fourier) * n)
    size = n_acs if acceleration > 1 else 0
    calibration = list(
        range(max(n // 2 - size // 2, first), min(n // 2 + (size + 1) // 2, n))
    )
    imaging = [
        i
        for i in range(first, n)
        if (i - n // 2) % acceleration == 0 and i not in calibration
    ]
    return calibration, imaging


def make_cartesian_plane_sampling(
    shape: tuple[int, int],
    acceleration: tuple[int, int] = (1, 1),
    n_acs: tuple[int, int] = (0, 0),
    *,
    caipi_shift: int = 0,
    partial_fourier: tuple[float, float] = (1.0, 1.0),
    elliptical: bool = False,
    elliptical_acs: bool = False,
    sampling: str = "lattice",
    seed: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Select the acquired ``(line, partition)`` views of a Cartesian ky-kz plane.

    A view is a zero-based encoded coordinate ``(y, z)`` on the
    ``(n_y, n_z)`` grid, whose k-space centre is ``(n_y // 2, n_z // 2)``.

    With ``sampling='lattice'`` the support is the CAIPIRINHA lattice. With
    ``R_y, R_z = acceleration`` and ``Δ = caipi_shift``, line ``y`` is
    acquired when ``(y - n_y // 2) mod R_y = 0``; on the ``j``-th acquired line
    counted from the centre line, ``j = (y - n_y // 2) // R_y``, partition
    ``z`` is acquired when ``(z - n_z // 2 - Δ j) mod R_z = 0``. The centre
    view is always acquired; ``Δ = 0`` gives a rectangular lattice.

    With ``sampling='poisson'`` the support is a variable-density Poisson-disc
    draw at nominal acceleration ``R_y * R_z`` from
    :func:`make_poisson_disc_mask`, which includes the calibration region and
    is restricted to the ellipse inscribed in the grid. ``caipi_shift`` is
    ignored.

    Partial Fourier and the ``elliptical`` crop are applied to either support,
    and the calibration views are then separated from the rest.

    Parameters
    ----------
    shape : tuple of int
        ``(n_y, n_z)``, the number of lines and partitions.
    acceleration : tuple of int, default=(1, 1)
        ``(R_y, R_z)``, the undersampling factor on each axis.
    n_acs : tuple of int, default=(0, 0)
        ``(n_acs_y, n_acs_z)``, extent of the fully sampled calibration region
        centred on the centre view. Ignored when ``R_y * R_z`` is 1.
    caipi_shift : int, default=0
        CAIPIRINHA shift ``Δ``: partitions the lattice is displaced by per
        acquired line. ``sampling='lattice'`` only.
    partial_fourier : tuple of float, default=(1.0, 1.0)
        Fraction of each axis acquired, in ``(0.5, 1]``. The views with the
        lowest indices are removed.
    elliptical : bool, default=False
        Restrict the imaging views to the ellipse inscribed in the grid.
    elliptical_acs : bool, default=False
        Restrict the calibration region to the ellipse inscribed in its
        rectangle.
    sampling : {'lattice', 'poisson'}, default='lattice'
        Support scheme: the deterministic CAIPIRINHA lattice, or a
        variable-density Poisson-disc draw.
    seed : int, default=0
        Random seed of the Poisson-disc draw. ``sampling='poisson'`` only.

    Returns
    -------
    calibration : list of tuple of int
        Calibration views ``(y, z)``, ordered by line and then by partition.
        Empty when there is no calibration region.
    imaging : list of tuple of int
        Acquired views ``(y, z)`` not in ``calibration``, in the same order.
        The two lists are disjoint; the acquired support is their union.

    Raises
    ------
    ValueError
        If an undersampling factor is below one, a calibration extent is
        negative, a partial-Fourier fraction is outside ``(0.5, 1]``, or
        ``sampling`` is unknown.

    See Also
    --------
    make_cartesian_axis_sampling : the same selection over one encoding axis.
    make_caipirinha_mask : the lattice as a boolean mask, without calibration.
    make_poisson_disc_mask : the Poisson-disc support as a boolean mask.
    make_radial_order : assignment of the selected views to echo trains.

    Notes
    -----
    This routine is the prescription-level support selection used by the
    shipped 3D Cartesian sequences. The mask generators produce support only;
    this routine combines a support scheme with the calibration region,
    partial Fourier and elliptical cropping, and returns the result as
    coordinate lists separated by role.

    Examples
    --------
    Twofold undersampling on both axes of a 4 x 4 grid, with a 2 x 2
    calibration region at the centre view ``(2, 2)``:

    >>> import pypulseqpp as pp
    >>> calibration, imaging = pp.make_cartesian_plane_sampling(
    ...     (4, 4), acceleration=(2, 2), n_acs=(2, 2)
    ... )
    >>> calibration
    [(1, 1), (1, 2), (2, 1), (2, 2)]
    >>> imaging
    [(0, 0), (0, 2), (2, 0)]

    The view ``(2, 2)`` is on the lattice and in the calibration region; it is
    listed once, in ``calibration``. A CAIPIRINHA shift of one partition per
    acquired line displaces the partitions of line 0 relative to line 2:

    >>> pp.make_cartesian_plane_sampling((4, 4), (2, 2), caipi_shift=1)[1]
    [(0, 1), (0, 3), (2, 0), (2, 2)]

    The coordinates convert directly to a boolean mask:

    >>> import numpy as np
    >>> mask = np.zeros((4, 4), dtype=bool)
    >>> mask[tuple(np.transpose(calibration + imaging))] = True
    >>> mask.astype(int)
    array([[1, 0, 1, 0],
           [0, 1, 1, 0],
           [1, 1, 1, 0],
           [0, 0, 0, 0]])
    """
    (n_y, n_z), (r_y, r_z) = shape, acceleration
    if r_y < 1 or r_z < 1:
        raise ValueError(
            f"acceleration must be at least 1 on each axis, got {acceleration}"
        )
    if n_acs[0] < 0 or n_acs[1] < 0:
        raise ValueError(f"n_acs must be nonnegative on each axis, got {n_acs}")
    if sampling not in _PLANE_SCHEMES:
        raise ValueError(f"sampling must be one of {_PLANE_SCHEMES}, got {sampling!r}")
    first_y = n_y - round(_checked_partial_fourier(partial_fourier[0]) * n_y)
    first_z = n_z - round(_checked_partial_fourier(partial_fourier[1]) * n_z)
    n_acs_y, n_acs_z = n_acs if r_y * r_z > 1 else (0, 0)

    def inside(y: int, z: int, extent_y: int, extent_z: int) -> bool:
        # Offsets from the centre view, normalised to the ellipse's axes.
        dy, dz = (y - n_y // 2) / extent_y, (z - n_z // 2) / extent_z
        return dy * dy + dz * dz <= 0.25

    lines_acs = range(
        max(n_y // 2 - n_acs_y // 2, first_y), min(n_y // 2 + (n_acs_y + 1) // 2, n_y)
    )
    partitions_acs = range(
        max(n_z // 2 - n_acs_z // 2, first_z), min(n_z // 2 + (n_acs_z + 1) // 2, n_z)
    )
    calibration = [
        (y, z)
        for y in lines_acs
        for z in partitions_acs
        if not elliptical_acs or inside(y, z, n_acs_y, n_acs_z)
    ]
    calibrating = set(calibration)
    if sampling == "poisson":
        drawn = (
            make_poisson_disc_mask(
                (n_y, n_z), float(r_y * r_z), calib=(n_acs_y, n_acs_z), seed=seed
            )
            if r_y * r_z > 1
            else np.ones((n_y, n_z), dtype=bool)
        )
        support = [(int(y), int(z)) for y, z in np.argwhere(drawn)]
    else:
        support = [
            (y, z)
            for y in range(n_y)
            if (y - n_y // 2) % r_y == 0
            for z in range(n_z)
            if (z - n_z // 2 - caipi_shift * ((y - n_y // 2) // r_y)) % r_z == 0
        ]
    imaging = [
        (y, z)
        for y, z in support
        if y >= first_y
        and z >= first_z
        and (not elliptical or inside(y, z, n_y, n_z))
        and (y, z) not in calibrating
    ]
    return calibration, imaging


def _checked_partial_fourier(partial_fourier: float) -> float:
    if not 0.5 < partial_fourier <= 1.0:
        raise ValueError("partial_fourier must be in (0.5, 1]")
    return partial_fourier


def make_random_mask(
    shape: tuple[int, int],
    accel: float,
    *,
    calib: tuple[int, int] = (0, 0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a uniform-random Cartesian sampling mask with a calibration region.

    ``mask[y, z]`` is ``True`` when the view ``(y, z)`` is acquired. The mask
    holds ``round(n_y * n_z / accel)`` views: the centred calibration block,
    and the remainder drawn uniformly without replacement from the other
    views. If the calibration block alone exceeds that number, the mask is
    the calibration block.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(n_y, n_z)``.
    accel : float
        Target acceleration factor, greater than one.
    calib : tuple of int, default=(0, 0)
        Extent of the fully sampled calibration block centred on
        ``(n_y // 2, n_z // 2)``.
    seed : int or None, default=None
        Seed of the random draw.

    Returns
    -------
    numpy.ndarray
        Boolean support mask of shape ``shape``. It encodes no acquisition
        order.

    Raises
    ------
    ValueError
        If the acceleration is not greater than one.

    See Also
    --------
    make_poisson_disc_mask : variable-density draw with a minimum distance.
    make_caipirinha_mask : deterministic lattice for parallel imaging.
    make_linear_order : echo-train ordering, which accepts a mask as input.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> mask = pp.make_random_mask((6, 6), 3.0, calib=(2, 2), seed=0)
    >>> mask.dtype, mask.shape, int(mask.sum())
    (dtype('bool'), (6, 6), 12)
    >>> bool(mask[2:4, 2:4].all())
    True

    ``np.argwhere`` lists the acquired views as ``(y, z)`` coordinates:

    >>> np.argwhere(mask)[:4].tolist()
    [[0, 0], [0, 1], [0, 2], [1, 1]]
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    ny, nz = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    y0 = ny // 2 - calib[0] // 2
    z0 = nz // 2 - calib[1] // 2
    mask[y0 : y0 + calib[0], z0 : z0 + calib[1]] = True

    n_target = round(ny * nz / accel)
    n_extra = max(0, n_target - int(mask.sum()))
    free = np.flatnonzero(~mask.ravel())
    chosen = rng.choice(free, size=min(n_extra, free.size), replace=False)
    flat = mask.ravel()
    flat[chosen] = True
    return flat.reshape(shape)


def make_caipirinha_mask(
    shape: tuple[int, int],
    ry: int,
    rz: int,
    *,
    delta: int = 1,
) -> np.ndarray:
    """Generate a CAIPIRINHA lattice sampling mask.

    ``mask[y, z]`` is ``True`` when the view ``(y, z)`` is acquired: line
    ``y`` is acquired when ``y mod ry = 0``, and on it partition ``z`` when
    ``(z - delta * (y // ry)) mod rz = 0``. The lattice is anchored at
    ``(0, 0)`` and spreads the aliasing along both phase-encoding directions
    [1]_.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(n_y, n_z)``.
    ry : int
        Undersampling factor along the first axis.
    rz : int
        Undersampling factor along the second axis.
    delta : int, default=1
        CAIPIRINHA shift: partitions the lattice is displaced by per acquired
        line. ``delta=0`` gives a rectangular ``ry x rz`` lattice.

    Returns
    -------
    numpy.ndarray
        Boolean support mask of shape ``shape``, with nominal acceleration
        ``ry * rz``; the finite grid can change the realised factor. It
        encodes no acquisition order.

    Raises
    ------
    ValueError
        If an undersampling factor is below one.

    See Also
    --------
    make_cartesian_plane_sampling : the same lattice anchored on the k-space
        centre, with calibration and partial Fourier.
    make_epi_shot_offsets : segmented blipped-CAIPI shot offsets that tile
        this lattice.

    References
    ----------
    .. [1] Breuer FA, Blaimer M, Mueller MF, et al. Controlled aliasing in
       volumetric parallel imaging (2D CAIPIRINHA). *Magnetic Resonance in
       Medicine*. 2006;55(3):549-556. https://doi.org/10.1002/mrm.20787

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> mask = pp.make_caipirinha_mask((4, 4), 2, 2, delta=1)
    >>> mask.astype(int)
    array([[1, 0, 1, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 1],
           [0, 0, 0, 0]])
    >>> np.argwhere(mask).tolist()
    [[0, 0], [0, 2], [2, 1], [2, 3]]
    """
    if ry < 1 or rz < 1:
        raise ValueError("ry and rz must be >= 1")
    ny, nz = shape
    ky = np.arange(ny)[:, None]
    kz = np.arange(nz)[None, :]
    shift = (ky // ry) * delta
    return (ky % ry == 0) & ((kz - shift) % rz == 0)


def make_poisson_disc_mask(
    shape: tuple[int, int],
    accel: float,
    *,
    calib: tuple[int, int] = (0, 0),
    seed: int = 0,
    max_attempts: int = 30,
    tol: float = 0.1,
    crop_corner: bool = True,
) -> np.ndarray:
    """Generate a variable-density Poisson-disc Cartesian sampling mask.

    ``mask[y, z]`` is ``True`` when the view ``(y, z)`` is acquired. The
    minimum distance between acquired views grows with the distance ``r``
    from the k-space centre as ``1 + s r``; the slope ``s`` is found by
    bisection so that the realised acceleration matches ``accel`` within
    ``tol``. Views are placed by Bridson's dart throwing [1]_. The algorithm
    is adapted from ``sigpy.mri.samp.poisson``.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(n_y, n_z)``.
    accel : float
        Target acceleration factor, greater than one.
    calib : tuple of int, default=(0, 0)
        Extent of the fully sampled calibration block centred on
        ``(n_y // 2, n_z // 2)``.
    seed : int, default=0
        Seed of the random draw. Equal seeds give equal masks.
    max_attempts : int, default=30
        Bridson candidate attempts per active point.
    tol : float, default=0.1
        Allowed deviation of the realised acceleration from ``accel``.
    crop_corner : bool, default=True
        Restrict the support to the ellipse inscribed in the grid.

    Returns
    -------
    numpy.ndarray
        Boolean support mask of shape ``shape``. It encodes no acquisition
        order; :func:`make_shuffling_order` is an echo ordering, not a
        support.

    Raises
    ------
    ValueError
        If the acceleration is not greater than one, or the draw cannot reach
        it within the shape given.

    See Also
    --------
    make_cartesian_plane_sampling : Poisson-disc support with calibration and
        partial Fourier, as coordinate lists (``sampling='poisson'``).

    References
    ----------
    .. [1] Bridson R. Fast Poisson disk sampling in arbitrary dimensions.
       *ACM SIGGRAPH 2007 Sketches*. 2007:22.
       https://doi.org/10.1145/1278780.1278807

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> mask = pp.make_poisson_disc_mask((24, 24), 3.0, calib=(6, 6), seed=1)
    >>> mask.dtype, mask.shape
    (dtype('bool'), (24, 24))
    >>> bool(mask[9:15, 9:15].all())
    True
    >>> bool((pp.make_poisson_disc_mask((24, 24), 3.0, calib=(6, 6), seed=1) == mask).all())
    True
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    from pypulseqpp._ext import sampling

    return sampling.poisson_disc_mask(
        shape, accel, calib, seed, max_attempts, tol, crop_corner
    )
