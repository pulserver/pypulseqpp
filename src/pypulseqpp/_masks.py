"""Cartesian undersampling masks and echo-train orderings.

Masks are boolean arrays; orderings contain zero-based indices into the
input coordinates. Optional padding uses None, not an acquired view.

References
----------
Echo-train reordering: Buonincontri et al., ISMRM abstract 566-05-007,
Fig. 2. Shuffling: Tamir et al., Magn Reson Med 2017;77:180-195.
Poisson-disc sampling derives from SigPy (BSD 3-Clause).
"""

from __future__ import annotations

__all__ = [
    "calc_sampled_lines",
    "calc_sampled_pairs",
    "make_caipirinha_mask",
    "make_centric_order",
    "make_linear_order",
    "make_poisson_disc_mask",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_random_mask",
    "make_shuffling_order",
]

import numpy as np


def calc_sampled_lines(
    n: int,
    r: int = 1,
    n_acs: int = 0,
    *,
    partial_fourier: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Return the calibration views of one encoded axis, and the others acquired.

    The lattice keeps every view ``i`` with ``(i - n // 2) % r == 0``, so the
    centre view is always acquired whatever the undersampling. Partial Fourier
    drops the views before ``n - round(partial_fourier * n)``, which leaves the
    centre in and the conjugate symmetry of k-space to cover what is missing.
    The calibration block is ``n_acs`` views centred on the same view, acquired
    whole; a fully sampled axis has none, because there is nothing to calibrate
    a reconstruction of.

    Parameters
    ----------
    n : int
        Views on the axis.
    r : int, default=1
        Undersampling: one view in every ``r`` of the lattice is acquired.
    n_acs : int, default=0
        Fully sampled calibration views at the centre. Ignored when ``r`` is 1.
    partial_fourier : float, default=1.0
        Fraction of the extent acquired, in ``(0.5, 1]``.

    Returns
    -------
    calibration : list of int
        The calibration block, ascending, and empty when there is none.
    lattice : list of int
        The other views acquired, ascending.

    Raises
    ------
    ValueError
        If ``r`` is below one, ``n_acs`` is negative, or ``partial_fourier``
        is outside ``(0.5, 1]``.

    Notes
    -----
    The two parts are returned separately because a scan plays them in that
    order -- the calibration first, so a reconstruction can estimate coil
    sensitivities while the rest is still being acquired -- and marks them
    differently. ``[*calibration, *lattice]`` is that play order and
    ``sorted(calibration + lattice)`` the ascending traversal.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> pp.calc_sampled_lines(8)
    ([], [0, 1, 2, 3, 4, 5, 6, 7])

    Twofold undersampling about the centre view, with four calibrating it:

    >>> pp.calc_sampled_lines(8, 2, 4)
    ([2, 3, 4, 5], [0, 6])

    Three quarters of the extent, counted from the far edge:

    >>> pp.calc_sampled_lines(8, partial_fourier=0.75)
    ([], [2, 3, 4, 5, 6, 7])
    """
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")
    if n_acs < 0:
        raise ValueError(f"n_acs must be nonnegative, got {n_acs}")
    first = n - round(_checked_partial_fourier(partial_fourier) * n)
    size = n_acs if r > 1 else 0
    calibration = list(
        range(max(n // 2 - size // 2, first), min(n // 2 + (size + 1) // 2, n))
    )
    lattice = [
        i for i in range(first, n) if (i - n // 2) % r == 0 and i not in calibration
    ]
    return calibration, lattice


def calc_sampled_pairs(
    shape: tuple[int, int],
    acceleration: tuple[int, int] = (1, 1),
    n_acs: tuple[int, int] = (0, 0),
    *,
    caipi_shift: int = 0,
    partial_fourier: tuple[float, float] = (1.0, 1.0),
    elliptical: bool = False,
    elliptical_acs: bool = False,
    shuffling: bool = False,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the calibration ``(line, partition)`` pairs, and the others acquired.

    Lines keep ``(y - n_y // 2) % r_y == 0``. The partitions of the ``j``-th
    acquired line from the centre keep
    ``(z - n_z // 2 - caipi_shift * j) % r_z == 0``, so the centre pair is
    always acquired and the partition lattice climbs ``caipi_shift`` per
    acquired line, which spreads the aliasing into both encoded directions
    rather than along one. Partial Fourier drops the lines and the partitions
    before the centre.

    Parameters
    ----------
    shape : tuple of int
        ``(n_y, n_z)``, the lines and the partitions.
    acceleration : tuple of int, default=(1, 1)
        ``(r_y, r_z)``, the undersampling on each axis.
    n_acs : tuple of int, default=(0, 0)
        ``(n_acs_y, n_acs_z)``, the calibration region centred on the centre
        pair. Ignored when neither axis is undersampled.
    caipi_shift : int, default=0
        Partitions the lattice climbs per acquired line. Zero is a plain
        rectangular lattice.
    partial_fourier : tuple of float, default=(1.0, 1.0)
        Fraction of each axis acquired, in ``(0.5, 1]``.
    elliptical : bool, default=False
        Keep only the pairs inside the ellipse inscribed in the grid, whose
        corners carry no resolution the axes do not already give.
    elliptical_acs : bool, default=False
        Shape the calibration region as the ellipse inscribed in it rather
        than as a rectangle.
    shuffling : bool, default=False
        Draw the pairs from a variable-density Poisson disc at ``r_y * r_z``
        instead of from the lattice, which spreads the aliasing incoherently
        rather than into a fixed replica and is what a reconstruction with a
        sparsity prior wants. ``caipi_shift`` then does nothing.
    seed : int, default=0
        Seed for that draw.

    Returns
    -------
    calibration : list of tuple of int
        The calibration region, line by line, each line's partitions
        ascending. Empty when there is none.
    lattice : list of tuple of int
        The other pairs acquired, in the same traversal.

    Raises
    ------
    ValueError
        If an undersampling factor is below one, a calibration extent is
        negative, or a partial-Fourier fraction is outside ``(0.5, 1]``.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> calibration, lattice = pp.calc_sampled_pairs((4, 4), (2, 2), (2, 2))
    >>> calibration
    [(1, 1), (1, 2), (2, 1), (2, 2)]
    >>> lattice
    [(0, 0), (0, 2), (2, 0)]

    A CAIPIRINHA shift moves each acquired line's partitions up by one:

    >>> pp.calc_sampled_pairs((4, 4), (2, 2), caipi_shift=1)[1]
    [(0, 1), (0, 3), (2, 0), (2, 2)]
    """
    (n_y, n_z), (r_y, r_z) = shape, acceleration
    if r_y < 1 or r_z < 1:
        raise ValueError(
            f"acceleration must be at least 1 on each axis, got {acceleration}"
        )
    if n_acs[0] < 0 or n_acs[1] < 0:
        raise ValueError(f"n_acs must be nonnegative on each axis, got {n_acs}")
    first_y = n_y - round(_checked_partial_fourier(partial_fourier[0]) * n_y)
    first_z = n_z - round(_checked_partial_fourier(partial_fourier[1]) * n_z)
    n_acs_y, n_acs_z = n_acs if r_y * r_z > 1 else (0, 0)

    def inside(y: int, z: int, extent_y: int, extent_z: int) -> bool:
        # Offsets from the centre pair, the one the encodes scale to zero.
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
    if shuffling:
        drawn = (
            make_poisson_disc_mask(
                (n_y, n_z), float(r_y * r_z), calib=(n_acs_y, n_acs_z), seed=seed
            )
            if r_y * r_z > 1
            else np.ones((n_y, n_z), dtype=bool)
        )
        kept = [(int(y), int(z)) for y, z in np.argwhere(drawn)]
    else:
        kept = [
            (y, z)
            for y in range(n_y)
            if (y - n_y // 2) % r_y == 0
            for z in range(n_z)
            if (z - n_z // 2 - caipi_shift * ((y - n_y // 2) // r_y)) % r_z == 0
        ]
    lattice = [
        (y, z)
        for y, z in kept
        if y >= first_y
        and z >= first_z
        and (not elliptical or inside(y, z, n_y, n_z))
        and (y, z) not in calibrating
    ]
    return calibration, lattice


def _elliptical_support(shape: tuple[int, int]) -> np.ndarray:
    """Return an elliptical mask centred at ``(ny // 2, nz // 2)``."""
    n_y, n_z = shape
    ky = (np.arange(n_y) - n_y // 2) / (n_y / 2)
    kz = (np.arange(n_z) - n_z // 2) / (n_z / 2)
    return (ky[:, None] ** 2 + kz[None, :] ** 2) <= 1.0


def _checked_partial_fourier(partial_fourier: float) -> float:
    if not 0.5 < partial_fourier <= 1.0:
        raise ValueError("partial_fourier must be in (0.5, 1]")
    return partial_fourier


def _as_coords(coords) -> np.ndarray:
    """Normalize a coordinate argument to an ``(N, 2)`` float array."""
    raw = np.asarray(coords)
    if raw.dtype == bool and raw.ndim in (1, 2):
        arr = np.argwhere(raw).astype(float)
    else:
        arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([arr, np.zeros_like(arr)])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must be shape (N,) or (N, 2)")
    return arr


def _split_into_shots(order: list[int], etl: int) -> list[list[int]]:
    """Consecutive chunks of ``etl``, the last one shorter when it has to be."""
    etl = max(1, int(etl))
    return [order[i : i + etl] for i in range(0, len(order), etl)]


def _polar(pts: np.ndarray, center) -> tuple[np.ndarray, np.ndarray]:
    """Return per-point radius and angle about ``center`` (mean when ``None``)."""
    origin = pts.mean(axis=0) if center is None else np.asarray(center, dtype=float)
    rel = pts - origin
    return np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])


def _finish_trains(
    trains: list[list[int | None]], etl: int, pad: bool
) -> list[list[int | None]]:
    """Pad each train to ``etl`` with ``None`` (``pad``), or drop the gaps."""
    if pad:
        return [train + [None] * (etl - len(train)) for train in trains]
    return [[index for index in train if index is not None] for train in trains]


def _deal_echo_major(
    order: list[int],
    secondary: np.ndarray,
    n_trains: int,
    etl: int,
    radius: np.ndarray,
    *,
    center_echo: int | None,
    adaptive: bool,
) -> list[list[int | None]]:
    """Deal a ranked index list into echo-major echo trains.

    ``order`` is the primary ranking. It is cut into ``etl`` groups of up to
    ``n_trains`` views; group ``g`` becomes one echo, and within a group the
    views are dealt across trains in ``secondary`` order so successive echoes
    of one train stay neighbours. ``center_echo`` places the group holding the
    k-space centre at that echo -- rolled for a monotone ranking, folded
    (``adaptive``) so the radius grows away from the target echo in both
    directions. This is the shared machinery behind the Cartesian echo-train
    orderings (566-05-007, Fig. 2).
    """
    groups = [order[g * n_trains : (g + 1) * n_trains] for g in range(etl)]
    echo_of_group = list(range(etl))
    if center_echo is not None:
        if adaptive:
            # Innermost group (rank 0) at the target echo, then outward.
            echo_of_group = sorted(range(etl), key=lambda echo: abs(echo - center_echo))
        else:
            centre = int(np.argmin(radius))
            holds_centre = next(
                index for index, group in enumerate(groups) if centre in group
            )
            roll = (holds_centre - center_echo) % etl
            groups = groups[roll:] + groups[:roll]

    trains: list[list[int | None]] = [[None] * etl for _ in range(n_trains)]
    for rank, group in enumerate(groups):
        echo = echo_of_group[rank]
        group = np.asarray(group, dtype=int)
        dealt = group[np.argsort(secondary[group], kind="stable")]
        for train, index in enumerate(dealt.tolist()):
            trains[train][echo] = index
    return trains


def make_linear_order(
    coords,
    train_length: int,
    *,
    center=None,
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Linear (raster) train ordering over a Cartesian point set.

    Views are ranked in raster order and dealt echo-major into trains: each
    echo is a raster band, and successive echoes of one train stay neighbours.
    This is scheme A ("linear reordering") of ISMRM abstract 566-05-007,
    Fig. 2.

    Parameters
    ----------
    coords : int or array_like
        Number of sequential views, or phase-encode locations with shape
        ``(N,)`` (ky only) or ``(N, 2)`` (ky, kz).
    train_length : int
        Echo-train or segment length.
    center : tuple of float or None, default=None
        The k-space centre the target echo is placed on. ``None`` uses the
        centroid of ``coords``; a caller with a fixed grid passes its centre.
    center_echo : int or None, default=None
        Echo the k-space centre is acquired at. ``None`` leaves the raster
        bands in order (centre wherever it falls); an integer rolls the bands
        so the one holding the centre plays at that echo -- the effective-TE
        control an echo train needs.
    pad : bool, default=False
        When True every train is padded to ``train_length`` with ``None`` so
        the echo index is the position in the train; when False (the default)
        the gaps are dropped and each train holds only its real views.

    Returns
    -------
    list of list of int
        Shots of view indices (indices into ``coords``); echo index is the
        position within the shot.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> shots = make_linear_order([[0, 0], [1, 0], [0, 1], [1, 1]], 2)
    >>> sorted(i for shot in shots for i in shot)
    [0, 1, 2, 3]
    """
    if np.asarray(coords).ndim == 0:
        count = int(coords)
        if count < 0:
            raise ValueError("coords must be nonnegative when given as a count")
        return _finish_trains(
            _split_into_shots(list(range(count)), train_length), train_length, pad
        )
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, _ = _polar(pts, center)
    order = np.lexsort((pts[:, 0], pts[:, 1]))
    trains = _deal_echo_major(
        order,
        pts[:, 0],
        n_trains,
        train_length,
        radius,
        center_echo=center_echo,
        adaptive=False,
    )
    return _finish_trains(trains, train_length, pad)


def make_centric_order(
    coords,
    train_length: int,
    *,
    center=None,
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Globally center-out Cartesian ordering, dealt into trains.

    This is the conventional centric segmented-GRE/MPRAGE ordering: sampled
    locations are ranked by distance from the encoded k-space centre, so the
    earliest echo of every train clusters near the centre. Unlike
    :func:`make_radial_adaptive_order` the echo bands stay in radius order
    unless ``center_echo`` rolls them to a later effective TE.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of shots is
        ``ceil(N / train_length)``.
    center : tuple of float or None, default=None
        k-space centre; ``None`` uses the centroid of ``coords``.
    center_echo : int or None, default=None
        Echo the k-space centre is acquired at; ``None`` keeps it at the first
        echo.
    pad : bool, default=False
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices, in a single global center-out order.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> ky, kz = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = make_centric_order(coords, 5)
    >>> all(len(s) <= 5 for s in shots)
    True

    See Also
    --------
    make_radial_order, make_linear_order, make_radial_adaptive_order
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, angle = _polar(pts, center)
    order = sorted(range(n), key=lambda index: (radius[index], angle[index]))
    trains = _deal_echo_major(
        order,
        angle,
        n_trains,
        train_length,
        radius,
        center_echo=center_echo,
        adaptive=False,
    )
    return _finish_trains(trains, train_length, pad)


def make_radial_order(
    coords,
    train_length: int,
    *,
    center=None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Center-out radial (wedge) echo-train ordering.

    k-space is partitioned into angular wedges (one per shot). Within each
    wedge the views are ordered by radial distance from the k-space center,
    so every echo train samples the center first and the periphery last —
    scheme B ("radial wedge reordering") of 566-05-007, Fig. 2, and the
    conventional proton-density 3D FSE center-out ordering (Busse et al.).
    The centre of k-space is therefore always at the first echo.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of wedges is
        ``ceil(N / train_length)``.
    center : tuple of float or None, default=None
        k-space centre; ``None`` uses the centroid of ``coords``.
    pad : bool, default=False
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices, each ordered center-out.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> ky, kz = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = make_radial_order(coords, 5)
    >>> all(len(s) <= 5 for s in shots)
    True

    See Also
    --------
    make_linear_order, make_radial_adaptive_order, make_shuffling_order
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    radius, angle = _polar(pts, center)
    n_shots = int(np.ceil(n / train_length))
    # Angular wedges of equal view count keep every echo train the same length.
    by_angle = sorted(range(n), key=lambda i: angle[i])
    shots: list[list[int | None]] = []
    for s in range(n_shots):
        wedge = by_angle[s * train_length : (s + 1) * train_length]
        wedge.sort(key=lambda i: radius[i])
        shots.append(wedge)
    return _finish_trains(shots, train_length, pad)


def make_radial_adaptive_order(
    coords,
    train_length: int,
    *,
    center=None,
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Adaptive radial echo-train ordering (individually parameterized trains).

    Views are ranked by radius and dealt echo-major into trains, but the echo
    a radius band plays at *folds* around ``center_echo``: the innermost band
    coincides with the target echo and the radius grows away from it in both echo
    directions. This is scheme C ("modified radial / adaptive reordering") of
    566-05-007, Fig. 2C-D, which enforces a UI-defined target TE at the centre
    of k-space without a central-k-space discontinuity, and within a band the
    views are ordered angularly so successive echoes stay neighbours.

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo-train or segment length; the number of shots is
        ``ceil(N / train_length)``.
    center : tuple of float or None, default=None
        k-space centre; ``None`` uses the centroid of ``coords``.
    center_echo : int or None, default=None
        Echo the k-space centre is acquired at; ``None`` keeps the innermost
        band at the first echo (equivalent to a plain radial ordering).
    pad : bool, default=False
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices; the echo order runs outward from the target
        echo in both directions.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> ky, kz = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> shots = make_radial_adaptive_order(coords, 7)
    >>> len(shots)
    7
    >>> sum(len(shot) for shot in shots) == len(coords)
    True
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, angle = _polar(pts, center)
    order = sorted(range(n), key=lambda index: (radius[index], angle[index]))
    trains = _deal_echo_major(
        order,
        angle,
        n_trains,
        train_length,
        radius,
        center_echo=0 if center_echo is None else center_echo,
        adaptive=True,
    )
    return _finish_trains(trains, train_length, pad)


def make_shuffling_order(
    coords,
    train_length: int,
    *,
    seed: int | None = None,
    cluster: bool = True,
    pad: bool = False,
) -> list[list[int | None]]:
    """Randomly shuffled echo-train ordering (T2 Shuffling).

    Phase encodes are randomly assigned to echo positions so that k-t space
    is sampled incoherently (Tamir et al., "T2 Shuffling"). To limit
    gradient switching within a train, nearby phase encodes are grouped into
    the same echo train (``cluster=True``) before the echo order within each
    train is randomized — mirroring the paper's mitigation of eddy-current
    effects by "assigning nearby phase encodes to the same echo train".

    Parameters
    ----------
    coords : array_like
        Phase-encode locations, shape ``(N,)`` or ``(N, 2)``.
    train_length : int
        Echo train length.
    seed : int or None, default=None
        Seed for reproducible shuffling.
    cluster : bool, default=True
        When True, group spatially nearby views into the same train before
        randomizing echo order; when False, assign views to trains at random.
    pad : bool, default=False
        Pad each train to ``train_length`` with ``None`` (see
        :func:`make_linear_order`).

    Returns
    -------
    list of list of int
        Shots of view indices in randomized echo order.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> ky, kz = np.meshgrid(np.arange(8), np.arange(8))
    >>> coords = np.column_stack([ky.ravel(), kz.ravel()])
    >>> a = make_shuffling_order(coords, 8, seed=0)
    >>> b = make_shuffling_order(coords, 8, seed=0)
    >>> a == b
    True
    """
    pts = _as_coords(coords)
    n = len(pts)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    n_shots = int(np.ceil(n / train_length))

    # Grid-strip clustering sorts by kz then ky so each train covers a
    # spatially compact region; the alternative starts from a random order.
    base_order = (
        sorted(range(n), key=lambda i: (pts[i, 1], pts[i, 0]))
        if cluster
        else list(rng.permutation(n))
    )

    shots: list[list[int | None]] = []
    for s in range(n_shots):
        train = base_order[s * train_length : (s + 1) * train_length]
        train = list(rng.permutation(train))
        shots.append(train)
    return _finish_trains(shots, train_length, pad)


def make_random_mask(
    shape: tuple[int, int],
    accel: float,
    *,
    calib: tuple[int, int] = (0, 0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a uniform-random undersampling mask with a calibration region.

    Sample ``round(ny * nz / accel)`` locations, retaining the calibration
    block and drawing the remainder uniformly without replacement. If the
    calibration block exceeds this target, retain it without adding samples.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(ny, nz)``.
    accel : float
        Target acceleration factor (> 1).
    calib : tuple of int, default=(0, 0)
        Fully sampled centered calibration shape.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Boolean mask of ``shape``.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> mask = make_random_mask((32, 32), 4.0, calib=(8, 8), seed=0)
    >>> mask.shape
    (32, 32)

    See Also
    --------
    make_poisson_disc_mask : incoherent but locally uniform alternative.
    make_caipirinha_mask : deterministic lattice for parallel imaging.
    calc_sampled_lines : the lines a mask keeps, as an index array.
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
    """Generate a 2D CAIPIRINHA lattice undersampling mask.

    Standard CAIPI shift pattern: row ``ky`` is sampled on the ``rz``-grid
    with a per-``ky``-block shift of ``delta`` along kz, spreading aliasing
    in both phase-encode directions (Breuer et al., MRM 2006).

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(ny, nz)``.
    ry : int
        Acceleration along the first axis.
    rz : int
        Acceleration along the second axis.
    delta : int, default=1
        CAIPI shift applied per sampled-ky step (``0 <= delta < rz``;
        ``delta=0`` degenerates to a regular ``ry x rz`` lattice).

    Returns
    -------
    numpy.ndarray
        Boolean mask with nominal acceleration ``ry * rz``; finite grid
        boundaries can change the realised factor.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> mask = make_caipirinha_mask((8, 8), 2, 2, delta=1)
    >>> int(mask.sum())
    16

    References
    ----------
    Breuer et al., CAIPIRINHA, DOI ``10.1002/mrm.20787``.

    See Also
    --------
    make_skipped_caipi_order : segmented EPI ordering over this lattice.
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
    """Generate a variable-density Poisson-disc undersampling mask.

    Adapted from SigPy's ``sigpy.mri.samp.poisson``: sampling
    density falls off as ``1 / (1 + s|r|)`` with the slope ``s`` found by
    binary search so the realized acceleration matches ``accel`` within
    ``tol``; points are placed with Bridson dart throwing.

    Parameters
    ----------
    shape : tuple of int
        Mask shape ``(ny, nz)``.
    accel : float
        Target acceleration factor (> 1).
    calib : tuple of int, default=(0, 0)
        Fully sampled centered calibration shape.
    seed : int, default=0
        Random seed.
    max_attempts : int, default=30
        Bridson candidate attempts per active point.
    tol : float, default=0.1
        Allowed deviation of the realized acceleration.
    crop_corner : bool, default=True
        Restrict sampling to the inscribed k-space ellipse.

    Returns
    -------
    numpy.ndarray
        Boolean mask of ``shape``.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> mask = make_poisson_disc_mask((48, 48), 4.0, calib=(8, 8), seed=1)
    >>> mask.shape
    (48, 48)

    References
    ----------
    SigPy ``sigpy.mri.samp.poisson`` (BSD 3-Clause); Bridson, SIGGRAPH 2007.
    """
    if accel <= 1:
        raise ValueError(f"accel must be greater than 1, got {accel}")
    from pypulseqpp._ext import sampling

    return sampling.poisson_disc_mask(
        shape, accel, calib, seed, max_attempts, tol, crop_corner
    )
