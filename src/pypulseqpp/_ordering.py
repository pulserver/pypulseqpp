"""Temporal ordering of an already selected set of views.

Two kinds of ordering are defined here. :func:`make_traversal_order` returns
a permutation of the positions of one loop axis. The echo-train orderings
(``make_*_order``) take centred view coordinates, with the k-space centre at
the origin, assign their rows to shots and echoes, and return indices into
that array, never coordinates. None of them selects views or creates labels.

References
----------
Linear, radial and adaptive radial echo-train reordering follow schemes A-C of
Buonincontri et al., ISMRM abstract 566-05-007, Fig. 2. Shuffled echo trains
follow Tamir et al., Magn Reson Med 2017;77:180-195.
"""

from __future__ import annotations

__all__ = [
    "make_centric_order",
    "make_linear_order",
    "make_radial_adaptive_order",
    "make_radial_order",
    "make_shuffling_order",
    "make_traversal_order",
]

import numpy as np


def _count(n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return n


def _sequential(n: int) -> np.ndarray:
    return np.arange(_count(n), dtype=np.intp)


def _reverse(n: int) -> np.ndarray:
    return _sequential(n)[::-1].copy()


def _interleaved(n: int) -> np.ndarray:
    values = _sequential(n)
    return np.concatenate((values[::2], values[1::2]))


def _center_out(n: int) -> np.ndarray:
    """Order positions by distance from the centre, the lower index first on a tie."""
    n = _count(n)
    center = (n - 1) / 2.0
    return np.asarray(
        sorted(range(n), key=lambda index: (abs(index - center), index)), dtype=np.intp
    )


def _outside_in(n: int) -> np.ndarray:
    return _center_out(n)[::-1].copy()


def _random_order(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(_count(n))


_TRAVERSALS = {
    "sequential": _sequential,
    "reverse": _reverse,
    "interleaved": _interleaved,
    "center_out": _center_out,
    "outside_in": _outside_in,
}


def make_traversal_order(
    n: int, order: str = "sequential", *, seed: int = 0
) -> np.ndarray:
    """Return the order in which a loop visits ``n`` positions.

    The result is a permutation ``p`` of ``range(n)``: the loop visits
    position ``p[0]`` first, ``p[1]`` second, and so on. The positions are
    abstract indices, typically slices, or the entries of a list of selected
    lines or partitions; ``[items[i] for i in p]`` reorders such a list.

    Parameters
    ----------
    n : int
        Number of positions.
    order : {'sequential', 'reverse', 'interleaved', 'center_out', 'outside_in', 'random'}, default='sequential'
        Traversal scheme. ``'interleaved'`` visits the even positions and
        then the odd ones, the conventional multi-slice order.
        ``'center_out'`` starts at the middle position and alternates
        outward, the lower position first on a tie; ``'outside_in'`` is its
        reverse. ``'random'`` is a seeded random permutation.
    seed : int, default=0
        Seed for ``order='random'``.

    Returns
    -------
    numpy.ndarray
        Integer permutation of ``range(n)``, in visiting order.

    Raises
    ------
    ValueError
        If ``n`` is negative or ``order`` is unknown.

    See Also
    --------
    make_cartesian_axis_sampling : selection of the views on one axis.
    make_centric_order : assignment of views to echo trains.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> pp.make_traversal_order(6, "interleaved").tolist()
    [0, 2, 4, 1, 3, 5]

    The permutation reorders an existing list, here the partitions of a
    stack-of-stars acquisition:

    >>> partitions = [10, 11, 12, 13, 14]
    >>> order = pp.make_traversal_order(len(partitions), "center_out")
    >>> order.tolist()
    [2, 1, 3, 0, 4]
    >>> [partitions[i] for i in order]
    [12, 11, 13, 10, 14]
    """
    if order == "random":
        return _random_order(n, seed)
    try:
        return _TRAVERSALS[order](n)
    except KeyError:
        raise ValueError(
            f"unknown order {order!r}; expected one of {', '.join(sorted((*_TRAVERSALS, 'random')))}"
        ) from None


def _centred(coords) -> np.ndarray:
    """Return centred view coordinates as an ``(N, 2)`` float array.

    ``(N,)`` input is a set of ``ky`` offsets with ``kz = 0``. Boolean input is
    refused: its indices would refer to an array the caller never sees.
    """
    raw = np.asarray(coords)
    if raw.dtype == bool:
        raise TypeError(
            "coords must be numeric centred coordinates, not a boolean mask; "
            "convert a mask with np.argwhere(mask) and subtract the k-space centre"
        )
    if raw.size == 0:
        return np.zeros((0, 2))
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("coords must be a numeric array")
    arr = raw.astype(float)
    if arr.ndim == 1:
        arr = np.column_stack([arr, np.zeros_like(arr)])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords must have shape (N,) or (N, 2), got {raw.shape}")
    return arr


def _checked_train(train_length, center_echo=None) -> int:
    """Validate the echo-train length and target echo shared by every ordering."""
    if isinstance(train_length, bool) or int(train_length) != train_length:
        raise ValueError(f"train_length must be an integer, got {train_length!r}")
    train_length = int(train_length)
    if train_length < 1:
        raise ValueError(f"train_length must be at least 1, got {train_length}")
    if center_echo is not None and not 0 <= center_echo < train_length:
        raise ValueError(
            f"center_echo must lie in [0, {train_length}), got {center_echo}"
        )
    return train_length


def _polar(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each view's distance from, and polar angle about, the origin."""
    return np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])


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
    of one train stay neighbours. ``center_echo`` places the group containing
    the view nearest the origin at that echo: by rotating the group order for
    a monotone ranking, or by folding it (``adaptive``) so the radius grows
    away from the target echo in both directions. This is the shared
    implementation of the Cartesian echo-train orderings (566-05-007, Fig. 2).
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
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Echo-train ordering of Cartesian views in linear (raster) order.

    The views are ranked in raster order, by ``kz`` and then by ``ky``, and
    the ranking is cut into ``train_length`` consecutive bands of
    ``ceil(N / train_length)`` views. Band ``e`` is acquired at echo ``e``,
    one view per shot, dealt across the shots in ``ky`` order. This is
    scheme A (linear reordering) of Buonincontri et al., Fig. 2.

    Parameters
    ----------
    coords : array_like
        Centred coordinates of the selected views: an ``(N, 2)`` array of
        ``(ky, kz)`` or an ``(N,)`` array of ``ky``, relative to the k-space
        centre, which is the origin. Encoded view indices ``(y, z)`` on an
        ``(n_y, n_z)`` grid convert as ``views - (n_y // 2, n_z // 2)``.
        Distances and angles are evaluated in the units supplied. Boolean
        masks are not accepted.
    train_length : int
        Echo-train length, at least 1. The number of shots is
        ``ceil(N / train_length)``.
    center_echo : int or None, default=None
        Echo at which the view nearest the origin is acquired, in
        ``[0, train_length)``. ``None`` keeps the bands in raster order; an
        integer rotates the band order so that the band containing that view
        is acquired at ``center_echo``.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``, so that the
        position in a train is the echo index. With ``False`` the ``None``
        entries are removed.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view
        acquired at echo ``e`` of shot ``s``. The values are indices, not
        coordinates; every row of ``coords`` appears exactly once. With
        ``pad=True``, ``None`` marks an echo that acquires no view. Empty
        ``coords`` give ``[]``.

    Raises
    ------
    TypeError
        If ``coords`` is a boolean mask or not numeric.
    ValueError
        If ``coords`` does not have shape ``(N,)`` or ``(N, 2)``,
        ``train_length`` is not a positive integer, or ``center_echo`` is
        outside ``[0, train_length)``.

    See Also
    --------
    make_centric_order, make_radial_order, make_radial_adaptive_order,
    make_shuffling_order

    Examples
    --------
    Four centred views, two echoes per shot. Echo 0 acquires the ``kz = -1``
    band and echo 1 the ``kz = 0`` band:

    >>> import pypulseqpp as pp
    >>> views = [(-1, -1), (0, -1), (-1, 0), (0, 0)]
    >>> trains = pp.make_linear_order(views, 2)
    >>> trains
    [[0, 2], [1, 3]]
    >>> [[views[i] for i in train] for train in trains]
    [[(-1, -1), (-1, 0)], [(0, -1), (0, 0)]]
    """
    train_length = _checked_train(train_length, center_echo)
    pts = _centred(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, _ = _polar(pts)
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
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Echo-train ordering of Cartesian views in centric order.

    The views are ranked by their distance from the origin (the k-space
    centre), ties broken by polar angle, and the ranking is cut into
    ``train_length`` consecutive bands of ``ceil(N / train_length)`` views.
    Band ``e`` is acquired at echo ``e``, one view per shot, dealt across the
    shots in angle order. The first echo of every shot therefore acquires a
    view near the centre: the conventional centric ordering of segmented
    gradient-echo and MPRAGE acquisitions.

    Parameters
    ----------
    coords : array_like
        Centred coordinates of the selected views: an ``(N, 2)`` array of
        ``(ky, kz)`` or an ``(N,)`` array of ``ky``, relative to the k-space
        centre, which is the origin. Encoded view indices ``(y, z)`` on an
        ``(n_y, n_z)`` grid convert as ``views - (n_y // 2, n_z // 2)``.
        Distances and angles are evaluated in the units supplied. Boolean
        masks are not accepted.
    train_length : int
        Echo-train length, at least 1. The number of shots is
        ``ceil(N / train_length)``.
    center_echo : int or None, default=None
        Echo at which the innermost band is acquired, in
        ``[0, train_length)``. ``None`` acquires it at echo 0; an integer
        rotates the band order, which moves the effective echo time without
        changing the train membership.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``, so that the
        position in a train is the echo index. With ``False`` the ``None``
        entries are removed.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view
        acquired at echo ``e`` of shot ``s``. The values are indices, not
        coordinates; every row of ``coords`` appears exactly once. With
        ``pad=True``, ``None`` marks an echo that acquires no view. Empty
        ``coords`` give ``[]``.

    Raises
    ------
    TypeError
        If ``coords`` is a boolean mask or not numeric.
    ValueError
        If ``coords`` does not have shape ``(N,)`` or ``(N, 2)``,
        ``train_length`` is not a positive integer, or ``center_echo`` is
        outside ``[0, train_length)``.

    See Also
    --------
    make_radial_order : centre-out order within angular wedges.
    make_radial_adaptive_order : distance order folded about a target echo.

    Examples
    --------
    Five centred ``ky`` views in shots of two echoes:

    >>> import pypulseqpp as pp
    >>> views = [-2, -1, 0, 1, 2]
    >>> trains = pp.make_centric_order(views, 2)
    >>> trains
    [[2, 4], [3, 0], [1]]
    >>> [[views[i] for i in train] for train in trains]
    [[0, 2], [1, -2], [-1]]

    Echo 0 acquires the three views nearest the centre, one per shot, and
    echo 1 the two outer views. The third shot has no second view.
    """
    train_length = _checked_train(train_length, center_echo)
    pts = _centred(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, angle = _polar(pts)
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
    pad: bool = False,
) -> list[list[int | None]]:
    """Echo-train ordering of Cartesian views in centre-out radial order.

    This orders Cartesian phase-encoding views, the radial view ordering of
    3D fast spin echo; it does not concern radial (projection) trajectories.
    The views are sorted by polar angle about the origin and cut into
    ``ceil(N / train_length)`` angular wedges of ``train_length`` views; each
    wedge is one shot. Within a wedge the views are acquired in order of
    increasing distance from the origin, so every shot acquires its view
    nearest the centre at echo 0. This is scheme B (radial wedge
    reordering) of Buonincontri et al., Fig. 2.

    Parameters
    ----------
    coords : array_like
        Centred coordinates of the selected views: an ``(N, 2)`` array of
        ``(ky, kz)`` or an ``(N,)`` array of ``ky``, relative to the k-space
        centre, which is the origin. Encoded view indices ``(y, z)`` on an
        ``(n_y, n_z)`` grid convert as ``views - (n_y // 2, n_z // 2)``.
        Distances and angles are evaluated in the units supplied. Boolean
        masks are not accepted.
    train_length : int
        Echo-train length, at least 1. The number of shots is
        ``ceil(N / train_length)``.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``, so that the
        position in a train is the echo index. With ``False`` the ``None``
        entries are removed.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view
        acquired at echo ``e`` of shot ``s``. The values are indices, not
        coordinates; every row of ``coords`` appears exactly once. With
        ``pad=True``, ``None`` marks an echo that acquires no view. Empty
        ``coords`` give ``[]``.

    Raises
    ------
    TypeError
        If ``coords`` is a boolean mask or not numeric.
    ValueError
        If ``coords`` does not have shape ``(N,)`` or ``(N, 2)``, or
        ``train_length`` is not a positive integer.

    See Also
    --------
    make_radial_adaptive_order : a target echo other than the first.

    Examples
    --------
    Eight centred views on a 3 x 3 ky-kz grid without its centre, in shots
    of four. Each shot is a half-plane wedge, acquired from the inner to the
    outer views:

    >>> import pypulseqpp as pp
    >>> views = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
    ...          (0, 1), (1, -1), (1, 0), (1, 1)]
    >>> trains = pp.make_radial_order(views, 4)
    >>> trains
    [[3, 6, 0, 5], [4, 1, 7, 2]]
    >>> [[views[i] for i in train] for train in trains]
    [[(0, -1), (1, 0), (-1, -1), (1, -1)], [(0, 1), (-1, 0), (1, 1), (-1, 1)]]
    """
    train_length = _checked_train(train_length)
    pts = _centred(coords)
    n = len(pts)
    if n == 0:
        return []
    radius, angle = _polar(pts)
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
    center_echo: int | None = None,
    pad: bool = False,
) -> list[list[int | None]]:
    """Echo-train ordering of Cartesian views in radial order about a target echo.

    The views are ranked by their distance from the origin (the k-space
    centre) and cut into ``train_length`` radius bands of
    ``ceil(N / train_length)`` views. The innermost band is acquired at
    ``center_echo`` and successive bands at the echoes increasingly distant
    from it, alternating before and after, so the distance from the centre
    increases monotonically away from the target echo in both directions.
    Within a band the views are dealt across the shots in polar-angle order.
    This is scheme C (modified radial reordering) of Buonincontri et al.,
    Fig. 2C-D, which acquires the k-space centre at a prescribed echo without
    a discontinuity at the centre.

    Parameters
    ----------
    coords : array_like
        Centred coordinates of the selected views: an ``(N, 2)`` array of
        ``(ky, kz)`` or an ``(N,)`` array of ``ky``, relative to the k-space
        centre, which is the origin. Encoded view indices ``(y, z)`` on an
        ``(n_y, n_z)`` grid convert as ``views - (n_y // 2, n_z // 2)``.
        Distances and angles are evaluated in the units supplied. Boolean
        masks are not accepted.
    train_length : int
        Echo-train length, at least 1. The number of shots is
        ``ceil(N / train_length)``.
    center_echo : int or None, default=None
        Echo at which the innermost band, and so the view nearest the
        origin, is acquired, in ``[0, train_length)``. ``None`` is echo 0.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``, so that the
        position in a train is the echo index. With ``False`` the ``None``
        entries are removed.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view
        acquired at echo ``e`` of shot ``s``. The values are indices, not
        coordinates; every row of ``coords`` appears exactly once. With
        ``pad=True``, ``None`` marks an echo that acquires no view. Empty
        ``coords`` give ``[]``.

    Raises
    ------
    TypeError
        If ``coords`` is a boolean mask or not numeric.
    ValueError
        If ``coords`` does not have shape ``(N,)`` or ``(N, 2)``,
        ``train_length`` is not a positive integer, or ``center_echo`` is
        outside ``[0, train_length)``.

    See Also
    --------
    make_radial_order : centre-out order within angular wedges.
    make_centric_order : distance order rotated rather than folded.

    Examples
    --------
    Nine centred ``ky`` views in three shots of three echoes, with the
    centre view acquired at echo 1:

    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> views = np.arange(-4, 5)
    >>> trains = pp.make_radial_adaptive_order(views, 3, center_echo=1)
    >>> trains
    [[6, 4, 8], [7, 5, 1], [2, 3, 0]]
    >>> views[np.array(trains)]
    array([[ 2,  0,  4],
           [ 3,  1, -3],
           [-2, -1, -4]])

    The distance from the centre is smallest at echo 1 in every shot:

    >>> np.abs(views[np.array(trains)]).argmin(axis=1).tolist()
    [1, 1, 1]
    """
    train_length = _checked_train(train_length, center_echo)
    pts = _centred(coords)
    n = len(pts)
    if n == 0:
        return []
    n_trains = int(np.ceil(n / train_length))
    radius, angle = _polar(pts)
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
    """Echo-train ordering of Cartesian views with random echo positions.

    This is the echo ordering used by T2 Shuffling [1]_. With
    ``cluster=True``, train membership is formed from contiguous views in
    raster order (``kz``, then ``ky``); echo positions within each train are
    randomly permuted. With ``cluster=False``, train membership is a random
    partition of the views as well.

    The routine orders views that are already selected; it does not select a
    variable-density support. ``make_cartesian_plane_sampling`` with
    ``sampling='poisson'`` selects such a support.

    Parameters
    ----------
    coords : array_like
        Centred coordinates of the selected views: an ``(N, 2)`` array of
        ``(ky, kz)`` or an ``(N,)`` array of ``ky``, relative to the k-space
        centre, which is the origin. Encoded view indices ``(y, z)`` on an
        ``(n_y, n_z)`` grid convert as ``views - (n_y // 2, n_z // 2)``.
        Distances and angles are evaluated in the units supplied. Boolean
        masks are not accepted.
    train_length : int
        Echo-train length, at least 1. The number of shots is
        ``ceil(N / train_length)``.
    seed : int or None, default=None
        Seed of the random permutations. Equal seeds give equal orders.
    cluster : bool, default=True
        Form each train from contiguous views in raster order; when
        ``False``, assign views to trains at random.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``, so that the
        position in a train is the echo index. With ``False`` the ``None``
        entries are removed.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view
        acquired at echo ``e`` of shot ``s``. The values are indices, not
        coordinates; every row of ``coords`` appears exactly once. With
        ``pad=True``, ``None`` marks an echo that acquires no view. Empty
        ``coords`` give ``[]``.

    Raises
    ------
    TypeError
        If ``coords`` is a boolean mask or not numeric.
    ValueError
        If ``coords`` does not have shape ``(N,)`` or ``(N, 2)``, or
        ``train_length`` is not a positive integer.

    See Also
    --------
    make_linear_order : the same train membership in raster echo order.

    References
    ----------
    .. [1] Tamir JI, Uecker M, Chen W, et al. T2 shuffling: sharp,
       multicontrast, volumetric fast spin-echo imaging. *Magnetic Resonance
       in Medicine*. 2017;77(1):180-195. https://doi.org/10.1002/mrm.26102

    Examples
    --------
    Six centred ``ky`` views in two shots of three echoes. Each shot
    contains a contiguous group of views; the echo at which each view is
    acquired is random:

    >>> import pypulseqpp as pp
    >>> views = [-3, -2, -1, 0, 1, 2]
    >>> trains = pp.make_shuffling_order(views, 3, seed=0)
    >>> trains
    [[2, 0, 1], [5, 4, 3]]
    >>> [[views[i] for i in train] for train in trains]
    [[-1, -3, -2], [2, 1, 0]]

    Another seed keeps the train membership and changes the echo positions:

    >>> other = pp.make_shuffling_order(views, 3, seed=1)
    >>> [[views[i] for i in train] for train in other]
    [[-3, -2, -1], [2, 0, 1]]
    """
    train_length = _checked_train(train_length)
    pts = _centred(coords)
    n = len(pts)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    n_shots = int(np.ceil(n / train_length))

    # Raster order, kz then ky, so each train is a contiguous group of views;
    # without clustering the membership starts from a random order.
    base_order = (
        sorted(range(n), key=lambda i: (pts[i, 1], pts[i, 0]))
        if cluster
        else rng.permutation(n).tolist()
    )

    shots: list[list[int | None]] = []
    for s in range(n_shots):
        train = base_order[s * train_length : (s + 1) * train_length]
        shots.append(rng.permutation(train).tolist())
    return _finish_trains(shots, train_length, pad)
