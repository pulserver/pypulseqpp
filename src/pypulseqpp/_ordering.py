"""Temporal ordering of an already selected set of views.

Two kinds of ordering are defined here. :func:`make_traversal_order` returns
a permutation of the positions of one loop axis. The echo-train orderings
(``make_*_order``) assign the rows of a coordinate array to shots and echoes
and return indices into that array, never coordinates. None of them selects
views or emits labels.

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


def _as_coords(coords) -> np.ndarray:
    """Normalize a coordinate argument to an ``(N, 2)`` float array."""
    raw = np.asarray(coords)
    if raw.dtype == bool and raw.ndim in (1, 2):
        arr = np.argwhere(raw).astype(float)
        if raw.ndim == 1:
            arr = arr[:, 0]
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
    """Assign selected views to echo trains in linear (raster) order.

    The views are ranked in raster order, by ``kz`` and then by ``ky``, and
    the ranking is cut into ``train_length`` consecutive bands of
    ``ceil(N / train_length)`` views. Band ``e`` is acquired at echo ``e``,
    one view per train, dealt across the trains in ``ky`` order. This is
    scheme A (linear reordering) of Buonincontri et al., Fig. 2.

    Parameters
    ----------
    coords : int or array_like
        The selected views: an ``(N, 2)`` array of ``(ky, kz)`` coordinates,
        an ``(N,)`` array of ``ky`` coordinates, or a boolean mask whose
        ``True`` entries are taken in :func:`numpy.argwhere` order. An integer
        ``N`` instead splits ``range(N)`` into consecutive trains.
    train_length : int
        Echo-train length: the number of echoes per shot.
    center : tuple of float or None, default=None
        k-space centre, in the units of ``coords``. ``None`` uses the centroid
        of ``coords``.
    center_echo : int or None, default=None
        Echo at which the view nearest ``center`` is acquired. ``None`` keeps
        the bands in raster order; an integer rotates the band order so the
        band containing that view is acquired at ``center_echo``.
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
        ``pad=True``, ``None`` marks an echo that acquires no view.

    Raises
    ------
    ValueError
        If ``coords`` is not a count, an ``(N,)`` or ``(N, 2)`` array, or a
        boolean mask.

    See Also
    --------
    make_centric_order, make_radial_order, make_radial_adaptive_order,
    make_shuffling_order

    Examples
    --------
    Four views, two echoes per train. Echo 0 acquires the ``kz = 0`` band and
    echo 1 the ``kz = 1`` band:

    >>> import pypulseqpp as pp
    >>> views = [(0, 0), (1, 0), (0, 1), (1, 1)]
    >>> trains = pp.make_linear_order(views, 2)
    >>> trains
    [[0, 2], [1, 3]]
    >>> [[views[i] for i in train] for train in trains]
    [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
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
    """Assign selected views to echo trains in centric order.

    The views are ranked by their distance from ``center``, ties broken by
    polar angle, and the ranking is cut into ``train_length`` consecutive
    bands of ``ceil(N / train_length)`` views. Band ``e`` is acquired at echo
    ``e``, one view per train, dealt across the trains in angle order. The
    first echo of every train therefore acquires a view near the centre: the
    conventional centric ordering of segmented gradient-echo and MPRAGE
    acquisitions.

    Parameters
    ----------
    coords : array_like
        The selected views, as for :func:`make_linear_order`.
    train_length : int
        Echo-train length; the number of shots is ``ceil(N / train_length)``.
    center : tuple of float or None, default=None
        k-space centre, in the units of ``coords``. ``None`` uses the centroid
        of ``coords``.
    center_echo : int or None, default=None
        Echo at which the innermost band is acquired. ``None`` acquires it at
        echo 0; an integer rotates the band order, which moves the effective
        echo time without changing the train membership.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view acquired
        at echo ``e`` of shot ``s``, as for :func:`make_linear_order`.

    See Also
    --------
    make_radial_order : centre-out order within angular wedges.
    make_radial_adaptive_order : distance order folded about a target echo.

    Examples
    --------
    Five views on the ky axis, one centre view and two per side, in trains of
    length two:

    >>> import pypulseqpp as pp
    >>> views = [-2, -1, 0, 1, 2]
    >>> trains = pp.make_centric_order(views, 2, center=(0, 0))
    >>> trains
    [[2, 4], [3, 0], [1]]
    >>> [[views[i] for i in train] for train in trains]
    [[0, 2], [1, -2], [-1]]

    Echo 0 acquires the three views nearest the centre, one per train, and
    echo 1 the two outer views. The third train has no second view.
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
    """Assign selected views to echo trains in centre-out radial order.

    The views are sorted by polar angle about ``center`` and cut into
    ``ceil(N / train_length)`` angular wedges of ``train_length`` views; each
    wedge is one shot. Within a wedge the views are acquired in order of
    increasing distance from ``center``, so every train acquires its view
    nearest the centre at echo 0. This is scheme B (radial wedge reordering)
    of Buonincontri et al., Fig. 2, the centre-out ordering of 3D fast spin
    echo with a short effective echo time.

    Parameters
    ----------
    coords : array_like
        The selected views, as for :func:`make_linear_order`.
    train_length : int
        Echo-train length; the number of wedges is ``ceil(N / train_length)``.
    center : tuple of float or None, default=None
        k-space centre, in the units of ``coords``. ``None`` uses the centroid
        of ``coords``.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view acquired
        at echo ``e`` of shot ``s``, as for :func:`make_linear_order`.

    See Also
    --------
    make_radial_adaptive_order : a target echo other than the first.

    Examples
    --------
    Eight views on a 3 x 3 ky-kz grid without its centre, in trains of four.
    Each train is a half-plane wedge, acquired from the inner to the outer
    views:

    >>> import pypulseqpp as pp
    >>> views = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
    ...          (0, 1), (1, -1), (1, 0), (1, 1)]
    >>> trains = pp.make_radial_order(views, 4, center=(0, 0))
    >>> trains
    [[3, 6, 0, 5], [4, 1, 7, 2]]
    >>> [[views[i] for i in train] for train in trains]
    [[(0, -1), (1, 0), (-1, -1), (1, -1)], [(0, 1), (-1, 0), (1, 1), (-1, 1)]]
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
    """Assign selected views to echo trains in radial order about a target echo.

    The views are ranked by their distance from ``center`` and cut into
    ``train_length`` radius bands of ``ceil(N / train_length)`` views. The
    innermost band is acquired at ``center_echo`` and successive bands at the
    echoes increasingly distant from it, alternating before and after, so the
    distance from the centre increases monotonically away from the target
    echo in both directions. Within a band the views are dealt across the
    trains in polar-angle order. This is scheme C (modified radial
    reordering) of Buonincontri et al., Fig. 2C-D, which acquires the k-space
    centre at a prescribed echo without a discontinuity at the centre.

    Parameters
    ----------
    coords : array_like
        The selected views, as for :func:`make_linear_order`.
    train_length : int
        Echo-train length; the number of shots is ``ceil(N / train_length)``.
    center : tuple of float or None, default=None
        k-space centre, in the units of ``coords``. ``None`` uses the centroid
        of ``coords``.
    center_echo : int or None, default=None
        Echo at which the innermost band, and so the view nearest ``center``,
        is acquired. ``None`` is echo 0.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view acquired
        at echo ``e`` of shot ``s``, as for :func:`make_linear_order`.

    See Also
    --------
    make_radial_order : centre-out order within angular wedges.
    make_centric_order : distance order rotated rather than folded.

    Examples
    --------
    Nine views on the ky axis in three trains of three echoes, with the
    centre view acquired at echo 1:

    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> views = np.arange(-4, 5)
    >>> trains = pp.make_radial_adaptive_order(
    ...     views, 3, center=(0, 0), center_echo=1
    ... )
    >>> trains
    [[6, 4, 8], [7, 5, 1], [2, 3, 0]]
    >>> views[np.array(trains)]
    array([[ 2,  0,  4],
           [ 3,  1, -3],
           [-2, -1, -4]])

    The distance from the centre is smallest at echo 1 in every train:

    >>> np.abs(views[np.array(trains)]).argmin(axis=1).tolist()
    [1, 1, 1]
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
    """Assign selected views to echo trains in randomly shuffled echo order.

    This is the echo ordering of T2 Shuffling: each view is acquired at a
    random echo position, so that every echo time samples an incoherent
    subset of the selected views and an echo-resolved reconstruction can
    recover the signal evolution along the train. With ``cluster=True`` the
    trains are consecutive raster-order (``kz``, then ``ky``) groups of
    ``train_length`` views, which keeps the views of one train close
    together as in the published method; only the echo positions within each
    train are random. With ``cluster=False`` the train membership is random
    as well.

    The routine orders views that are already selected; it does not choose a
    variable-density support. :func:`make_cartesian_plane_sampling` with
    ``sampling='poisson'`` selects that support.

    Parameters
    ----------
    coords : array_like
        The selected views, as for :func:`make_linear_order`.
    train_length : int
        Echo-train length; the number of shots is ``ceil(N / train_length)``.
    seed : int or None, default=None
        Seed of the random permutations. Equal seeds give equal orders.
    cluster : bool, default=True
        Form each train from a contiguous raster-order group of views; when
        ``False``, assign views to trains at random.
    pad : bool, default=False
        Pad every train to ``train_length`` with ``None``.

    Returns
    -------
    trains : list of list of int
        ``trains[s][e]`` is the row index into ``coords`` of the view acquired
        at echo ``e`` of shot ``s``, as for :func:`make_linear_order`.

    See Also
    --------
    make_linear_order : the same train membership in raster echo order.

    Examples
    --------
    Six views on the ky axis in two trains of three. Each train contains a
    contiguous group of views; the echo at which each view is acquired is
    random:

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
        else rng.permutation(n).tolist()
    )

    shots: list[list[int | None]] = []
    for s in range(n_shots):
        train = base_order[s * train_length : (s + 1) * train_length]
        shots.append(rng.permutation(train).tolist())
    return _finish_trains(shots, train_length, pad)
