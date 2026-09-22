"""The contracts of the sampling API: support, masks, orderings and EPI offsets.

Support routines return encoded coordinates or boolean masks and nothing else;
ordering routines take centred coordinates, with the k-space centre at the
origin, and return indices into the array they are given; the EPI
routine returns offsets relative to a shot origin the caller chooses. These
tests state those contracts rather than particular outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp

ORDERINGS = [
    pp.make_linear_order,
    pp.make_centric_order,
    pp.make_radial_order,
    pp.make_radial_adaptive_order,
    pp.make_shuffling_order,
]

#: A 6 x 5 ky-kz grid centred on (0, 0), with its corners removed.
GRID = np.array(
    [
        (y, z)
        for y in range(-3, 3)
        for z in range(-2, 3)
        if (y / 3) ** 2 + (z / 2.5) ** 2 <= 1.2
    ]
)


# Support -------------------------------------------------------------------


@pytest.mark.parametrize("n", [15, 16])
@pytest.mark.parametrize("acceleration", [1, 2, 3])
@pytest.mark.parametrize("n_acs", [0, 4])
@pytest.mark.parametrize("partial_fourier", [1.0, 0.75])
def test_axis_support_is_disjoint_in_range_and_keeps_the_centre(
    n, acceleration, n_acs, partial_fourier
):
    calibration, imaging = pp.make_cartesian_axis_sampling(
        n, acceleration, n_acs, partial_fourier=partial_fourier
    )

    assert not set(calibration) & set(imaging)
    assert all(0 <= view < n for view in [*calibration, *imaging])
    assert n // 2 in {*calibration, *imaging}
    assert calibration == sorted(calibration) and imaging == sorted(imaging)


def test_axis_partial_fourier_removes_the_lowest_views_only():
    _, full = pp.make_cartesian_axis_sampling(16)
    _, partial = pp.make_cartesian_axis_sampling(16, partial_fourier=0.75)

    assert partial == full[4:]


@pytest.mark.parametrize("sampling", ["lattice", "poisson"])
@pytest.mark.parametrize("elliptical", [False, True])
def test_plane_support_is_disjoint_in_range_and_keeps_the_centre(sampling, elliptical):
    shape = (24, 16)
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape,
        (2, 2),
        (6, 4),
        caipi_shift=1,
        partial_fourier=(0.75, 1.0),
        elliptical=elliptical,
        sampling=sampling,
        seed=5,
    )

    assert not set(calibration) & set(imaging)
    assert all(
        0 <= y < shape[0] and 0 <= z < shape[1] for y, z in calibration + imaging
    )
    assert (12, 8) in {*calibration, *imaging}
    assert min(y for y, _ in calibration + imaging) >= 24 - round(0.75 * 24)


def test_plane_support_is_the_lattice_within_partial_fourier_plus_the_acs_block():
    n_y, n_z, ry, rz, shift = 16, 12, 2, 3, 1
    first_y = n_y - round(0.75 * n_y)
    calibration, imaging = pp.make_cartesian_plane_sampling(
        (n_y, n_z), (ry, rz), (4, 2), caipi_shift=shift, partial_fourier=(0.75, 1.0)
    )
    lattice = {
        (y, z)
        for y in range(first_y, n_y)
        for z in range(n_z)
        if (y - n_y // 2) % ry == 0
        and (z - n_z // 2 - shift * ((y - n_y // 2) // ry)) % rz == 0
    }
    acs = {(y, z) for y in range(6, 10) for z in range(5, 7)}

    assert {*calibration, *imaging} == lattice | acs
    assert set(calibration) == acs


def test_plane_calibration_is_the_fully_sampled_centred_block():
    calibration, _ = pp.make_cartesian_plane_sampling((16, 12), (2, 3), (4, 2))

    assert sorted(calibration) == [(y, z) for y in range(6, 10) for z in range(5, 7)]


def test_plane_lattice_is_the_caipirinha_lattice_through_the_centre():
    n_y, n_z, ry, rz, shift = 12, 9, 2, 3, 1
    _, imaging = pp.make_cartesian_plane_sampling(
        (n_y, n_z), (ry, rz), caipi_shift=shift
    )
    mask = pp.make_caipirinha_mask((n_y, n_z), ry, rz, delta=shift)
    # The mask is anchored at (0, 0) and the support at the centre view.
    shifted = np.roll(mask, (n_y // 2, n_z // 2), axis=(0, 1))

    assert sorted(imaging) == sorted(map(tuple, np.argwhere(shifted).tolist()))


def test_plane_poisson_support_is_the_poisson_disc_draw():
    shape, seed = (24, 24), 3
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape, (2, 2), (6, 6), sampling="poisson", seed=seed
    )
    mask = pp.make_poisson_disc_mask(
        shape, 4.0, calib=(6, 6), seed=seed, crop_corner=False
    )

    assert sorted(calibration + imaging) == sorted(
        map(tuple, np.argwhere(mask).tolist())
    )


def test_an_unknown_support_scheme_is_refused():
    with pytest.raises(ValueError, match="sampling"):
        pp.make_cartesian_plane_sampling((8, 8), (2, 2), sampling="shuffling")


def _inside(views, shape, extent=None):
    """Whether each view lies in the ellipse inscribed in ``extent`` about the centre."""
    n_y, n_z = shape
    e_y, e_z = extent or shape
    views = np.asarray(views)
    dy = (views[:, 0] - n_y // 2) / e_y
    dz = (views[:, 1] - n_z // 2) / e_z
    return dy * dy + dz * dz <= 0.25


@pytest.mark.parametrize("sampling", ["lattice", "poisson"])
@pytest.mark.parametrize("elliptical", [False, True])
@pytest.mark.parametrize("elliptical_acs", [False, True])
def test_the_ellipse_arguments_mean_the_same_for_both_schemes(
    sampling, elliptical, elliptical_acs
):
    shape, acs = (32, 32), (10, 10)
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape,
        (2, 2),
        acs,
        elliptical=elliptical,
        elliptical_acs=elliptical_acs,
        sampling=sampling,
        seed=2,
    )
    rows = range(16 - 5, 16 + 5)
    block = {(y, z) for y in rows for z in rows}

    # The calibration region is the rectangle, or the ellipse inscribed in it.
    expected = (
        {v for v in block if _inside([v], shape, acs)[0]} if elliptical_acs else block
    )
    assert set(calibration) == expected
    # ``elliptical`` removes imaging views outside the inscribed ellipse, and
    # only ``elliptical`` does.
    outside = ~_inside(imaging, shape)
    assert not outside.any() if elliptical else outside.any()


def test_poisson_support_with_an_elliptical_acs_does_not_fill_its_rectangle():
    shape, acs = (48, 48), (12, 12)
    rows = range(24 - 6, 24 + 6)
    corners = {
        (y, z) for y in rows for z in rows if not _inside([(y, z)], shape, acs)[0]
    }
    filled = []
    for seed in range(6):
        calibration, imaging = pp.make_cartesian_plane_sampling(
            shape, (2, 2), acs, elliptical_acs=True, sampling="poisson", seed=seed
        )
        filled.append(corners <= {*calibration, *imaging})
    rectangular = pp.make_cartesian_plane_sampling(
        shape, (2, 2), acs, sampling="poisson", seed=0
    )

    assert corners <= set(rectangular[0])
    assert not all(filled)


def test_poisson_support_without_an_ellipse_reaches_the_corners_of_the_grid():
    _, imaging = pp.make_cartesian_plane_sampling(
        (32, 32), (2, 2), (8, 8), sampling="poisson", seed=1
    )

    assert (~_inside(imaging, (32, 32))).sum() > 10


@pytest.mark.parametrize(("n", "c"), [(16, 1), (16, 3), (15, 3), (15, 4), (16, 4)])
def test_the_poisson_calibration_block_is_centred_on_the_centre_view(n, c):
    mask = pp.make_poisson_disc_mask((n, n), 4.0, calib=(c, c), seed=0, tol=0.5)
    rows = slice(n // 2 - c // 2, n // 2 + (c + 1) // 2)

    assert mask[rows, rows].all()
    assert mask[n // 2, n // 2]


# Masks ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda seed: pp.make_random_mask((20, 16), 3.0, calib=(4, 4), seed=seed),
        lambda seed: pp.make_poisson_disc_mask((20, 16), 3.0, calib=(4, 4), seed=seed),
        lambda seed: pp.make_caipirinha_mask((20, 16), 2, 2, delta=1),
    ],
    ids=["random", "poisson", "caipirinha"],
)
def test_a_mask_is_boolean_of_the_requested_shape_and_reproducible(make):
    mask = make(4)

    assert mask.dtype == bool and mask.shape == (20, 16)
    assert np.array_equal(mask, make(4))


@pytest.mark.parametrize(
    "mask",
    [
        pp.make_random_mask((20, 16), 3.0, calib=(4, 4), seed=0),
        pp.make_poisson_disc_mask((20, 16), 3.0, calib=(4, 4), seed=0),
    ],
    ids=["random", "poisson"],
)
def test_a_seeded_mask_contains_its_calibration_block(mask):
    assert mask[8:12, 6:10].all()


# Orderings -----------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("pad", [False, True])
def test_an_ordering_returns_each_input_row_index_exactly_once(order, pad):
    trains = order(GRID, 4, pad=pad)
    indices = [i for train in trains for i in train if i is not None]

    assert sorted(indices) == list(range(len(GRID)))
    assert all(isinstance(i, int) for i in indices)


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
def test_an_ordering_returns_indices_not_coordinates(order):
    # Coordinates far outside range(N) cannot be mistaken for indices.
    far = GRID * 100 + 1000
    trains = order(far, 4)

    assert {i for train in trains for i in train} == set(range(len(far)))


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
def test_padding_uses_none_only_for_unacquired_echoes(order):
    seeded = {"seed": 0} if order is pp.make_shuffling_order else {}
    padded = order(GRID, 4, pad=True, **seeded)
    unpadded = order(GRID, 4, pad=False, **seeded)

    assert all(len(train) == 4 for train in padded)
    assert [[i for i in train if i is not None] for train in padded] == unpadded
    none = sum(train.count(None) for train in padded)
    assert none == 4 * len(padded) - len(GRID)


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
def test_an_ordering_refuses_a_boolean_mask(order):
    mask = np.zeros((6, 5), dtype=bool)
    mask[GRID[:, 0] + 3, GRID[:, 1] + 2] = True

    with pytest.raises(TypeError, match="argwhere"):
        order(mask, 4)


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
def test_an_ordering_refuses_a_bare_count(order):
    with pytest.raises(ValueError, match="shape"):
        order(8, 4)


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("train_length", [0, -1, 2.5])
def test_an_ordering_refuses_a_train_length_that_is_not_a_positive_integer(
    order, train_length
):
    with pytest.raises(ValueError, match="train_length"):
        order(GRID, train_length)


@pytest.mark.parametrize(
    "order",
    [pp.make_linear_order, pp.make_centric_order, pp.make_radial_adaptive_order],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("center_echo", [-1, 4, 7])
def test_a_center_echo_outside_the_train_is_refused_not_wrapped(order, center_echo):
    with pytest.raises(ValueError, match="center_echo"):
        order(GRID, 4, center_echo=center_echo)


@pytest.mark.parametrize("order", ORDERINGS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("empty", [[], np.zeros((0, 2)), np.zeros(0)])
def test_an_ordering_of_no_views_is_empty(order, empty):
    assert order(empty, 4) == []
    assert order(empty, 4, pad=True) == []


@pytest.mark.parametrize("center_echo", [0, 1, 2, 3])
def test_the_adaptive_order_acquires_the_centre_at_the_target_echo(center_echo):
    trains = pp.make_radial_adaptive_order(GRID, 4, center_echo=center_echo, pad=True)
    centre = int(np.flatnonzero((GRID == 0).all(axis=1))[0])

    assert any(train[center_echo] == centre for train in trains)


@pytest.mark.parametrize("center_echo", [0, 2])
def test_the_centric_order_acquires_the_centre_at_the_target_echo(center_echo):
    trains = pp.make_centric_order(GRID, 4, center_echo=center_echo, pad=True)
    centre = int(np.flatnonzero((GRID == 0).all(axis=1))[0])

    assert any(train[center_echo] == centre for train in trains)


def test_the_radial_order_starts_every_train_nearest_the_centre():
    trains = pp.make_radial_order(GRID, 4)
    radius = np.hypot(*GRID.T)

    assert all(radius[train[0]] == min(radius[i] for i in train) for train in trains)


#: Encoded supports whose centroid is not the k-space centre: an even matrix,
#: partial Fourier in both directions, and an asymmetric undersampled support.
ASYMMETRIC = {
    "even-matrix": pp.make_cartesian_plane_sampling((16, 8)),
    "partial-fourier": pp.make_cartesian_plane_sampling(
        (16, 12), partial_fourier=(0.625, 0.75)
    ),
    "asymmetric-undersampled": pp.make_cartesian_plane_sampling(
        (20, 10), (2, 1), (4, 2), partial_fourier=(0.7, 1.0)
    ),
}


def _centred(name):
    calibration, imaging = ASYMMETRIC[name]
    views = np.array(calibration + imaging)
    shape = {"even-matrix": (16, 8), "partial-fourier": (16, 12)}.get(name, (20, 10))
    return views - np.array(shape) // 2


@pytest.mark.parametrize("name", sorted(ASYMMETRIC))
def test_the_support_centroid_is_not_the_kspace_centre_in_these_cases(name):
    assert np.linalg.norm(_centred(name).mean(axis=0)) > 0.1


@pytest.mark.parametrize("name", sorted(ASYMMETRIC))
@pytest.mark.parametrize("center_echo", [0, 2, 5])
def test_the_adaptive_order_puts_the_kspace_centre_not_the_centroid_at_the_target(
    name, center_echo
):
    centred = _centred(name)
    centre = int(np.flatnonzero((centred == 0).all(axis=1))[0])
    trains = pp.make_radial_adaptive_order(
        centred, 6, center_echo=center_echo, pad=True
    )

    assert any(train[center_echo] == centre for train in trains)


@pytest.mark.parametrize("name", sorted(ASYMMETRIC))
def test_the_centric_order_starts_at_the_kspace_centre(name):
    centred = _centred(name)
    centre = int(np.flatnonzero((centred == 0).all(axis=1))[0])
    trains = pp.make_centric_order(centred, 6, pad=True)

    assert any(train[0] == centre for train in trains)


@pytest.mark.parametrize("name", sorted(ASYMMETRIC))
def test_the_radial_order_starts_every_shot_at_its_view_nearest_the_origin(name):
    centred = _centred(name)
    radius = np.hypot(*centred.T)
    trains = pp.make_radial_order(centred, 6)
    centre = int(np.flatnonzero((centred == 0).all(axis=1))[0])

    assert all(radius[t[0]] == min(radius[i] for i in t) for t in trains)
    assert any(t[0] == centre for t in trains)


def test_shuffling_is_reproducible_by_seed_and_keeps_clustered_membership():
    first = pp.make_shuffling_order(GRID, 4, seed=11)
    again = pp.make_shuffling_order(GRID, 4, seed=11)
    other = pp.make_shuffling_order(GRID, 4, seed=12)

    assert first == again
    assert first != other
    assert [sorted(t) for t in first] == [sorted(t) for t in other]


# EPI -----------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["linear", "caipi", "zigzag"])
def test_epi_offsets_start_at_the_origin(scheme):
    extent = 12 if scheme == "zigzag" else None
    offsets = pp.make_epi_shot_offsets(
        9, scheme=scheme, acceleration=2, partition_acceleration=3, extent=extent
    )

    assert offsets.shape == (9, 2)
    assert offsets[0].tolist() == [0, 0]


def test_epi_offsets_are_relative_to_the_shot_origin():
    offsets = pp.make_epi_shot_offsets(5, acceleration=2)

    for origin in ([0, 0], [3, 1], [7, 4]):
        views = offsets + origin
        assert views[0].tolist() == origin
        assert np.array_equal(views - origin, offsets)


@pytest.mark.parametrize(("acceleration", "segments"), [(1, 1), (2, 1), (2, 3), (3, 2)])
def test_segmented_shots_interleave_to_every_accelerated_line(acceleration, segments):
    offsets = pp.make_epi_shot_offsets(4, acceleration=acceleration, segments=segments)
    lines = sorted(
        int(y) for s in range(segments) for y in offsets[:, 0] + s * acceleration
    )

    assert np.all(np.diff(offsets[:, 0]) == acceleration * segments)
    assert lines == list(range(0, 4 * segments * acceleration, acceleration))


def test_caipi_offsets_tile_the_caipirinha_lattice():
    ry, rz, shift = 2, 3, 1
    offsets = pp.make_epi_shot_offsets(
        6, scheme="caipi", acceleration=ry, partition_acceleration=rz, caipi_shift=shift
    )
    mask = pp.make_caipirinha_mask((12, rz), ry, rz, delta=shift)

    assert sorted(map(tuple, offsets.tolist())) == sorted(
        map(tuple, np.argwhere(mask).tolist())
    )
