"""The contracts of the sampling API: support, masks, orderings, EPI offsets, renames.

Support routines return encoded coordinates or boolean masks and nothing else;
ordering routines return indices into the coordinates they are given; the EPI
routine returns offsets relative to a shot origin the caller chooses. These
tests state those contracts rather than particular outputs.
"""

from __future__ import annotations

import warnings

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
    mask = pp.make_poisson_disc_mask(shape, 4.0, calib=(6, 6), seed=seed)

    assert sorted(calibration + imaging) == sorted(
        map(tuple, np.argwhere(mask).tolist())
    )


def test_an_unknown_support_scheme_is_refused():
    with pytest.raises(ValueError, match="sampling"):
        pp.make_cartesian_plane_sampling((8, 8), (2, 2), sampling="shuffling")


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
def test_an_ordering_accepts_a_boolean_mask(order):
    mask = np.zeros((6, 5), dtype=bool)
    mask[GRID[:, 0] + 3, GRID[:, 1] + 2] = True
    trains = order(mask, 4)

    assert sorted(i for train in trains for i in train) == list(range(len(GRID)))


def test_a_one_dimensional_boolean_mask_is_a_set_of_lines():
    trains = pp.make_linear_order(np.array([True, False, True, True]), 2)

    assert sorted(i for train in trains for i in train) == [0, 1, 2]


@pytest.mark.parametrize("center_echo", [0, 1, 2, 3])
def test_the_adaptive_order_acquires_the_centre_at_the_target_echo(center_echo):
    trains = pp.make_radial_adaptive_order(
        GRID, 4, center=(0, 0), center_echo=center_echo, pad=True
    )
    centre = int(np.flatnonzero((GRID == 0).all(axis=1))[0])

    assert any(train[center_echo] == centre for train in trains)


@pytest.mark.parametrize("center_echo", [0, 2])
def test_the_centric_order_acquires_the_centre_at_the_target_echo(center_echo):
    trains = pp.make_centric_order(
        GRID, 4, center=(0, 0), center_echo=center_echo, pad=True
    )
    centre = int(np.flatnonzero((GRID == 0).all(axis=1))[0])

    assert any(train[center_echo] == centre for train in trains)


def test_the_radial_order_starts_every_train_nearest_the_centre():
    trains = pp.make_radial_order(GRID, 4, center=(0, 0))
    radius = np.hypot(*GRID.T)

    assert all(radius[train[0]] == min(radius[i] for i in train) for train in trains)


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


# Renamed names ---------------------------------------------------------------

RENAMED = {
    "calc_sampled_lines": ("make_cartesian_axis_sampling", (16, 2, 4), {}),
    "calc_sampled_pairs": (
        "make_cartesian_plane_sampling",
        ((8, 8), (2, 2), (2, 2)),
        {},
    ),
    "calc_traversal_order": ("make_traversal_order", (7, "center_out"), {}),
    "calc_epi_order": ("make_epi_shot_offsets", (5,), {"acceleration": 2}),
}


@pytest.mark.parametrize("old", sorted(RENAMED))
def test_a_renamed_routine_still_resolves_warns_and_forwards(old):
    new, args, kwargs = RENAMED[old]

    with pytest.warns(DeprecationWarning, match=new):
        result = getattr(pp, old)(*args, **kwargs)

    assert repr(result) == repr(getattr(pp, new)(*args, **kwargs))
    assert getattr(pp, old).__wrapped__ is getattr(pp, new)


@pytest.mark.parametrize("old", sorted(RENAMED))
def test_a_renamed_routine_is_importable_but_not_advertised(old):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # What ``from pypulseqpp import <old>`` does; resolving warns nothing.
        imported = getattr(__import__("pypulseqpp", fromlist=[old]), old)

    assert callable(imported)
    assert old not in pp.__all__
    assert RENAMED[old][0] in pp.__all__


def test_the_former_shuffling_flag_selects_poisson_support():
    kwargs = {"shape": (24, 24), "acceleration": (2, 2), "n_acs": (6, 6), "seed": 2}
    with pytest.warns(DeprecationWarning):
        former = pp.calc_sampled_pairs(shuffling=True, **kwargs)

    assert former == pp.make_cartesian_plane_sampling(sampling="poisson", **kwargs)


def test_the_former_r_argument_is_the_acceleration():
    with pytest.warns(DeprecationWarning):
        former = pp.calc_sampled_lines(16, r=2, n_acs=4)

    assert former == pp.make_cartesian_axis_sampling(16, acceleration=2, n_acs=4)
