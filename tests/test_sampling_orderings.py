"""Echo-train coverage, zero-based centre echoes and None padding.

The orderings take centred coordinates, so the encoded grid is shifted by its
k-space centre before it is ordered.
"""

from __future__ import annotations

import numpy as np
import pytest

from pypulseqpp import (
    make_centric_order,
    make_linear_order,
    make_pclo_order,
    make_radial_adaptive_order,
    make_radial_order,
    make_shuffling_order,
)

GRID = np.column_stack(
    [g.ravel() for g in np.meshgrid(np.arange(8), np.arange(4), indexing="ij")]
)
CENTER = (4, 2)
ETL = 8


def _radius(index):
    return float(np.hypot(GRID[index, 0] - 4, GRID[index, 1] - 2))


def _centre_echo(trains):
    for train in trains:
        for echo, index in enumerate(train):
            if index is not None and tuple(GRID[index]) == CENTER:
                return echo
    raise AssertionError("centre view was never encoded")


ALL_ORDERINGS = [
    make_linear_order,
    make_centric_order,
    make_pclo_order,
    make_radial_order,
    make_radial_adaptive_order,
    make_shuffling_order,
]


@pytest.mark.parametrize("order", ALL_ORDERINGS)
def test_every_ordering_covers_its_views_once(order):
    trains = order(GRID, ETL)
    flat = [index for train in trains for index in train if index is not None]
    assert sorted(flat) == list(range(len(GRID))), order.__name__


@pytest.mark.parametrize("order", ALL_ORDERINGS)
def test_pad_gives_every_train_the_train_length(order):
    trains = order(GRID, ETL, pad=True)
    assert all(len(train) == ETL for train in trains)
    # The real views still cover the grid exactly once.
    flat = [index for train in trains for index in train if index is not None]
    assert sorted(flat) == list(range(len(GRID)))


@pytest.mark.parametrize(
    "order", [make_linear_order, make_centric_order, make_radial_adaptive_order]
)
def test_center_echo_places_the_centre_on_the_target_echo(order):
    target = 3
    trains = order(GRID - CENTER, ETL, center_echo=target, pad=True)
    assert _centre_echo(trains) == target


def test_radial_puts_the_centre_on_the_first_echo():
    trains = make_radial_order(GRID - CENTER, ETL, pad=True)
    assert _centre_echo(trains) == 0


def test_radial_adaptive_radius_grows_away_from_the_target_echo():
    target = 3
    trains = make_radial_adaptive_order(
        GRID - CENTER, ETL, center_echo=target, pad=True
    )
    by_echo: dict[int, list[float]] = {}
    for train in trains:
        for echo, index in enumerate(train):
            if index is not None:
                by_echo.setdefault(echo, []).append(_radius(index))
    mean_radius = {echo: np.mean(values) for echo, values in by_echo.items()}
    assert min(mean_radius, key=mean_radius.get) == target
    # Monotone growth on each side of the target echo.
    for echo in range(target, max(mean_radius)):
        assert mean_radius[echo + 1] >= mean_radius[echo] - 1e-9
    for echo in range(target, 0, -1):
        assert mean_radius[echo - 1] >= mean_radius[echo] - 1e-9


def test_shuffling_is_reproducible_by_seed():
    a = make_shuffling_order(GRID, ETL, seed=7, pad=True)
    b = make_shuffling_order(GRID, ETL, seed=7, pad=True)
    c = make_shuffling_order(GRID, ETL, seed=8, pad=True)
    assert a == b
    assert a != c


@pytest.mark.parametrize("length", [1, 2, 5, 6, 9, 10])
def test_pclo_ring_permutation_preserves_bands(length):
    from pypulseqpp._ordering import _pclo_rings

    # Two views per ring; strict radial bands must survive echo reassignment.
    coords = np.arange(2 * length)
    bands = make_centric_order(coords, length, pad=True)
    for centre in range(length):
        rings = _pclo_rings(centre, length)
        assert sorted(rings) == list(range(length))
        assert rings[0] == centre
        assert np.all(np.abs(np.diff(rings)) <= 2)
        trains = make_pclo_order(coords, length, center_echo=centre, pad=True)
        for ring, echo in enumerate(rings):
            assert [row[echo] for row in trains] == [row[ring] for row in bands]
    assert make_pclo_order(coords, length, center_echo=0, pad=True) == bands


def test_pclo_published_equation_three_example():
    from pypulseqpp._ordering import _pclo_rings

    np.testing.assert_array_equal(_pclo_rings(3, 9), [3, 1, 0, 2, 4, 5, 6, 7, 8])
    # A centre in the first half of a 60-echo train, as a 44 ms Busse TE puts it.
    np.testing.assert_array_equal(
        _pclo_rings(21, 60),
        [*range(21, -1, -2), *range(0, 21, 2), *range(22, 60)],
    )


def test_pclo_late_centre_is_not_mirrored():
    from pypulseqpp._ordering import _pclo_rings

    np.testing.assert_array_equal(_pclo_rings(8, 9), [8, 6, 4, 2, 0, 1, 3, 5, 7])
    np.testing.assert_array_equal(
        _pclo_rings(32, 60),
        [*range(32, -1, -2), *range(1, 32, 2), *range(33, 60)],
    )


@pytest.mark.parametrize("length", [9, 10, 45, 60, 61])
def test_pclo_echoes_after_the_centre_read_further_out_the_later_they_come(length):
    from pypulseqpp._ordering import _pclo_rings

    for centre in range(length // 2, length):
        rings = _pclo_rings(centre, length)
        assert sorted(rings) == list(range(length))
        assert rings[0] == centre
        assert np.all(np.abs(np.diff(rings)) <= 2)
        # The echo index rises with radius outside the rings the earlier
        # echoes fill, and the last echo reads the outermost ring.
        later = np.flatnonzero(rings > centre + 1)
        assert np.all(np.diff(rings[later]) == 1)
        assert np.all(later > np.flatnonzero(rings <= centre).max())
        # A train flat up to the centre and decaying after it, as a Busse
        # schedule is, weights k-space without rising from centre to edge.
        echo = np.arange(length)
        signal = np.where(echo <= centre, 1.0, np.exp(-(echo - centre) / 10.0))
        assert np.all(np.diff(signal[rings]) <= 1e-12)
