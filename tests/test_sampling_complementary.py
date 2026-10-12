"""Complementary Poisson-disc echo assignment over a fixed support.

The reference behaviour is Levine et al., Magn Reson Med 2017;77:1774-1785:
every echo is a Poisson-disc pattern of its own, and the echoes together are
the support.
"""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp

SHAPE = (96, 72)
ETL = 24


@pytest.fixture(scope="module")
def views():
    mask = pp.make_poisson_disc_mask(SHAPE, 5.0, calib=(12, 12), seed=3)
    return np.argwhere(mask) - np.array(SHAPE) // 2


def _echoes(trains, n):
    echo = np.full(n, -1)
    for train in trains:
        for e, index in enumerate(train):
            if index is not None:
                echo[index] = e
    return echo


def _closest_in_echo(views, echo):
    """Mean over echoes of the smallest distance between two views of one echo."""
    closest = []
    for e in np.unique(echo):
        p = views[echo == e].astype(float)
        d = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        closest.append(d.min())
    return float(np.mean(closest))


def test_every_echo_holds_one_view_per_shot_and_the_echoes_are_the_support(views):
    trains = pp.make_complementary_order(views, ETL, seed=0, pad=True)
    n_shots = -(-len(views) // ETL)
    echo = _echoes(trains, len(views))

    assert len(trains) == n_shots
    assert sorted(i for t in trains for i in t if i is not None) == list(
        range(len(views))
    )
    counts = np.bincount(echo, minlength=ETL)
    short = ETL * n_shots - len(views)
    assert counts.tolist() == [n_shots] * (ETL - short) + [n_shots - 1] * short


def test_equal_seeds_give_equal_orders_and_other_seeds_other_orders(views):
    first = pp.make_complementary_order(views, ETL, seed=5)

    assert pp.make_complementary_order(views, ETL, seed=5) == first
    assert pp.make_complementary_order(views, ETL, seed=6) != first


def test_views_of_one_echo_lie_further_apart_than_with_random_echo_positions(views):
    complementary = _echoes(pp.make_complementary_order(views, ETL, seed=0), len(views))
    shuffled = _echoes(pp.make_shuffling_order(views, ETL, seed=0), len(views))

    assert _closest_in_echo(views, complementary) > 2 * _closest_in_echo(
        views, shuffled
    )


def test_neighbouring_echoes_complement_each_other_within_the_temporal_width(views):
    # Views of adjacent echoes are closer under no temporal exclusion than
    # under a wide one.
    def adjacent(width):
        echo = _echoes(
            pp.make_complementary_order(views, ETL, seed=0, temporal_width=width),
            len(views),
        )
        pairs = [
            _closest_in_echo(
                views[(echo == e) | (echo == e + 1)],
                np.zeros(np.count_nonzero((echo == e) | (echo == e + 1)), dtype=int),
            )
            for e in range(ETL - 1)
        ]
        return np.mean(pairs)

    assert adjacent(4.0) > adjacent(0.0)


def test_a_complementary_order_refuses_fractional_or_repeated_views(views):
    with pytest.raises(ValueError, match="integer"):
        pp.make_complementary_order(views + 0.5, ETL)
    with pytest.raises(ValueError, match="repeat"):
        pp.make_complementary_order(np.vstack([views, views[:1]]), ETL)
