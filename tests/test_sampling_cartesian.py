"""What a Cartesian scan acquires: the imaging, the calibration block, partial Fourier.

Every shipped Cartesian sequence chooses its views with these two routines, so
what they promise is what those sequences do. The centre of k-space is the
invariant they are built around: it carries the contrast, and a reconstruction
that estimates coil sensitivities needs it whatever the undersampling.
"""

from __future__ import annotations

import numpy as np
import pytest

import pypulseqpp as pp


@pytest.mark.parametrize("n", [64, 120, 128, 127])
@pytest.mark.parametrize("r", [1, 2, 3, 4])
@pytest.mark.parametrize("n_acs", [0, 16, 25])
def test_the_centre_view_is_acquired_whatever_the_undersampling(n, r, n_acs):
    calibration, imaging = pp.make_cartesian_axis_sampling(n, r, n_acs)

    assert n // 2 in {*calibration, *imaging}


@pytest.mark.parametrize("n", [64, 127])
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("n_acs", [8, 16])
def test_the_calibration_block_and_the_lattice_do_not_overlap(n, r, n_acs):
    calibration, imaging = pp.make_cartesian_axis_sampling(n, r, n_acs)

    assert not set(calibration) & set(imaging)
    assert calibration == sorted(calibration)
    assert imaging == sorted(imaging)


@pytest.mark.parametrize("n", [64, 127])
@pytest.mark.parametrize("n_acs", [0, 16])
def test_a_fully_sampled_axis_has_no_calibration_block_and_every_view(n, n_acs):
    calibration, imaging = pp.make_cartesian_axis_sampling(n, 1, n_acs)

    assert calibration == []
    assert imaging == list(range(n))


@pytest.mark.parametrize("partial_fourier", [0.75, 0.8, 1.0])
def test_partial_fourier_drops_the_views_before_the_centre_and_no_others(
    partial_fourier,
):
    n = 64
    calibration, imaging = pp.make_cartesian_axis_sampling(
        n, 2, 8, partial_fourier=partial_fourier
    )
    acquired = {*calibration, *imaging}

    first = n - round(partial_fourier * n)
    assert min(acquired) >= first
    assert n // 2 in acquired


def test_the_lattice_steps_by_the_undersampling_factor_about_the_centre():
    _, imaging = pp.make_cartesian_axis_sampling(64, 4, 0)

    assert all((view - 32) % 4 == 0 for view in imaging)


@pytest.mark.parametrize("acceleration", [(1, 1), (2, 1), (2, 2), (3, 2)])
def test_the_centre_pair_is_acquired_whatever_the_undersampling(acceleration):
    calibration, imaging = pp.make_cartesian_plane_sampling(
        (32, 16), acceleration, (8, 4)
    )

    assert (16, 8) in {*calibration, *imaging}


def test_a_caipirinha_shift_climbs_one_partition_per_acquired_line():
    _, imaging = pp.make_cartesian_plane_sampling((8, 8), (2, 2), caipi_shift=1)

    partitions = {
        line: sorted(z for y, z in imaging if y == line) for line, _ in imaging
    }
    steps = [min(partitions[line]) for line in sorted(partitions)]
    assert steps == sorted(steps) or len(set(steps)) > 1


def test_elliptical_sampling_drops_the_corners_and_keeps_the_centre():
    _, full = pp.make_cartesian_plane_sampling((32, 32))
    _, ellipse = pp.make_cartesian_plane_sampling((32, 32), elliptical=True)

    assert (0, 0) in full and (0, 0) not in ellipse
    assert (16, 16) in ellipse
    assert len(ellipse) < len(full)


def test_a_shuffled_draw_covers_the_calibration_region_and_thins_the_rest():
    calibration, imaging = pp.make_cartesian_plane_sampling(
        (48, 48), (2, 2), (8, 8), elliptical=True, sampling="poisson", seed=7
    )

    assert len(calibration) == 64
    assert not set(calibration) & set(imaging)
    assert 0 < len(imaging) < 48 * 48


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"acceleration": 0}, "at least 1"),
        ({"n_acs": -1}, "nonnegative"),
        ({"partial_fourier": 0.4}, "partial_fourier"),
    ],
)
def test_an_out_of_range_argument_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pp.make_cartesian_axis_sampling(64, **kwargs)


@pytest.mark.parametrize("sampling", ["lattice", "poisson"])
def test_the_plane_count_is_the_number_of_views_the_plane_sampling_lists(sampling):
    arguments = {
        "shape": (48, 40),
        "acceleration": (2, 2),
        "n_acs": (12, 10),
        "caipi_shift": 1 if sampling == "lattice" else 0,
        "partial_fourier": (0.75, 1.0),
        "elliptical": True,
        "elliptical_acs": True,
        "sampling": sampling,
        "seed": 3,
    }

    calibration, imaging = pp.make_cartesian_plane_sampling(**arguments)

    assert pp.count_cartesian_plane_sampling(**arguments) == (
        len(calibration),
        len(imaging),
    )


def test_the_probability_of_a_lattice_view_is_whether_the_plane_sampling_lists_it():
    arguments = {
        "shape": (48, 40),
        "acceleration": (2, 2),
        "n_acs": (12, 10),
        "caipi_shift": 1,
        "partial_fourier": (0.75, 1.0),
        "elliptical": True,
        "elliptical_acs": True,
    }

    listed = pp.make_cartesian_plane_sampling(**arguments)
    probabilities = pp.calc_cartesian_plane_sampling_probability(**arguments)

    for views, probability in zip(listed, probabilities, strict=True):
        mask = np.zeros(arguments["shape"])
        mask[tuple(np.transpose(views))] = 1.0
        assert np.array_equal(probability, mask)


@pytest.mark.parametrize(
    ("shape", "acceleration", "n_acs"),
    [((128, 64), (2, 1), (24, 24)), ((96, 96), (2, 2), (0, 0))],
)
def test_the_poisson_disc_probability_counts_the_views_and_the_fullest_partition_of_a_mean_draw(
    shape, acceleration, n_acs
):
    arguments = {
        "acceleration": acceleration,
        "n_acs": n_acs,
        "elliptical": True,
        "sampling": "poisson",
    }
    draws = []
    for seed in range(8):
        calibration, imaging = pp.make_cartesian_plane_sampling(
            shape, **arguments, seed=seed
        )
        mask = np.zeros(shape)
        mask[tuple(np.transpose(calibration + imaging))] = 1.0
        draws.append(mask)
    mean = np.mean(draws, axis=0)

    expected = sum(pp.calc_cartesian_plane_sampling_probability(shape, **arguments))

    assert expected.sum() == pytest.approx(mean.sum(), rel=0.03)
    # Over five partitions, which averages a draw's fluctuation out.
    smoothed = np.convolve(mean.sum(axis=0), np.ones(5) / 5, mode="same")
    assert expected.sum(axis=0).max() == pytest.approx(smoothed.max(), rel=0.04)
