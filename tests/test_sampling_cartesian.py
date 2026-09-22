"""What a Cartesian scan acquires: the lattice, the calibration block, partial Fourier.

Every shipped Cartesian sequence chooses its views with these two routines, so
what they promise is what those sequences do. The centre of k-space is the
invariant they are built around: it carries the contrast, and a reconstruction
that estimates coil sensitivities needs it whatever the undersampling.
"""

from __future__ import annotations

import pytest

import pypulseqpp as pp


@pytest.mark.parametrize("n", [64, 120, 128, 127])
@pytest.mark.parametrize("r", [1, 2, 3, 4])
@pytest.mark.parametrize("n_acs", [0, 16, 25])
def test_the_centre_view_is_acquired_whatever_the_undersampling(n, r, n_acs):
    calibration, lattice = pp.calc_sampled_lines(n, r, n_acs)

    assert n // 2 in {*calibration, *lattice}


@pytest.mark.parametrize("n", [64, 127])
@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("n_acs", [8, 16])
def test_the_calibration_block_and_the_lattice_do_not_overlap(n, r, n_acs):
    calibration, lattice = pp.calc_sampled_lines(n, r, n_acs)

    assert not set(calibration) & set(lattice)
    assert calibration == sorted(calibration)
    assert lattice == sorted(lattice)


@pytest.mark.parametrize("n", [64, 127])
@pytest.mark.parametrize("n_acs", [0, 16])
def test_a_fully_sampled_axis_has_no_calibration_block_and_every_view(n, n_acs):
    calibration, lattice = pp.calc_sampled_lines(n, 1, n_acs)

    assert calibration == []
    assert lattice == list(range(n))


@pytest.mark.parametrize("partial_fourier", [0.75, 0.8, 1.0])
def test_partial_fourier_drops_the_views_before_the_centre_and_no_others(
    partial_fourier,
):
    n = 64
    calibration, lattice = pp.calc_sampled_lines(
        n, 2, 8, partial_fourier=partial_fourier
    )
    acquired = {*calibration, *lattice}

    first = n - round(partial_fourier * n)
    assert min(acquired) >= first
    assert n // 2 in acquired


def test_the_lattice_steps_by_the_undersampling_factor_about_the_centre():
    _, lattice = pp.calc_sampled_lines(64, 4, 0)

    assert all((view - 32) % 4 == 0 for view in lattice)


@pytest.mark.parametrize("acceleration", [(1, 1), (2, 1), (2, 2), (3, 2)])
def test_the_centre_pair_is_acquired_whatever_the_undersampling(acceleration):
    calibration, lattice = pp.calc_sampled_pairs((32, 16), acceleration, (8, 4))

    assert (16, 8) in {*calibration, *lattice}


def test_a_caipirinha_shift_climbs_one_partition_per_acquired_line():
    _, lattice = pp.calc_sampled_pairs((8, 8), (2, 2), caipi_shift=1)

    partitions = {
        line: sorted(z for y, z in lattice if y == line) for line, _ in lattice
    }
    steps = [min(partitions[line]) for line in sorted(partitions)]
    assert steps == sorted(steps) or len(set(steps)) > 1


def test_elliptical_sampling_drops_the_corners_and_keeps_the_centre():
    _, full = pp.calc_sampled_pairs((32, 32))
    _, ellipse = pp.calc_sampled_pairs((32, 32), elliptical=True)

    assert (0, 0) in full and (0, 0) not in ellipse
    assert (16, 16) in ellipse
    assert len(ellipse) < len(full)


def test_a_shuffled_draw_covers_the_calibration_region_and_thins_the_rest():
    calibration, lattice = pp.calc_sampled_pairs(
        (48, 48), (2, 2), (8, 8), elliptical=True, shuffling=True, seed=7
    )

    assert len(calibration) == 64
    assert not set(calibration) & set(lattice)
    assert 0 < len(lattice) < 48 * 48


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"r": 0}, "at least 1"),
        ({"n_acs": -1}, "nonnegative"),
        ({"partial_fourier": 0.4}, "partial_fourier"),
    ],
)
def test_an_out_of_range_argument_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pp.calc_sampled_lines(64, **kwargs)
