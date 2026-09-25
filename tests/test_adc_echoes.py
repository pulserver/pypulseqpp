"""Per readout: the axes k-space moves along and the samples nearest its centre."""

import math

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def plain_echoes(seq):
    """The rule, written out over ``adc_kspace`` for every readout at once."""
    k = seq.adc_kspace()
    rows = [block for block in seq.block_events if seq.block_events[block][5]]
    moving, echo, start = [], [], 0
    for block in rows:
        n = int(seq.get_block(block).adc.num_samples)
        kk = k[:, start : start + n]
        start += n
        span = np.ptp(kk, axis=1) if n else np.zeros(3)
        moves = span > 1e-6 * span.max()
        moving.append(moves)
        if n < 2 or not moves.any():
            echo.append((-1, -1))
            continue
        swept = kk * moves[:, None]
        distance = np.sqrt((swept**2).sum(axis=0))
        nearest = int(distance.argmin())

        def step(at, swept=swept):
            return np.sqrt(((swept[:, at + 1] - swept[:, at]) ** 2).sum())

        beside = max(step(min(max(nearest - 1, 0), n - 2)), step(min(nearest, n - 2)))
        tied = np.flatnonzero(distance <= distance[nearest] + 1e-2 * beside)
        echo.append((int(tied[0]), int(tied[-1])))
    return np.array(rows), np.array(moving, dtype=bool).reshape(-1, 3), np.array(echo)


def gradient_echo(system, *, samples=64, rotation=None, prewind=True):
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid(
        "x", flat_area=samples / 0.2, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(samples, duration=3.2e-3, delay=readout.rise_time, system=system)
    turn = () if rotation is None else (pp.make_rotation(rotation),)
    seq.add_block(pp.make_block_pulse(math.pi / 12, duration=1e-3, use="excitation"))
    if prewind:
        seq.add_block(
            pp.make_trapezoid(
                "x", area=-readout.area / 2, duration=1e-3, system=system
            ),
            *turn,
        )
    seq.add_block(readout, adc, *turn)
    return seq


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def test_a_cartesian_readout_moves_along_its_readout_axis_alone(system):
    echoes = gradient_echo(system).adc_echoes()

    assert echoes.block.tolist() == [3]
    assert echoes.moving.tolist() == [[True, False, False]]


def test_an_even_readout_through_the_centre_ties_its_two_middle_samples(system):
    assert gradient_echo(system, samples=64).adc_echoes().echo.tolist() == [[31, 32]]


def test_an_odd_readout_through_the_centre_has_one_nearest_sample(system):
    assert gradient_echo(system, samples=65).adc_echoes().echo.tolist() == [[32, 32]]


def test_a_readout_moves_along_the_axis_its_block_rotation_turns_it_onto(system):
    """The axes are those after each block's rotation: the logical frame."""
    quarter = Rotation.from_euler("z", 90, degrees=True)

    echoes = gradient_echo(system, rotation=quarter).adc_echoes()

    assert echoes.moving.tolist() == [[False, True, False]]
    assert echoes.echo.tolist() == [[31, 32]]


def test_a_readout_that_starts_at_the_centre_echoes_at_its_first_sample(system):
    assert gradient_echo(system, prewind=False).adc_echoes().echo.tolist() == [[0, 0]]


def test_a_readout_that_does_not_move_has_no_echo(system):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"))
    seq.add_block(pp.make_adc(128, duration=12.8e-3, system=system))

    echoes = seq.adc_echoes()

    assert echoes.moving.tolist() == [[False, False, False]]
    assert echoes.echo.tolist() == [[-1, -1]]


def test_a_readout_of_one_sample_has_no_echo(system):
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid("x", area=1000, duration=2e-3, system=system)
    seq.add_block(readout, pp.make_adc(1, duration=1e-5, delay=1e-3, system=system))

    assert seq.adc_echoes().echo.tolist() == [[-1, -1]]


def test_each_readout_starts_at_its_column_of_adc_kspace(system):
    seq = sequences.gre_radial2D_sequence(
        fov=220e-3, n=64, n_slices=1, tr=None, n_dummy=0
    )

    echoes = seq.adc_echoes()

    assert (
        echoes.first_sample.tolist()
        == np.concatenate(([0], np.cumsum(echoes.num_samples)[:-1])).tolist()
    )
    assert int(echoes.num_samples.sum()) == seq.adc_kspace().shape[1]


@pytest.mark.parametrize(
    "build",
    [
        lambda: sequences.gre_radial2D_sequence(
            fov=220e-3, n=64, n_slices=1, tr=None, n_dummy=0
        ),
        lambda: sequences.epi2D_sequence(n_slices=1),
        lambda: sequences.gre_spiral2D_sequence(),
    ],
    ids=["radial", "epi", "spiral"],
)
def test_the_echoes_are_the_rule_written_out_over_the_whole_trajectory(build):
    seq = build()

    echoes = seq.adc_echoes()

    blocks, moving, echo = plain_echoes(seq)
    np.testing.assert_array_equal(echoes.block, blocks)
    np.testing.assert_array_equal(echoes.moving, moving)
    np.testing.assert_array_equal(echoes.echo, echo)


def test_k_space_is_followed_a_range_at_a_time_without_changing_the_answer(system):
    """More than 2^17 samples, so later readouts are integrated from a later excitation."""
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid(
        "x", flat_area=512 / 0.2, flat_time=5.12e-3, system=system
    )
    adc = pp.make_adc(512, duration=5.12e-3, delay=readout.rise_time, system=system)
    pulse = pp.make_block_pulse(math.pi / 12, duration=1e-3, use="excitation")
    for line in range(300):
        seq.add_block(pulse)
        seq.add_block(
            pp.make_trapezoid(
                "x", area=-readout.area / 2, duration=2e-3, system=system
            ),
            pp.make_trapezoid(
                "y", area=(line - 150) * 5.0, duration=2e-3, system=system
            ),
        )
        seq.add_block(readout, adc)

    echoes = seq.adc_echoes()

    assert int(echoes.num_samples.sum()) > 1 << 17
    _, moving, echo = plain_echoes(seq)
    np.testing.assert_array_equal(echoes.moving, moving)
    np.testing.assert_array_equal(echoes.echo, echo)


def test_a_sequence_without_readouts_has_no_entries():
    echoes = pp.Sequence(pp.Opts()).adc_echoes()

    assert echoes.block.shape == (0,)
    assert echoes.moving.shape == (0, 3)
    assert echoes.echo.shape == (0, 2)
