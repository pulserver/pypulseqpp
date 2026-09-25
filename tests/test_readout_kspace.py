"""The k-space of a run of readouts: where following the whole sequence puts each sample."""

import math

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def columns(seq, first, stop, **options):
    """The columns of the whole sequence's ``adc_kspace`` the readouts hold."""
    echoes = seq.adc_echoes()
    starts = np.append(echoes.first_sample, echoes.num_samples.sum())
    return seq.adc_kspace(**options)[:, starts[first] : starts[stop]]


def assert_placed(seq, first, stop, **options):
    expected = columns(seq, first, stop, **options)
    found = seq.adc_kspace(readouts=(first, stop), **options)
    assert found.shape == expected.shape
    np.testing.assert_allclose(
        found, expected, rtol=0, atol=1e-9 * (1 + np.abs(expected).max(initial=0))
    )


def readout(system, samples=64):
    gx = pp.make_trapezoid(
        "x", flat_area=samples / 0.2, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(samples, duration=3.2e-3, delay=gx.rise_time, system=system)
    return gx, adc


def prewinder(system, gx, duration=1e-3):
    return pp.make_trapezoid("x", area=-gx.area / 2, duration=duration, system=system)


@pytest.mark.parametrize(
    "build",
    [
        lambda: sequences.gre_radial2D_sequence(
            fov=220e-3, n=64, n_slices=1, tr=None, n_dummy=0
        ),
        lambda: sequences.epi2D_sequence(n_slices=1),
        lambda: sequences.gre_spiral2D_sequence(),
        lambda: sequences.se2D_sequence(n_slices=1),
    ],
    ids=["radial", "epi", "spiral", "spin-echo"],
)
def test_each_readout_is_where_the_whole_sequence_puts_it(build):
    seq = build()
    whole = seq.adc_kspace()
    echoes = seq.adc_echoes()

    for index, (start, n) in enumerate(
        zip(echoes.first_sample, echoes.num_samples, strict=True)
    ):
        found = seq.adc_kspace(readouts=(index, index + 1))
        expected = whole[:, start : start + n]
        np.testing.assert_allclose(
            found, expected, rtol=0, atol=1e-9 * (1 + np.abs(whole).max())
        )


def test_a_run_of_readouts_is_the_columns_they_hold_in_the_whole_trajectory():
    seq = sequences.epi2D_sequence(n_slices=2)
    count = seq.adc_echoes().block.size

    for first, stop in ((0, count), (1, count // 2), (count // 3, count - 1)):
        assert_placed(seq, first, stop)


def test_a_readout_integrated_after_the_echo_search_splits_the_sequence_is_placed(
    system,
):
    """More than 2^17 samples, so the echo search integrates later ones from a later excitation."""
    seq = pp.Sequence(system)
    gx, adc = readout(system, samples=512)
    pulse = pp.make_block_pulse(math.pi / 12, duration=1e-3, use="excitation")
    for line in range(300):
        seq.add_block(pulse)
        seq.add_block(
            prewinder(system, gx, duration=2e-3),
            pp.make_trapezoid(
                "y", area=(line - 150) * 5.0, duration=2e-3, system=system
            ),
        )
        seq.add_block(gx, adc)
    assert int(seq.adc_echoes().num_samples.sum()) > 1 << 17

    for first, stop in ((0, 1), (255, 257), (299, 300), (100, 300)):
        assert_placed(seq, first, stop)


@pytest.mark.parametrize("use", ["saturation", "inversion", "refocusing"])
def test_a_pulse_that_does_not_reset_k_space_does_not_start_the_integration(
    system, use
):
    """The prewinder before the pulse still moves the readout that follows it."""
    seq = pp.Sequence(system)
    gx, adc = readout(system)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"))
    seq.add_block(prewinder(system, gx))
    seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3, use=use))
    seq.add_block(gx, adc)

    k = seq.adc_kspace(readouts=(0, 1))

    assert_placed(seq, 0, 1)
    # The prewinder's -area/2, inverted by a refocusing pulse; the readout
    # has swept only its ramp by its first sample.
    prewound = gx.area / 2 if use == "refocusing" else -gx.area / 2
    assert abs(k[0, 0] - prewound) < 0.1 * gx.area


def test_a_pulse_in_a_block_that_acquires_does_not_start_the_integration(system):
    """The samples the block takes before the pulse's centre carry the k-space accrued before it."""
    seq = pp.Sequence(system)
    gx, _ = readout(system)
    pulse = pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation")
    seq.add_block(pulse)
    seq.add_block(prewinder(system, gx))
    seq.add_block(pulse, pp.make_adc(32, duration=1e-3, system=system))

    k = seq.adc_kspace(readouts=(0, 1))

    assert_placed(seq, 0, 1)
    assert k[0, 0] == pytest.approx(-gx.area / 2)


def test_the_readouts_around_a_block_holding_a_pulse_and_a_readout_are_placed(system):
    seq = pp.Sequence(system)
    gx, adc = readout(system)
    pulse = pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation")
    fid = pp.make_adc(32, duration=1e-3, delay=pp.calc_duration(pulse), system=system)
    seq.add_block(pulse)
    seq.add_block(gx, adc)
    seq.add_block(pulse, fid)
    seq.add_block(prewinder(system, gx))
    seq.add_block(gx, adc)

    for first, stop in ((0, 3), (1, 2), (2, 3), (1, 3)):
        assert_placed(seq, first, stop)


def test_a_readout_in_its_excitations_block_is_placed(system):
    seq = pp.Sequence(system)
    pulse = pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation")
    gx = pp.make_trapezoid("x", area=1000, duration=4e-3, system=system)
    fid = pp.make_adc(32, duration=1e-3, delay=2e-3, system=system)
    for _ in range(3):
        seq.add_block(pulse, gx, fid)

    for first, stop in ((0, 1), (1, 2), (2, 3), (0, 3)):
        assert_placed(seq, first, stop)


def test_a_readout_follows_its_blocks_rotation(system):
    seq = pp.Sequence(system)
    gx, adc = readout(system)
    turn = pp.make_rotation(Rotation.from_euler("z", 90, degrees=True))
    for _ in range(2):
        seq.add_block(
            pp.make_block_pulse(math.pi / 12, duration=1e-3, use="excitation")
        )
        seq.add_block(prewinder(system, gx), turn)
        seq.add_block(gx, adc, turn)

    k = seq.adc_kspace(readouts=(1, 2))

    assert_placed(seq, 1, 2)
    assert np.ptp(k[0]) < 1e-6 * np.ptp(k[1])


def test_a_run_takes_the_trajectory_delay_and_background_gradient_it_is_given():
    seq = sequences.gre_radial2D_sequence(
        fov=220e-3, n=64, n_slices=1, tr=None, n_dummy=0
    )
    options = {
        "trajectory_delay": (2e-6, -1e-6, 0.0),
        "gradient_offset": (5.0, 0.0, -3.0),
    }

    assert_placed(seq, 10, 12, **options)
    assert not np.allclose(
        seq.adc_kspace(readouts=(10, 12), **options), seq.adc_kspace(readouts=(10, 12))
    )


def test_an_empty_run_has_no_samples(system):
    seq = pp.Sequence(system)
    gx, adc = readout(system)
    seq.add_block(gx, adc)

    assert seq.adc_kspace(readouts=(1, 1)).shape == (3, 0)
    assert seq.adc_kspace(readouts=(0, 0)).shape == (3, 0)


@pytest.mark.parametrize("readouts", [(2, 1), (-1, 1), (0, 3)])
def test_a_run_outside_the_readouts_is_refused(system, readouts):
    seq = pp.Sequence(system)
    gx, adc = readout(system)
    seq.add_block(gx, adc)
    seq.add_block(gx, adc)

    with pytest.raises(ValueError, match="0 <= first <= stop <= 2"):
        seq.adc_kspace(readouts=readouts)


def test_a_run_of_readouts_is_not_combined_with_a_block_range(system):
    seq = pp.Sequence(system)
    gx, adc = readout(system)
    seq.add_block(gx, adc)

    with pytest.raises(ValueError, match="either block_range or readouts"):
        seq.adc_kspace(block_range=(1, 1), readouts=(0, 1))
    with pytest.raises(ValueError, match="exactly two numbers"):
        seq.adc_kspace(readouts=(0,))
