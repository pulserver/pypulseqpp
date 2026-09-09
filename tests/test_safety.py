"""What the gradients ask of the amplifiers.

Two limits bound every gradient a scanner will play: how strong it may be and
how fast it may change. Both are read off the libraries rather than off an
expanded waveform -- a gradient is a normalised shape and one amplitude, so
the steepest step belongs to the shape and what an instance asks for is that
times its own amplitude.

Which means the answer has to be checked against the waveform it stands for,
and that is what most of these tests do: expand the sequence and take the
peaks the long way round.
"""

import math

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import safety


def played_per_axis(sequence):
    """The strongest each axis reaches, taken from the expanded waveforms."""
    return [
        float(np.abs(channel[1]).max()) if channel.shape[1] else 0.0
        for channel in sequence.waveforms()
    ]


def peaks_the_long_way(sequence):
    """The strongest gradient and slew, taken from the expanded waveforms."""
    strongest = 0.0
    steepest = 0.0
    for channel in sequence.waveforms():
        if channel.shape[1] == 0:
            continue
        strongest = max(strongest, float(np.abs(channel[1]).max()))
        if channel.shape[1] > 1:
            steepest = max(
                steepest,
                float(np.abs(np.diff(channel[1]) / np.diff(channel[0])).max()),
            )
    return strongest, steepest


@pytest.fixture
def system():
    """A scanner, with the gradient raster spelled out.

    A step between two blocks is judged as that step over the raster it is
    taken in, so a test about continuity that leaves the raster to whatever
    `Opts` defaults to is partly a test about the defaults.
    """
    return pp.Opts(
        max_grad=30,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
    )


# -- read off the libraries, checked against the waveforms ------------------


def test_the_strongest_gradient_is_the_strongest_in_the_waveform(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3, system=system))
    sequence.add_block(
        pp.make_trapezoid(
            "y", amplitude=0.9 * system.max_grad, flat_time=1e-3, system=system
        )
    )
    sequence.add_block(pp.make_trapezoid("z", area=200, duration=2e-3, system=system))

    _, report = safety.check_max_grad(sequence)

    strongest, _ = peaks_the_long_way(sequence)
    assert report.per_axis.value == pytest.approx(strongest, rel=1e-9)


def test_the_steepest_slew_is_the_steepest_in_the_waveform(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3, system=system))
    sequence.add_block(
        pp.make_arbitrary_grad(
            "y",
            waveform=np.sin(np.linspace(0, np.pi, 40)) * 1e5,
            first=0.0,
            last=0.0,
            system=system,
        )
    )

    _, report = safety.check_max_slew(sequence)

    _, steepest = peaks_the_long_way(sequence)
    assert report.per_axis.value == pytest.approx(steepest, rel=1e-6)


def test_an_extended_trapezoid_slews_over_its_own_times(system):
    """Its samples sit where its time shape says, not on the raster.

    Dividing by the raster instead would say a ramp lasting two milliseconds
    happens in ten microseconds.
    """
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_extended_trapezoid(
            "x",
            amplitudes=np.array([0, 1e5, 1e5, 0]),
            times=np.array([0, 1e-3, 4e-3, 5e-3]),
            system=system,
        )
    )

    _, report = safety.check_max_slew(sequence)

    _, steepest = peaks_the_long_way(sequence)
    assert report.per_axis.value == pytest.approx(steepest, rel=1e-9)
    assert report.per_axis.value == pytest.approx(1e5 / 1e-3, rel=1e-9)


def test_the_peaks_do_not_depend_on_how_often_a_shape_is_played(system):
    """The shape is weighed once; playing it again is a multiply."""
    once = pp.Sequence(system)
    often = pp.Sequence(system)
    readout = pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
    once.add_block(readout)
    for _ in range(500):
        often.add_block(readout)

    _, first = safety.check_max_grad(once)
    _, again = safety.check_max_grad(often)

    assert first.per_axis.value == pytest.approx(again.per_axis.value)


def test_a_scaled_gradient_asks_for_less(system):
    """Amplitude is what an instance carries, and the check reads it."""
    sequence = pp.Sequence(system)
    encode = pp.make_trapezoid("y", area=1000, duration=1e-3, system=system)
    sequence.add_block(pp.scale_grad(encode, 0.25))

    _, grad = safety.check_max_grad(sequence)
    _, slew = safety.check_max_slew(sequence)

    assert grad.per_axis.value == pytest.approx(0.25 * abs(encode.amplitude))
    assert slew.per_axis.value == pytest.approx(
        0.25 * abs(encode.amplitude) / encode.rise_time
    )


# -- the verdicts -----------------------------------------------------------


def test_a_sequence_within_the_limits_passes(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3, system=system))

    assert safety.check_max_grad(sequence)[0]
    assert safety.check_max_slew(sequence)[0]


def test_a_gradient_stronger_than_the_scanner_allows_is_refused(system):
    sequence = pp.Sequence(system)
    strong = pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
    strong.amplitude = 2 * system.max_grad
    sequence.add_block(strong)

    is_ok, report = safety.check_max_grad(sequence)

    assert not is_ok
    assert report.per_axis.axis == "x"
    assert report.per_axis.block == 1
    assert report.per_axis.value > report.limit


def test_a_ramp_steeper_than_the_scanner_allows_is_refused(system):
    sequence = pp.Sequence(system)
    steep = pp.make_trapezoid("z", area=1000, duration=1e-3, system=system)
    steep.rise_time = steep.rise_time / 10
    sequence.add_block(steep)

    is_ok, report = safety.check_max_slew(sequence)

    assert not is_ok
    assert report.per_axis.axis == "z"
    assert report.per_axis.value > report.limit


def test_a_sequence_can_be_weighed_against_another_scanner(system):
    """Asking whether a sequence will run somewhere else, without rebuilding it."""
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_trapezoid(
            "x", amplitude=0.9 * system.max_grad, flat_time=1e-3, system=system
        )
    )

    assert safety.check_max_grad(sequence)[0]

    weaker = pp.Opts(max_grad=10, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    is_ok, report = safety.check_max_grad(sequence, system=weaker)

    assert not is_ok
    assert report.limit == pytest.approx(weaker.max_grad)


def test_weighing_needs_limits_from_somewhere():
    with pytest.raises(ValueError, match="no limits"):
        safety.check_max_grad(pp.Sequence())


# -- continuity -------------------------------------------------------------


def test_a_gradient_that_continues_the_last_one_is_not_a_jump(system):
    """Two halves of one waveform, split across blocks, are still one ramp."""
    sequence = pp.Sequence(system)
    rising = np.linspace(0, 1e5, 20)
    sequence.add_block(
        pp.make_arbitrary_grad("x", waveform=rising, first=0.0, last=1e5, system=system)
    )
    sequence.add_block(
        pp.make_arbitrary_grad(
            "x", waveform=rising[::-1], first=1e5, last=0.0, system=system
        )
    )

    is_ok, report = safety.check_grad_continuity(sequence)

    assert report.discontinuities == []
    assert report.ends_at_zero
    assert is_ok


def test_a_gradient_that_starts_where_the_last_did_not_end_is_a_jump(system):
    sequence = pp.Sequence(system)
    rising = pp.make_arbitrary_grad(
        "x", waveform=np.linspace(0, 1e5, 200), first=0.0, last=1e5, system=system
    )
    # Two rising ramps back to back: the second starts at zero, where the
    # first left off at full amplitude.
    sequence.add_block(rising)
    sequence.add_block(rising)

    is_ok, report = safety.check_grad_continuity(sequence)

    assert not is_ok
    assert len(report.discontinuities) == 1
    jump = report.discontinuities[0]
    assert jump.block == 2
    assert jump.axis == 0
    assert jump.before == pytest.approx(1e5)
    assert jump.after == pytest.approx(0.0)
    assert jump.slew > jump.limit


def test_a_sequence_that_leaves_a_gradient_on_has_not_ramped_down(system):
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_arbitrary_grad(
            "y", waveform=np.linspace(0, 1e5, 20), first=0.0, last=1e5, system=system
        )
    )

    is_ok, report = safety.check_grad_continuity(sequence)

    assert not report.ends_at_zero
    assert not is_ok


def test_trapezoids_begin_and_end_at_zero_so_never_jump(system):
    sequence = pp.Sequence(system)
    for _ in range(5):
        sequence.add_block(
            pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
        )

    _, report = safety.check_grad_continuity(sequence)

    assert report.discontinuities == []
    assert report.ends_at_zero


def test_a_block_playing_nothing_drops_the_axis_to_zero(system):
    """A block with no gradient on an axis is that axis off, not held.

    So a waveform left at full amplitude is a jump into the next block, even
    when that block plays nothing at all -- which is what a sequence forgetting
    to ramp down looks like from the inside.
    """
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_arbitrary_grad(
            "x", waveform=np.linspace(0, 1e5, 20), first=0.0, last=1e5, system=system
        )
    )
    sequence.add_block(pp.make_delay(1e-3))

    _, report = safety.check_grad_continuity(sequence)

    assert [(j.block, j.axis) for j in report.discontinuities] == [(2, 0)]
    # And by the end the axis really is off, so the ramp-down check passes.
    assert report.ends_at_zero


# -- what the vector peak is for -------------------------------------------


def test_the_vector_peak_is_what_a_rotation_could_put_on_one_axis(system):
    """Three axes at once ask for more than any of them alone."""
    sequence = pp.Sequence(system)
    amplitude = 0.5 * system.max_grad
    sequence.add_block(
        pp.make_trapezoid("x", amplitude=amplitude, flat_time=1e-3, system=system),
        pp.make_trapezoid("y", amplitude=amplitude, flat_time=1e-3, system=system),
        pp.make_trapezoid("z", amplitude=amplitude, flat_time=1e-3, system=system),
    )

    _, report = safety.check_max_grad(sequence)

    assert report.per_axis.value == pytest.approx(amplitude)
    assert report.vector.value == pytest.approx(math.sqrt(3) * amplitude)
    assert report.vector.axis is None


# -- each amplifier on its own ---------------------------------------------


def test_each_axis_reports_the_peak_it_is_asked_for(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=600, duration=1e-3, system=system))
    sequence.add_block(pp.make_trapezoid("y", area=1000, duration=1e-3, system=system))
    sequence.add_block(pp.make_trapezoid("z", area=300, duration=1e-3, system=system))

    _, report = safety.check_max_grad(sequence)

    assert [peak.value for peak in report.axes] == pytest.approx(
        played_per_axis(sequence), rel=1e-9
    )
    assert [peak.axis for peak in report.axes] == ["x", "y", "z"]


def test_the_worst_axis_is_the_worst_of_the_three(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=600, duration=1e-3, system=system))
    sequence.add_block(pp.make_trapezoid("y", area=1000, duration=1e-3, system=system))

    for report in (
        safety.check_max_grad(sequence)[1],
        safety.check_max_slew(sequence)[1],
    ):
        worst = max(report.axes, key=lambda peak: peak.value)
        assert report.per_axis.value == worst.value
        assert report.per_axis.axis == worst.axis


def test_an_axis_carrying_nothing_reports_zero(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3, system=system))

    _, report = safety.check_max_grad(sequence)

    assert report.axes[0].value > 0
    assert [peak.value for peak in report.axes[1:]] == [0.0, 0.0]


def test_a_rotated_sequence_is_weighed_on_the_axes_it_plays(system):
    """What is stored on x is played over x and y, and weighed there.

    A rotation is a thing one playout does: it moves where a gradient is
    played without moving the row it is stored in. So the axis peaks are what
    the amplifiers are asked for, which is not what the rows say.
    """
    from scipy.spatial.transform import Rotation

    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", area=600, duration=1e-3, system=system)
    for turn in (0, 37, 90, 214):
        sequence.add_block(
            gradient, pp.make_rotation(Rotation.from_euler("z", turn, degrees=True))
        )

    _, grad = safety.check_max_grad(sequence)
    _, slew = safety.check_max_slew(sequence)

    assert [peak.value for peak in grad.axes] == pytest.approx(
        played_per_axis(sequence), rel=1e-9
    )
    # The turn spreads one row over two amplifiers without asking either for
    # more than the row holds, and asks the third for nothing at all.
    assert grad.per_axis.value == pytest.approx(grad.vector.value)
    assert grad.axes[2].value == pytest.approx(0.0, abs=1e-6)
    assert slew.axes[2].value == pytest.approx(0.0, abs=1e-6)


# -- the shape stored and the shape drawn ----------------------------------


def test_a_waveform_is_weighed_where_it_is_drawn_not_where_it_is_sampled(system):
    """A shape kept at raster centres reaches values none of its samples do.

    The samples say what the gradient is in the middle of each raster
    interval; the interpreter draws between the interval *boundaries*, which
    are half a raster away and follow from the samples rather than being any
    of them. A waveform that curves can pass outside every sample it holds,
    and it is the drawn waveform the amplifier plays.
    """
    waveform = 1e5 * np.sin(np.linspace(0, math.pi, 40)) ** 3
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_arbitrary_grad("x", waveform=waveform, system=system))

    drawn = sequence.waveforms()[0]
    _, grad = safety.check_max_grad(sequence)
    _, slew = safety.check_max_slew(sequence)

    # The drawn waveform really does overshoot what is stored, or this test
    # would hold whichever of the two were weighed.
    assert np.abs(drawn[1]).max() > np.abs(waveform).max() * 1.001

    assert grad.axes[0].value == pytest.approx(np.abs(drawn[1]).max(), rel=1e-12)
    assert slew.axes[0].value == pytest.approx(
        np.abs(np.diff(drawn[1]) / np.diff(drawn[0])).max(), rel=1e-12
    )


def test_an_extended_trapezoid_is_drawn_through_the_samples_it_names(system):
    """Times of its own means the samples *are* the corners, and stay so."""
    times = np.array([0.0, 2e-4, 6e-4, 8e-4])
    amplitudes = np.array([0.0, 8e4, 8e4, 0.0])
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_extended_trapezoid("x", amplitudes=amplitudes, times=times)
    )

    drawn = sequence.waveforms()[0]
    _, grad = safety.check_max_grad(sequence)

    assert grad.axes[0].value == pytest.approx(np.abs(drawn[1]).max(), rel=1e-12)
    assert grad.axes[0].value == pytest.approx(amplitudes.max())


# -- what the three axes ask for together ----------------------------------


def test_the_vector_peak_is_taken_where_the_axes_slew_at_once(system):
    """Peaks that do not happen together do not add up.

    One block ramping x and then, after it has finished, y. Combining the two
    peaks would say the amplifiers are asked for both at once; they never are.
    """
    ramp = 2e-4
    amplitude = 0.4 * system.max_grad
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_trapezoid(
            "x",
            amplitude=amplitude,
            rise_time=ramp,
            flat_time=0,
            fall_time=ramp,
            system=system,
        ),
        pp.make_trapezoid(
            "y",
            amplitude=amplitude,
            rise_time=ramp,
            flat_time=0,
            fall_time=ramp,
            delay=2 * ramp,
            system=system,
        ),
    )

    _, report = safety.check_max_slew(sequence)

    alone = amplitude / ramp
    assert report.per_axis.value == pytest.approx(alone)
    assert report.vector.value == pytest.approx(alone)
    assert report.vector.value < math.sqrt(2) * alone


def test_the_vector_peak_adds_up_axes_that_do_slew_at_once(system):
    ramp = 2e-4
    amplitude = 0.4 * system.max_grad
    sequence = pp.Sequence(system)
    sequence.add_block(
        *[
            pp.make_trapezoid(
                axis,
                amplitude=amplitude,
                rise_time=ramp,
                flat_time=0,
                fall_time=ramp,
                system=system,
            )
            for axis in ("x", "y")
        ]
    )

    _, report = safety.check_max_slew(sequence)

    assert report.vector.value == pytest.approx(math.sqrt(2) * amplitude / ramp)


def test_a_turned_block_asks_a_different_amplifier_to_slew(system):
    """A quarter turn about z plays what is stored on x on y instead."""
    from scipy.spatial.transform import Rotation

    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_trapezoid("x", area=600, duration=1e-3, system=system),
        pp.make_rotation(Rotation.from_euler("z", 90, degrees=True)),
    )

    _, report = safety.check_max_slew(sequence)

    assert report.axes[0].value == pytest.approx(0.0, abs=1e-6)
    assert report.axes[1].value > 0
    # How much is asked for between the amplifiers is what the turn leaves
    # alone; which one is asked for it is not.
    assert report.vector.value == pytest.approx(report.axes[1].value)


def test_a_turn_between_two_blocks_breaks_a_waveform_in_two(system):
    """The same ramp continued at a different rotation is a jump, not a ramp."""
    from scipy.spatial.transform import Rotation

    rising = np.linspace(0, 1e5, 20)
    straight = pp.Sequence(system)
    turned = pp.Sequence(system)
    for sequence, extras in (
        (straight, ()),
        (turned, (pp.make_rotation(Rotation.from_euler("z", 90, degrees=True)),)),
    ):
        sequence.add_block(
            pp.make_arbitrary_grad(
                "x", waveform=rising, first=0.0, last=1e5, system=system
            )
        )
        sequence.add_block(
            pp.make_arbitrary_grad(
                "x", waveform=rising[::-1], first=1e5, last=0.0, system=system
            ),
            *extras,
        )

    assert safety.check_grad_continuity(straight)[0]

    is_ok, report = safety.check_grad_continuity(turned)
    assert not is_ok
    # x is left at full amplitude and dropped; y is asked for it from nothing.
    assert sorted(jump.axis for jump in report.discontinuities) == [0, 1]
    assert report.ends_at_zero
