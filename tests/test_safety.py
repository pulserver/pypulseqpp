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
    return pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


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

    is_ok, report = safety.check_max_slew(sequence)

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

    is_ok, report = safety.check_max_slew(sequence)

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

    is_ok, report = safety.check_max_slew(sequence)

    assert not report.ends_at_zero
    assert not is_ok


def test_trapezoids_begin_and_end_at_zero_so_never_jump(system):
    sequence = pp.Sequence(system)
    for _ in range(5):
        sequence.add_block(
            pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
        )

    _, report = safety.check_max_slew(sequence)

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

    _, report = safety.check_max_slew(sequence)

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


def test_a_rotated_sequence_plays_axes_its_stored_rows_do_not_name(system):
    """What is weighed is stored on x; what is played is spread over x and y.

    Which is why a report on what the scan does reads the waveforms rather
    than these peaks: a rotation is a thing one playout does, so it moves
    where a gradient is played without moving the row it is stored in.
    """
    from scipy.spatial.transform import Rotation

    sequence = pp.Sequence(system)
    gradient = pp.make_trapezoid("x", area=600, duration=1e-3, system=system)
    for quarter in range(4):
        sequence.add_block(
            gradient,
            pp.make_rotation(Rotation.from_euler("z", 90 * quarter, degrees=True)),
        )

    _, report = safety.check_max_grad(sequence)
    played = played_per_axis(sequence)

    assert report.axes[1].value == 0
    assert played[1] == pytest.approx(report.axes[0].value, rel=1e-9)
