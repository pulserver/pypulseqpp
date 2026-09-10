"""Block validation; gradient joins are checked by check_timing, not insertion."""

import numpy as np
import pytest

import pypulseqpp as pp


@pytest.fixture
def system():
    """A scanner, with the raster spelled out.

    Whether a step between two blocks is a discontinuity is that step over the
    raster it is taken in, judged against the slew limit -- so a test about
    discontinuities that leaves the raster to whatever the defaults happen to
    be is a test about the defaults.
    """
    return pp.Opts(
        max_grad=50,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
    )


@pytest.fixture
def events(system):
    """The toolbox's own set: a trapezoid, a delay, and five shapes that do
    and do not begin and end at zero."""

    def extended(amplitudes, times):
        return pp.make_extended_trapezoid(
            "x", amplitudes=np.array(amplitudes, float), times=np.array(times, float)
        )

    return {
        "trap": pp.make_trapezoid("x", area=1000, duration=1e-3, system=system),
        "extended": extended([0, 1e5, 0], [0, 1e-4, 2e-4]),
        "extended_delay": extended([0, 1e5, 0], [1e-4, 2e-4, 3e-4]),
        "ends_high": extended([0, 1e5, 1e5], [0, 1e-4, 2e-4]),
        "starts_high": extended([1e5, 1e5, 0], [0, 1e-4, 2e-4]),
        "starts_higher": extended([2e5, 1e5, 0], [0, 1e-4, 2e-4]),
        "all_high": extended([1e5, 1e5, 1e5], [0, 1e-4, 2e-4]),
        "delay": pp.make_delay(1e-3),
    }


def built(system, events, names):
    sequence = pp.Sequence(system)
    for name in names:
        sequence.add_block(events[name])
    return sequence


def reported(sequence):
    """The kinds of fault a timing check finds, and the blocks they are in."""
    return [
        (finding.error_type, finding.block) for finding in sequence.check_timing()[1]
    ]


# -- what a block takes ----------------------------------------------------


def test_a_block_of_nothing_is_refused(system):
    """Upstream PyPulseq is the API this stands in for, and it refuses None.

    The toolbox takes it and adds no block; upstream raises, so this raises
    too -- with a message that says what was expected instead.
    """
    with pytest.raises(ValueError, match="event objects"):
        pp.Sequence(system).add_block(None)


def test_a_block_that_is_partly_nothing_is_refused_too(system):
    with pytest.raises(ValueError, match="event objects"):
        pp.Sequence(system).add_block(None, pp.make_delay(1.0), None)


# -- gradients that join up, and gradients that do not ---------------------


@pytest.mark.parametrize(
    ("blocks", "faults"),
    [
        # A trapezoid begins and ends at zero, so anything starting and ending
        # at zero follows it and is followed by it.
        (("trap", "extended", "trap"), []),
        (("extended_delay",), []),
        # Two halves of one waveform, split across a join.
        (("ends_high", "starts_high"), []),
        # A gradient that starts away from where the axis was left.
        (("trap", "starts_high"), [("GRADIENT_DISCONTINUITY", 2)]),
        (("starts_high",), [("GRADIENT_DISCONTINUITY", 1)]),
        (("delay", "starts_high"), [("GRADIENT_DISCONTINUITY", 2)]),
        # ...including where it is the same waveform at another amplitude.
        (("ends_high", "starts_higher"), [("GRADIENT_DISCONTINUITY", 2)]),
        # An axis left on, and then dropped by whatever comes next.
        (("ends_high", "delay"), [("GRADIENT_DISCONTINUITY", 2)]),
        (("ends_high", "trap"), [("GRADIENT_DISCONTINUITY", 2)]),
        # An axis left on at the end of the scan is never ramped down.
        (("ends_high",), [("GRADIENT_NOT_RAMPED_DOWN", 1)]),
        (
            ("delay", "all_high"),
            [("GRADIENT_DISCONTINUITY", 2), ("GRADIENT_NOT_RAMPED_DOWN", 2)],
        ),
    ],
    ids=lambda value: "+".join(value) if value and isinstance(value[0], str) else None,
)
def test_a_gradient_is_judged_against_where_the_axis_was_left(
    system, events, blocks, faults
):
    sequence = built(system, events, blocks)

    assert reported(sequence) == faults
    assert sequence.check_timing()[0] == (not faults)


# -- rewriting a block moves both of its joins -----------------------------


@pytest.fixture
def three_delays(system, events):
    return built(system, events, ("delay", "delay", "delay"))


@pytest.fixture
def joined_up(system, events):
    """Three blocks that carry one waveform between them, and pass."""
    sequence = built(system, events, ("ends_high", "all_high", "starts_high"))
    assert sequence.check_timing()[0]
    return sequence


@pytest.mark.parametrize("index", [1, 2, 3])
def test_a_gradient_set_among_delays_starts_where_nothing_left_it(
    three_delays, events, index
):
    three_delays.set_block(index, events["starts_high"])

    assert reported(three_delays) == [("GRADIENT_DISCONTINUITY", index)]


def test_emptying_a_block_breaks_the_waveform_on_both_sides(joined_up, events):
    joined_up.set_block(2, events["delay"])

    assert reported(joined_up) == [
        ("GRADIENT_DISCONTINUITY", 2),
        ("GRADIENT_DISCONTINUITY", 3),
    ]


def test_a_block_that_joins_on_one_side_only_is_reported_on_the_other(
    joined_up, events
):
    """`starts_high` ends at zero where the block after it starts high."""
    joined_up.set_block(2, events["starts_high"])

    assert reported(joined_up) == [("GRADIENT_DISCONTINUITY", 3)]


def test_a_block_can_only_be_set_where_a_block_is(joined_up, events):
    """The toolbox numbers blocks in a dict and will leave gaps; here the
    block table is a table, and a row has to exist to be written."""
    with pytest.raises(ValueError, match="outside"):
        joined_up.set_block(6, events["starts_high"])
