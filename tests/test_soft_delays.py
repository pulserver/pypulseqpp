"""Setting a soft delay to the value a scan is to be run at.

A soft delay says how a block's duration follows from a value the console
supplies -- `duration = value / factor + offset` -- so a TE or a TR is chosen
at the scanner rather than baked into the file. `apply_soft_delay` writes the
chosen values into the block durations, and `get_default_soft_delay_values`
reads back what each one stands for if nobody sets it.

Ported from the reference toolbox's `test_soft_delay.py`, as plain functions.
`apply_soft_delay` is upstream's own method run against this sequence: it
reads the blocks and writes their durations, which mean the same thing here,
so it is taken rather than rewritten.
"""

import numpy as np
import pypulseq as upstream
import pytest

import pypulseqpp as pp


@pytest.fixture
def system():
    return upstream.Opts()


def test_a_soft_delay_sets_how_long_its_block_lasts(system):
    """The default duration is the block's, with nothing else in it."""
    sequence = pp.Sequence(system)

    sequence.add_block(pp.make_soft_delay("TE", default_duration=7.5e-3))
    sequence.add_block(pp.make_soft_delay("TR", default_duration=150e-3))

    assert sequence.block_durations[1] == pytest.approx(7.5e-3)
    assert sequence.block_durations[2] == pytest.approx(150e-3)


def test_applying_a_value_by_name_sets_the_block_duration(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_soft_delay("TE", default_duration=5e-3))
    sequence.add_block(pp.make_soft_delay("TR", default_duration=100e-3))

    sequence.apply_soft_delay(TE=8e-3, TR=500e-3)

    assert sequence.block_durations[1] == pytest.approx(8e-3)
    assert sequence.block_durations[2] == pytest.approx(500e-3)


@pytest.mark.parametrize(
    ("factor", "offset", "applied"),
    [
        (-2.0, 0.1, 0.04),
        (1000.0, 0.0, 1e-3),
        (0.1, 0.0, 100.0),
        (1.0, 2e-3, 5e-3),
    ],
)
def test_a_value_becomes_the_duration_the_delay_says_it_does(
    system, factor, offset, applied
):
    """`duration = value / factor + offset`, rounded onto the block raster."""
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_soft_delay("D", factor=factor, offset=offset, default_duration=1e-3)
    )

    sequence.apply_soft_delay(D=applied)

    wanted = applied / factor + offset
    raster = system.block_duration_raster
    assert sequence.block_durations[1] == pytest.approx(round(wanted / raster) * raster)


def test_a_name_the_sequence_does_not_carry_is_refused(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_soft_delay("TE", default_duration=5e-3))

    with pytest.raises(ValueError, match="'TI' not found in sequence"):
        sequence.apply_soft_delay(TI=1.0)


def test_only_the_delays_named_are_moved(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_soft_delay("TE", default_duration=5e-3))
    sequence.add_block(pp.make_soft_delay("TR", default_duration=100e-3))

    sequence.apply_soft_delay(TE=8e-3)

    assert sequence.block_durations[1] == pytest.approx(8e-3)
    assert sequence.block_durations[2] == pytest.approx(100e-3)


def test_the_same_delay_moves_every_block_that_plays_it(system):
    sequence = pp.Sequence(system)
    pulse = pp.make_block_pulse(
        np.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    for _ in range(3):
        sequence.add_block(pulse)
        sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))

    sequence.apply_soft_delay(TE=9e-3)

    assert [sequence.block_durations[i] for i in (2, 4, 6)] == pytest.approx(
        [9e-3, 9e-3, 9e-3]
    )


def test_applying_a_delay_matches_upstream(system):
    """Held against the toolbox whose method this is."""

    def build(module):
        sequence = module.Sequence(system=system)
        pulse = module.make_block_pulse(
            np.pi / 2, duration=1e-3, system=system, use="excitation"
        )
        for _ in range(2):
            sequence.add_block(pulse)
            sequence.add_block(
                module.make_soft_delay("TE", numID=0, default_duration=5e-3)
            )
            sequence.add_block(
                module.make_soft_delay("TR", numID=1, default_duration=20e-3)
            )
        return sequence

    theirs, ours = build(upstream), build(pp)
    theirs.apply_soft_delay(TE=8e-3, TR=30e-3)
    ours.apply_soft_delay(TE=8e-3, TR=30e-3)

    assert list(ours.block_durations.values()) == pytest.approx(
        list(theirs.block_durations.values())
    )


# -- what a delay stands for if nobody sets it -----------------------------


def test_the_default_of_each_delay_is_read_back_by_name(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))
    sequence.add_block(pp.make_soft_delay("TR", numID=1, default_duration=20e-3))

    defaults, problems, _ = sequence.get_default_soft_delay_values()

    assert defaults == {"TE": pytest.approx(5e-3), "TR": pytest.approx(20e-3)}
    assert problems == []


def test_a_delay_whose_blocks_disagree_is_reported(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))
    sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))
    sequence.block_durations[2] = 9e-3

    _, problems, _ = sequence.get_default_soft_delay_values()

    assert len(problems) == 1
    assert "inconsistent with the previous default" in problems[0]


def test_the_range_a_value_may_take_keeps_the_block_positive(system):
    """A block cannot last less than nothing, so the offset bounds the value."""
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_soft_delay(
            "TE", numID=0, offset=-2e-3, factor=1.0, default_duration=5e-3
        )
    )

    _, _, limits = sequence.get_default_soft_delay_values()

    assert limits[0]["min"] == pytest.approx(2e-3)
    assert limits[0]["max"] == np.inf


def test_a_gap_in_the_numbering_is_warned_about(system):
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))
    sequence.add_block(pp.make_soft_delay("TR", numID=2, default_duration=20e-3))

    with pytest.warns(UserWarning, match="numeric ID 1 is unused"):
        defaults, _, _ = sequence.get_default_soft_delay_values()

    assert set(defaults) == {"TE", "TR"}


def test_applying_a_delay_is_a_pass_over_the_block_table(system):
    """The soft delays are found without decoding anything else."""
    sequence = pp.Sequence(system)
    pulse = pp.make_block_pulse(
        np.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    readout = pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
    for _ in range(500):
        sequence.add_block(pulse)
        sequence.add_block(readout)
        sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))

    sequence.apply_soft_delay(TE=9e-3)

    assert [sequence.block_durations[i] for i in (3, 300, 1500)] == pytest.approx(
        [9e-3, 9e-3, 9e-3]
    )


def test_a_delay_left_unnamed_by_a_block_that_has_none_is_untouched(system):
    """Only blocks heading a soft delay move."""
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_delay(3e-3))
    sequence.add_block(pp.make_soft_delay("TE", numID=0, default_duration=5e-3))

    sequence.apply_soft_delay(TE=9e-3)

    assert sequence.block_durations[1] == pytest.approx(3e-3)
    assert sequence.block_durations[2] == pytest.approx(9e-3)
