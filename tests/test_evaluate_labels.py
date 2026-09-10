"""Sticky label state, evolution modes and scalar return conventions."""

import numpy as np
import pytest

import pypulseqpp as pp

toolbox = pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the answer; see reference.py",
)

EVOLUTIONS = ("none", "blocks", "adc", "label")


def both(build):
    """The same sequence, built here and by the toolbox."""
    return build(pp), build(toolbox)


def counted(module):
    """Three lines, each setting LIN, incrementing SLC and acquiring."""
    system = module.Opts()
    sequence = pp.Sequence(system) if module is pp else module.Sequence()
    for line in range(3):
        sequence.add_block(
            module.make_label(type="SET", label="LIN", value=line),
            module.make_label(type="INC", label="SLC", value=1),
            module.make_adc(64, dwell=1e-5, system=system),
            module.make_delay(1e-3),
        )
    return sequence


def sparsely_labelled(module):
    """A label set on one block among several that carry none."""
    system = module.Opts()
    sequence = pp.Sequence(system) if module is pp else module.Sequence()
    sequence.add_block(module.make_delay(1e-3))
    sequence.add_block(
        module.make_label(type="SET", label="REP", value=7),
        module.make_delay(1e-3),
    )
    sequence.add_block(module.make_adc(64, dwell=1e-5, system=system))
    sequence.add_block(module.make_delay(1e-3))
    return sequence


def assert_same(ours, theirs):
    assert set(ours) == set(theirs)
    for name in ours:
        np.testing.assert_array_equal(
            np.atleast_1d(ours[name]), np.atleast_1d(theirs[name]), err_msg=name
        )


@pytest.mark.parametrize(
    "build", [counted, sparsely_labelled], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("evolution", EVOLUTIONS)
def test_the_answer_is_the_toolboxs(build, evolution):
    ours, theirs = both(build)

    assert_same(
        ours.evaluate_labels(evolution=evolution),
        theirs.evaluate_labels(evolution=evolution),
    )


def test_without_an_evolution_a_label_is_one_number():
    """`labels['LIN'] == 4`, not `labels['LIN'] == array([4])`."""
    found = counted(pp).evaluate_labels()

    assert found["LIN"] == 2
    assert found["SLC"] == 3
    assert np.ndim(found["LIN"]) == 0


def test_a_label_is_recorded_where_it_is_asked_for():
    """Four ways of asking, and they do not record the same number of points."""
    sequence = sparsely_labelled(pp)

    assert len(np.atleast_1d(sequence.evaluate_labels(evolution="blocks")["REP"])) == 4
    assert np.ndim(sequence.evaluate_labels()["REP"]) == 0


def test_one_point_is_not_an_evolution():
    """The toolbox's own rule: asked for an evolution it does not have, it
    hands back what the labels finish at instead.

    A sequence with one ADC records one point under ``adc``, and a sequence
    with none records nothing at all -- and either way what comes back is the
    final value, not an array.
    """
    sequence = sparsely_labelled(pp)
    theirs = sparsely_labelled(toolbox)

    assert np.ndim(sequence.evaluate_labels(evolution="adc")["REP"]) == 0
    assert np.ndim(theirs.evaluate_labels(evolution="adc")["REP"]) == 0

    unacquired = pp.Sequence(pp.Opts())
    unacquired.add_block(
        pp.make_label(type="SET", label="REP", value=7), pp.make_delay(1e-3)
    )

    assert unacquired.evaluate_labels(evolution="adc") == {"REP": 7}


def test_a_label_reads_as_zero_before_it_is_set():
    sequence = sparsely_labelled(pp)

    np.testing.assert_array_equal(
        sequence.evaluate_labels(evolution="blocks")["REP"], [0, 7, 7, 7]
    )


def test_what_a_label_starts_at_can_be_given():
    """Which is what evaluating a sequence a piece at a time needs."""
    ours, theirs = both(counted)

    assert_same(
        ours.evaluate_labels(init={"LIN": 10, "REP": 5}),
        theirs.evaluate_labels(init={"LIN": 10, "REP": 5}),
    )


def test_a_label_named_only_in_init_is_reported_unchanged():
    found = counted(pp).evaluate_labels(init={"REP": 5})

    assert found["REP"] == 5


def test_only_the_blocks_asked_for_are_walked():
    sequence = counted(pp)

    assert sequence.evaluate_labels(block_range=[1, 2])["LIN"] == 1
    assert sequence.evaluate_labels(block_range=[2, 3])["LIN"] == 2
    np.testing.assert_array_equal(
        sequence.evaluate_labels(evolution="blocks", block_range=[1, 2])["LIN"], [0, 1]
    )


def test_a_sequence_using_no_label_reports_none():
    system = pp.Opts()
    sequence = pp.Sequence(system)
    sequence.add_block(pp.make_delay(1e-3))

    assert sequence.evaluate_labels() == {}
    assert sequence.evaluate_labels(evolution="blocks") == {}


def test_an_evolution_that_is_not_one_is_refused():
    with pytest.raises(ValueError, match="evolution must be one of"):
        counted(pp).evaluate_labels(evolution="every other Tuesday")


def test_the_block_count_reads_as_a_count():
    sequence = counted(pp)

    assert sequence.num_blocks == 3
    assert sequence.num_blocks == len(sequence)
    assert sequence.num_blocks == len(sequence.block_events)
